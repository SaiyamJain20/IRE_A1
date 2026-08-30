import argparse
import os
import polars as pl
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from src.evaluation.metrics import auc_score, mrr_score, ndcg_k, intra_list_diversity, novelty, coverage, bootstrap_ci, calculate_recall
from src.models.semantic_retriever import SemanticRetriever
from src.models.bm25_retriever import BM25Retriever
from src.models.query_generator import build_queries_for_users

def score_semantic_candidates(retriever, history_clicks, candidate_ids):
    user_emb = retriever.get_user_representation(history_clicks)
    scores = []
    for cid in candidate_ids:
        if user_emb is not None and cid in retriever.article_embeddings:
            cand_emb = retriever.article_embeddings[cid]
            score = float(np.dot(user_emb, cand_emb))
        else:
            score = 0.0
        scores.append(score)
    return scores

def score_bm25_candidates_vectorized(retriever, query, candidate_ids, article_id_to_idx):
    if not query or not retriever.bm25:
        return [0.0] * len(candidate_ids)
        
    tokenized_query = set(retriever._tokenize(query))
    bm25 = retriever.bm25
    
    # Map candidate IDs to corpus indices once
    cand_indices = [article_id_to_idx.get(str(cid)) for cid in candidate_ids]
    
    scores = np.zeros(len(candidate_ids))
    
    for q in tokenized_query:
        if q not in retriever.inverted_index:
            continue
        doc_dict = retriever.inverted_index[q]
        idf = bm25.idf.get(q, 0.0)
        
        # Batch: find which candidates appear in this term's posting list
        match_positions = []
        match_freqs = []
        match_doc_lens = []
        for i, idx in enumerate(cand_indices):
            if idx is not None and idx in doc_dict:
                match_positions.append(i)
                match_freqs.append(doc_dict[idx])
                match_doc_lens.append(bm25.doc_len[idx])
        
        if not match_positions:
            continue
            
        pos = np.array(match_positions)
        freqs = np.array(match_freqs, dtype=np.float64)
        doc_lens = np.array(match_doc_lens, dtype=np.float64)
        
        numerator = freqs * (bm25.k1 + 1)
        denominator = freqs + bm25.k1 * (1 - bm25.b + bm25.b * doc_lens / bm25.avgdl)
        scores[pos] += idf * (numerator / denominator)
                
    return scores.tolist()

def parse_mind_impressions(imp_str):
    if not imp_str: return [], []
    cands, labels = [], []
    for x in imp_str.split(" "):
        parts = x.split("-")
        if len(parts) == 2:
            cands.append(parts[0])
            labels.append(int(parts[1]))
    return cands, labels

def evaluate_setting(dataset, model_type, feature_store, no_future_features=False):
    print(f"\n{'='*50}", flush=True)
    print(f"Evaluating: Dataset={dataset} | Model={model_type} | No Future Features={no_future_features}", flush=True)
    print(f"{'='*50}", flush=True)
    
    dataset_dir = os.path.join(feature_store, dataset)
    if not os.path.exists(dataset_dir):
        print(f"Dataset path {dataset_dir} not found. Skipping.")
        return None
        
    articles_df = pl.read_parquet(os.path.join(dataset_dir, "articles.parquet"))
    
    # Build article category lookup for slicing
    cat_dict = dict(zip(
        [str(x) for x in articles_df.get_column("article_id").to_list()],
        articles_df.get_column("category").fill_null("unknown").to_list()
    ))
    
    if no_future_features:
        print("Applying Serving-Time Constraints: Dropping 'abstract' and 'body' features")
        articles_df = articles_df.with_columns(
            pl.lit("").alias("abstract"),
            pl.lit("").alias("body")
        )
        
    val_df = pl.read_parquet(os.path.join(dataset_dir, "val", "behaviors.parquet"))
    
    article_id_to_idx = {}
    sem_retriever = None
    bm25_retriever = None
    
    if model_type in ["semantic", "hybrid"]:
        print("Loading Semantic Retriever...", flush=True)
        model_name = "paraphrase-multilingual-MiniLM-L12-v2" if dataset == "EBNERD" else "all-MiniLM-L6-v2"
        sem_retriever = SemanticRetriever(model_name=model_name, dataset=dataset)
        emb_cache = os.path.join(dataset_dir, f"embeddings_{model_name.replace('/', '_')}.npy")
        sem_retriever.fit(articles_df, embedding_cache_path=emb_cache)
        
    if model_type in ["bm25", "hybrid"]:
        lang = "da" if dataset == "EBNERD" else "en"
        bm25_retriever = BM25Retriever(language=lang)
        bm25_retriever.fit(articles_df)
        # Extract IDF dict from BM25 for smarter query construction
        idf_dict = bm25_retriever.bm25.idf if bm25_retriever.bm25 else {}
        print("Constructing queries from click history (with IDF + category boosting)...")
        val_df = build_queries_for_users(val_df, articles_df, idf_dict=idf_dict, dataset=dataset)
        article_id_to_idx = {str(aid): i for i, aid in enumerate(bm25_retriever.article_ids)}
        
    print("Evaluating Impressions...")
    
    y_trues, y_scores, recommended_lists = [], [], []
    history_lengths = []
    
    for row in tqdm(val_df.iter_rows(named=True), total=len(val_df)):
        history = row.get("history_article_ids")
        if hasattr(history, "to_list"): history = history.to_list()
        elif hasattr(history, "tolist"): history = history.tolist()
        elif not isinstance(history, list): history = list(history) if history is not None else []
        history_clicks = [str(x) for x in history][-20:]
        
        history_lengths.append(len(history))
        
        cands, labels = [], []
        if dataset == "MIND":
            imp_str = row.get("impressions")
            cands, labels = parse_mind_impressions(imp_str)
        else:
            inview = row.get("article_ids_inview")
            clicked = row.get("article_ids_clicked")
            if inview and clicked:
                if hasattr(inview, "to_list"): inview = inview.to_list()
                if hasattr(clicked, "to_list"): clicked = clicked.to_list()
                cands = [str(x) for x in inview]
                clicked_set = set(str(x) for x in clicked)
                labels = [1 if c in clicked_set else 0 for c in cands]
                
        if not cands or sum(labels) == 0:
            continue
            
        if model_type == "semantic":
            scores = score_semantic_candidates(sem_retriever, history_clicks, cands)
        elif model_type == "bm25":
            query = row.get("generated_query")
            scores = score_bm25_candidates_vectorized(bm25_retriever, query, cands, article_id_to_idx)
        elif model_type == "hybrid":
            query = row.get("generated_query")
            b_scores = score_bm25_candidates_vectorized(bm25_retriever, query, cands, article_id_to_idx)
            s_scores = score_semantic_candidates(sem_retriever, history_clicks, cands)
            
            b_min, b_max = min(b_scores), max(b_scores)
            s_min, s_max = min(s_scores), max(s_scores)
            
            b_norm = [(s - b_min) / (b_max - b_min) if b_max > b_min else 0.0 for s in b_scores]
            s_norm = [(s - s_min) / (s_max - s_min) if s_max > s_min else 0.0 for s in s_scores]
            
            # Weight semantic higher because BM25 is very weak on EBNERD
            alpha = 0.1 if dataset == "EBNERD" else 0.5
            scores = [alpha * b + (1 - alpha) * s for b, s in zip(b_norm, s_norm)]
            
        y_trues.append(labels)
        y_scores.append(scores)
        top_indices = np.argsort(scores)[::-1][:5]
        recommended_lists.append([cands[i] for i in top_indices])

    print("\n--- Accuracy Metrics ---", flush=True)
    auc_list = [auc_score(yt, ys) for yt, ys in zip(y_trues, y_scores) if len(np.unique(yt)) > 1]
    mrr_list = [mrr_score(yt, ys) for yt, ys in zip(y_trues, y_scores)]
    ndcg5_list = [ndcg_k(yt, ys, 5) for yt, ys in zip(y_trues, y_scores)]
    ndcg10_list = [ndcg_k(yt, ys, 10) for yt, ys in zip(y_trues, y_scores)]
    
    print(f"AUC:      {np.mean(auc_list):.4f}")
    print(f"MRR:      {np.mean(mrr_list):.4f}")
    print(f"nDCG@5:   {np.mean(ndcg5_list):.4f}")
    print(f"nDCG@10:  {np.mean(ndcg10_list):.4f}")
    
    print("\n--- Beyond-Accuracy Metrics ---", flush=True)
    total_items = len(articles_df)
    
    pop_dict = {}
    for row in val_df.iter_rows(named=True):
        clicks = []
        if dataset == "MIND":
            _, clicks = parse_mind_impressions(row.get("impressions"))
        else:
            clicks = row.get("article_ids_clicked")
            if hasattr(clicks, "to_list"): clicks = clicks.to_list()
        for c in (clicks or []):
            pop_dict[str(c)] = pop_dict.get(str(c), 0) + 1
            
    cov = coverage(recommended_lists, total_items)
    nov = novelty(recommended_lists, pop_dict, total_users=len(val_df))
    
    embs_dict = {}
    if sem_retriever:
        embs_dict = getattr(sem_retriever, 'article_embeddings', {})
    else:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2" if dataset == "EBNERD" else "all-MiniLM-L6-v2"
        emb_cache = os.path.join(dataset_dir, f"embeddings_{model_name.replace('/', '_')}.npy")
        if os.path.exists(emb_cache):
            cached_embs = np.load(emb_cache)
            embs_dict = {str(aid): emb for aid, emb in zip(articles_df.get_column("article_id").to_list(), cached_embs)}
            
    ild = intra_list_diversity(recommended_lists, embs_dict) if embs_dict else 0.0
    
    print(f"Coverage: {cov:.4f}")
    print(f"Novelty:  {nov:.4f}")
    print(f"ILD:      {ild:.4f}")
    
    print("\n--- Data Slicing (Quartile History Lengths) ---")
    hist_arr = np.array(history_lengths[:len(y_trues)])
    if len(hist_arr) > 0:
        p25, p75 = np.percentile(hist_arr, 25), np.percentile(hist_arr, 75)
        short_hist_idx = np.where(hist_arr <= p25)[0]
        long_hist_idx = np.where(hist_arr >= p75)[0]
        
        if len(short_hist_idx) > 0:
            short_mrr = np.mean([mrr_list[i] for i in short_hist_idx])
            print(f"Short-History MRR (<= {p25:.1f}): {short_mrr:.4f} ({len(short_hist_idx)} users)")
        if len(long_hist_idx) > 0:
            long_mrr = np.mean([mrr_list[i] for i in long_hist_idx])
            print(f"Long-History MRR (>= {p75:.1f}): {long_mrr:.4f} ({len(long_hist_idx)} users)")

    print("\n--- Data Slicing (Head vs. Tail Articles) ---")
    # Head = top 10% most popular articles, Tail = bottom 50%
    if pop_dict:
        sorted_pops = sorted(pop_dict.values())
        head_threshold = sorted_pops[int(len(sorted_pops) * 0.9)] if sorted_pops else 0
        tail_threshold = sorted_pops[int(len(sorted_pops) * 0.5)] if sorted_pops else 0
        head_articles = {aid for aid, cnt in pop_dict.items() if cnt >= head_threshold}
        tail_articles = {aid for aid, cnt in pop_dict.items() if cnt <= tail_threshold}
        
        head_mrrs, tail_mrrs = [], []
        for i, (yt, ys) in enumerate(zip(y_trues, y_scores)):
            # Check if any ground-truth positive is a head or tail article
            top_cands = recommended_lists[i] if i < len(recommended_lists) else []
            pos_cands = [recommended_lists[i][j] for j in range(min(len(yt), len(top_cands))) if j < len(yt) and yt[j] == 1] if top_cands else []
            if any(c in head_articles for c in pos_cands):
                head_mrrs.append(mrr_list[i])
            if any(c in tail_articles for c in pos_cands):
                tail_mrrs.append(mrr_list[i])
        
        if head_mrrs:
            print(f"Head-Article MRR (top 10% popular): {np.mean(head_mrrs):.4f} ({len(head_mrrs)} impressions)")
        if tail_mrrs:
            print(f"Tail-Article MRR (bottom 50% popular): {np.mean(tail_mrrs):.4f} ({len(tail_mrrs)} impressions)")

    print("\n--- Confidence Intervals (95% Bootstrap) ---", flush=True)
    print("Calculating Bootstrap CIs... (this will be instant)", flush=True)
    
    def fast_bootstrap_ci(metric_arr, n_bootstraps=1000, ci=95):
        if not len(metric_arr): return 0.0, 0.0
        rng = np.random.RandomState(42)
        indices = rng.randint(0, len(metric_arr), (n_bootstraps, len(metric_arr)))
        means = np.mean(np.array(metric_arr)[indices], axis=1)
        return np.percentile(means, (100-ci)/2.0), np.percentile(means, 100-(100-ci)/2.0)

    for metric_name, metric_list in [
        ("AUC", auc_list),
        ("MRR", mrr_list),
        ("nDCG@5", ndcg5_list),
        ("nDCG@10", ndcg10_list),
    ]:
        try:
            lower, upper = fast_bootstrap_ci(metric_list, n_bootstraps=1000)
            print(f"{metric_name} 95% CI: [{lower:.4f}, {upper:.4f}]", flush=True)
        except Exception as e:
            print(f"{metric_name} CI computation failed: {e}", flush=True)
            
    print("\n--- Recall Evaluation (Full Corpus) ---")
    k_values = [50, 100, 200]
    
    def evaluate_single_recall_row(row_dict):
        gt_col = "impressions" if dataset == "MIND" else "article_ids_clicked"
        gt_clicks = []
        if dataset == "MIND":
            imp_str = row_dict.get(gt_col)
            if imp_str:
                gt_clicks = [x.split("-")[0] for x in imp_str.split(" ") if x.split("-")[1] == '1']
        else:
            clicks = row_dict.get(gt_col)
            if hasattr(clicks, "to_list"): gt_clicks = clicks.to_list()
            elif isinstance(clicks, list): gt_clicks = clicks
            else: gt_clicks = list(clicks) if clicks is not None else []
            
        gt_clicks = [str(x) for x in gt_clicks]
        if not gt_clicks:
            return None
            
        query = row_dict.get("generated_query") if model_type == "bm25" else row_dict.get("history_article_ids")
        if model_type != "bm25" and query is not None:
             if hasattr(query, "to_list"): query = query.to_list()
             elif not isinstance(query, list): query = list(query)
             query = [str(x) for x in query][-20:]
             
        if not query:
            return None
            
        retriever = sem_retriever if model_type in ["semantic", "hybrid"] else bm25_retriever
        retrieved = retriever.retrieve(query, top_k=max(k_values))
        return {k: calculate_recall(retrieved, gt_clicks, k) for k in k_values}

    print(f"Evaluating {model_type.upper()} Recall (parallelized)...")
    rows_to_process = [row for row in val_df.iter_rows(named=True)]
    
    num_cores = min(multiprocessing.cpu_count(), 8)
    try:
        recall_results = Parallel(n_jobs=num_cores, backend="threading")(
            delayed(evaluate_single_recall_row)(r) for r in tqdm(rows_to_process)
        )
    except Exception as e:
        print(f"Parallel recall failed ({e}), falling back to sequential...")
        recall_results = [evaluate_single_recall_row(r) for r in tqdm(rows_to_process)]
    
    agg_recall = {k: [] for k in k_values}
    for res in recall_results:
        if res:
            for k in k_values:
                agg_recall[k].append(res[k])
                
    for k in k_values:
        mean_recall = np.mean(agg_recall[k]) if agg_recall[k] else 0.0
        print(f"Recall@{k}: {mean_recall:.4f}")
    
    # Return metrics dict for comparative summary
    return {
        "dataset": dataset, "model": model_type,
        "AUC": np.mean(auc_list), "MRR": np.mean(mrr_list),
        "nDCG@5": np.mean(ndcg5_list), "nDCG@10": np.mean(ndcg10_list),
        "Coverage": cov, "Novelty": nov, "ILD": ild,
    }

def main():
    parser = argparse.ArgumentParser(description="Offline Evaluation Harness")
    parser.add_argument("--feature_store", type=str, default="feature_store")
    parser.add_argument("--dataset", type=str, default="all", choices=["EBNERD", "MIND", "all"])
    parser.add_argument("--model_type", type=str, default="all", choices=["semantic", "bm25", "hybrid", "all"])
    parser.add_argument("--no_future_features", action="store_true", help="Run ablation study without future features like abstracts")
    args = parser.parse_args()
    
    datasets = ["EBNERD", "MIND"] if args.dataset == "all" else [args.dataset]
    models = ["semantic", "bm25", "hybrid"] if args.model_type == "all" else [args.model_type]
    
    all_results = []
    for d in datasets:
        for m in models:
            result = evaluate_setting(d, m, args.feature_store, args.no_future_features)
            if result:
                all_results.append(result)
    
    # Print comparative summary table if multiple runs completed
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("COMPARATIVE SUMMARY: Lexical vs Semantic vs Hybrid")
        print(f"{'='*80}")
        header = f"{'Dataset':<10} {'Model':<10} {'AUC':>7} {'MRR':>7} {'nDCG@5':>8} {'nDCG@10':>9} {'Cov':>7} {'Nov':>7} {'ILD':>7}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(f"{r['dataset']:<10} {r['model']:<10} {r['AUC']:>7.4f} {r['MRR']:>7.4f} {r['nDCG@5']:>8.4f} {r['nDCG@10']:>9.4f} {r['Coverage']:>7.4f} {r['Novelty']:>7.2f} {r['ILD']:>7.4f}")
        print()

if __name__ == "__main__":
    main()
