import argparse
import os
import polars as pl
import numpy as np
from tqdm import tqdm
from src.models.semantic_retriever import SemanticRetriever

def parse_mind_impressions(imp_str):
    if not imp_str: return []
    return [x.split("-")[0] for x in imp_str.split(" ")]

def main():
    parser = argparse.ArgumentParser(description="Generate Codabench Predictions")
    parser.add_argument("--feature_store", type=str, default="feature_store")
    parser.add_argument("--dataset", type=str, default="MIND", choices=["EBNERD", "MIND", "MINDlarge"])
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which dataset split to generate predictions for (Codabench requires 'test')")
    args = parser.parse_args()
    
    dataset_dir = os.path.join(args.feature_store, args.dataset)
    print(f"Loading {args.dataset} data from {dataset_dir}...")
    
    articles_df = pl.read_parquet(os.path.join(dataset_dir, "articles.parquet"))
    
    print(f"Loading {args.split} behaviors...")
    val_df = pl.read_parquet(os.path.join(dataset_dir, args.split, "behaviors.parquet"))
    
    from src.models.bm25_retriever import BM25Retriever
    from src.models.query_generator import build_queries_for_users
    from src.pipeline.evaluate import score_semantic_candidates, score_bm25_candidates_vectorized
    
    sem_retriever = None
    bm25_retriever = None
    article_id_to_idx = {}
    
    if args.model_type in ["semantic", "hybrid"]:
        print("Loading Semantic Retriever...")
        model_name = "paraphrase-multilingual-MiniLM-L12-v2" if args.dataset == "EBNERD" else "all-MiniLM-L6-v2"
        sem_retriever = SemanticRetriever(model_name=model_name, dataset=args.dataset)
        emb_cache = os.path.join(dataset_dir, f"embeddings_{model_name.replace('/', '_')}.npy")
        sem_retriever.fit(articles_df, embedding_cache_path=emb_cache)
        
    if args.model_type in ["bm25", "hybrid"]:
        print("Loading BM25 Retriever...")
        lang = "da" if args.dataset == "EBNERD" else "en"
        bm25_retriever = BM25Retriever(language=lang)
        bm25_retriever.fit(articles_df)
        idf_dict = bm25_retriever.bm25.idf if bm25_retriever.bm25 else {}
        print("Constructing queries from click history...")
        val_df = build_queries_for_users(val_df, articles_df, idf_dict=idf_dict, dataset=args.dataset)
        article_id_to_idx = {str(aid): i for i, aid in enumerate(bm25_retriever.article_ids)}
    
    os.makedirs("outputs", exist_ok=True)
    output_file = os.path.join("outputs", f"predictions_{args.dataset.lower()}.txt")
    print(f"Generating predictions to {output_file}...")
    
    # Fast history lookup for EBNERD to avoid OOM
    history_dict = {}
    history_path = os.path.join(dataset_dir, args.split, "history.parquet")
    if os.path.exists(history_path):
        print(f"Loading user history dict from {history_path}...")
        hist_df = pl.read_parquet(history_path)
        # Use column names we know exist in history.parquet
        user_col = "user_id"
        hist_col = "article_id_fixed"
        if hist_col in hist_df.columns:
            # We want lists, which we can cast to string lists later
            history_dict = dict(zip(hist_df[user_col], hist_df[hist_col]))
    
    with open(output_file, "w") as f:
        for row in tqdm(val_df.iter_rows(named=True), total=len(val_df)):
            imp_id = row.get("impression_id")
            user_id = row.get("user_id")
            
            history = row.get("history_article_ids")
            if not history and user_id in history_dict:
                history = history_dict[user_id]
                
            if hasattr(history, "to_list"): history = history.to_list()
            elif hasattr(history, "tolist"): history = history.tolist()
            elif not isinstance(history, list): history = list(history) if history is not None else []
            history_clicks = [str(x) for x in history][-20:]
            
            cands = []
            if "MIND" in args.dataset:
                cands = parse_mind_impressions(row.get("impressions"))
            else:
                inview = row.get("article_ids_inview")
                if hasattr(inview, "to_list"): inview = inview.to_list()
                if inview:
                    cands = [str(x) for x in inview]
            
            if not cands:
                continue
                
            if args.model_type == "semantic":
                scores = score_semantic_candidates(sem_retriever, history_clicks, cands)
            elif args.model_type == "bm25":
                query = row.get("generated_query")
                scores = score_bm25_candidates_vectorized(bm25_retriever, query, cands, article_id_to_idx)
            elif args.model_type == "hybrid":
                query = row.get("generated_query")
                b_scores = score_bm25_candidates_vectorized(bm25_retriever, query, cands, article_id_to_idx)
                s_scores = score_semantic_candidates(sem_retriever, history_clicks, cands)
                
                b_min, b_max = min(b_scores), max(b_scores)
                s_min, s_max = min(s_scores), max(s_scores)
                
                b_norm = [(s - b_min) / (b_max - b_min) if b_max > b_min else 0.0 for s in b_scores]
                s_norm = [(s - s_min) / (s_max - s_min) if s_max > s_min else 0.0 for s in s_scores]
                
                # Weight semantic higher because BM25 is very weak on EBNERD
                alpha = 0.1 if args.dataset == "EBNERD" else 0.5
                scores = [alpha * b + (1 - alpha) * s for b, s in zip(b_norm, s_norm)]
            
            # Rank candidates (highest score gets rank 1)
            # Argsort sorts ascending, so we do [::-1] to get descending indices
            descending_indices = np.argsort(scores)[::-1]
            
            # Create a ranks array where the position of the candidate corresponds to its rank (1-indexed)
            ranks = [0] * len(cands)
            for rank, idx in enumerate(descending_indices):
                ranks[idx] = rank + 1
                
            ranks_str = "[" + ",".join(map(str, ranks)) + "]"
            f.write(f"{imp_id} {ranks_str}\n")
            
    print(f"Predictions successfully written to {output_file}")
    
    import zipfile
    zip_filename = os.path.join("outputs", f"predictions_{args.dataset.lower()}_{args.split}.zip")
    print(f"Creating zip file for submission: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Codabench MIND requests 'prediction.txt', EBNERD requests 'predictions.txt'
        arcname = 'prediction.txt' if "MIND" in args.dataset else 'predictions.txt'
        zipf.write(output_file, arcname=arcname)
    print("Done! You can now submit this zip file to Codabench.")

if __name__ == "__main__":
    main()
