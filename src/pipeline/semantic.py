import polars as pl
import argparse
import os
from tqdm import tqdm
from src.models.semantic_retriever import SemanticRetriever
from joblib import Parallel, delayed
import multiprocessing

def calculate_recall(retrieved_items, ground_truth_items, k):
    """Calculate recall@K"""
    if not ground_truth_items:
        return 0.0
    
    retrieved_k = set(retrieved_items[:k])
    gt_set = set(ground_truth_items)
    
    hits = len(retrieved_k.intersection(gt_set))
    return hits / len(gt_set)

def evaluate_semantic_recall(behaviors_df, retriever, k_values=[50, 100, 200]):
    """Evaluate semantic recall for different K values across all behaviors"""
    
    results = {k: [] for k in k_values}
    
    if "article_ids_clicked" in behaviors_df.columns:
        gt_col = "article_ids_clicked"
    elif "impressions" in behaviors_df.columns:
        # MIND format parsing
        def extract_clicks(imp_str):
            if not imp_str: return []
            return [x.split("-")[0] for x in imp_str.split(" ") if x.endswith("-1")]
            
        behaviors_df = behaviors_df.with_columns(
            pl.col("impressions").map_elements(extract_clicks, return_dtype=pl.List(pl.Utf8)).alias("article_ids_clicked")
        )
        gt_col = "article_ids_clicked"
    else:
        print("Could not find ground truth clicks column. Evaluation cannot proceed.")
        return
        
    print("Evaluating Recall (sequential)...")
    
    def evaluate_single_row(row_dict):
        history = row_dict.get("history_article_ids")
        gt_clicks = row_dict.get(gt_col)
        
        if not gt_clicks or not history:
            return None
            
        # Format history
        if hasattr(history, "to_list"): history = history.to_list()
        elif hasattr(history, "tolist"): history = history.tolist()
        elif not isinstance(history, list): history = list(history)
        history_clicks = [str(x) for x in history][-20:] # limit to last 20 clicks for user representation
            
        # Format ground truth
        if hasattr(gt_clicks, "to_list"): gt_clicks = gt_clicks.to_list()
        elif hasattr(gt_clicks, "tolist"): gt_clicks = gt_clicks.tolist()
        elif not isinstance(gt_clicks, list): gt_clicks = list(gt_clicks)
        gt_clicks = [str(x) for x in gt_clicks]
        
        # Retrieve candidates based on semantic embedding
        retrieved = retriever.retrieve(history_clicks, top_k=max(k_values))
        
        # Calculate metrics
        row_results = {k: calculate_recall(retrieved, gt_clicks, k) for k in k_values}
        return row_results

    rows_to_process = [row for row in behaviors_df.iter_rows(named=True)]
    
    parallel_results = []
    for row in tqdm(rows_to_process):
        parallel_results.append(evaluate_single_row(row))
    
    for row_res in parallel_results:
        if row_res:
            for k in k_values:
                results[k].append(row_res[k])
            
    print("\n--- Semantic Recall Results ---")
    for k in k_values:
        mean_recall = sum(results[k]) / len(results[k]) if results[k] else 0.0
        print(f"Recall@{k}: {mean_recall:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run Semantic Candidate Generation and Evaluation")
    parser.add_argument("--feature_store", type=str, default="feature_store", help="Path to feature store")
    parser.add_argument("--dataset", type=str, default="EBNERD", choices=["EBNERD", "MIND"])
    parser.add_argument("--model", type=str, default=None, help="Sentence Transformer model name")
    args = parser.parse_args()
    
    if args.model is None:
        args.model = "paraphrase-multilingual-MiniLM-L12-v2" if args.dataset == "EBNERD" else "all-MiniLM-L6-v2"
    
    dataset_dir = os.path.join(args.feature_store, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"Feature store not found for {args.dataset}")
        return
        
    print(f"Loading {args.dataset} data from {dataset_dir}...")
    articles_df = pl.read_parquet(os.path.join(dataset_dir, "articles.parquet"))
    val_df = pl.read_parquet(os.path.join(dataset_dir, "val", "behaviors.parquet"))
    
    # 1. Initialize Retriever
    retriever = SemanticRetriever(model_name=args.model)
    
    # 2. Compute Embeddings and Build Index
    emb_cache = os.path.join(dataset_dir, f"embeddings_{args.model.replace('/', '_')}.npy")
    retriever.fit(articles_df, embedding_cache_path=emb_cache)
    
    # 3 & 4. Retrieve and Evaluate
    evaluate_semantic_recall(val_df, retriever)

if __name__ == "__main__":
    main()
