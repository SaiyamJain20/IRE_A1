import polars as pl
import numpy as np
from tqdm import tqdm
import os
import argparse
from src.models.query_generator import build_queries_for_users
from src.models.bm25_retriever import BM25Retriever

def calculate_recall(retrieved_items, ground_truth_items, k):
    """Calculate recall@K"""
    if not ground_truth_items:
        return 0.0
    
    retrieved_k = set(retrieved_items[:k])
    gt_set = set(ground_truth_items)
    
    hits = len(retrieved_k.intersection(gt_set))
    return hits / len(gt_set)

def evaluate_recall(behaviors_df, retriever, k_values=[50, 100, 200]):
    """Evaluate recall for different K values across all behaviors"""
    
    results = {k: [] for k in k_values}
    
    # In MIND/EB-NeRD, behaviors often have impressions like 'ID-1 ID-0'
    # For recall evaluation, we extract the ground truth clicked articles (the '1's)
    # Since we don't have the explicit impressions parsing in cleaner yet, we'll implement it here for EBNERD
    # EB-NeRD test/val behaviors have 'article_ids_inview' and 'article_ids_clicked' or similar in behaviors.
    # The exact column name in EBNERD behaviors depends on the raw schema (often `article_ids_clicked`).
    
    # Check if we have standard EBNERD columns
    if "article_ids_clicked" in behaviors_df.columns:
        gt_col = "article_ids_clicked"
    elif "impressions" in behaviors_df.columns:
        # MIND format parsing
        # "N55689-1 N35729-0" -> extract IDs with "-1"
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
        
    print("Evaluating Recall (using multiprocessing)...")
    from joblib import Parallel, delayed
    import multiprocessing
    
    num_cores = multiprocessing.cpu_count()
    
    def evaluate_single_row(row_dict):
        query = row_dict.get("generated_query")
        gt_clicks = row_dict.get(gt_col)
        
        if not gt_clicks or not query:
            return None
            
        # Ensure it's a list and items are strings
        if hasattr(gt_clicks, "to_list"):
            gt_clicks = gt_clicks.to_list()
        elif hasattr(gt_clicks, "tolist"):
            gt_clicks = gt_clicks.tolist()
        elif isinstance(gt_clicks, list):
            pass
        else:
            gt_clicks = list(gt_clicks)
            
        gt_clicks = [str(x) for x in gt_clicks]
        
        # Retrieve candidates
        retrieved = retriever.retrieve(query, top_k=max(k_values))
        
        # Calculate metrics
        row_results = {k: calculate_recall(retrieved, gt_clicks, k) for k in k_values}
        return row_results

    # Convert rows to dicts for pickling in joblib
    rows_to_process = [row for row in behaviors_df.iter_rows(named=True)]
    
    # Run parallel evaluation
    parallel_results = Parallel(n_jobs=num_cores)(
        delayed(evaluate_single_row)(row) for row in tqdm(rows_to_process)
    )
    
    for row_res in parallel_results:
        if row_res:
            for k in k_values:
                results[k].append(row_res[k])
            
    # Aggregate results
    print("\n--- BM25 Recall Results ---")
    for k in k_values:
        mean_recall = sum(results[k]) / len(results[k]) if results[k] else 0.0
        print(f"Recall@{k}: {mean_recall:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run BM25 Candidate Generation and Evaluation")
    parser.add_argument("--feature_store", type=str, default="feature_store", help="Path to feature store")
    parser.add_argument("--dataset", type=str, default="EBNERD", choices=["EBNERD", "MIND"])
    args = parser.parse_args()
    
    dataset_dir = os.path.join(args.feature_store, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"Feature store not found for {args.dataset}")
        return
        
    # Load data
    print(f"Loading {args.dataset} data from {dataset_dir}...")
    articles_df = pl.read_parquet(os.path.join(dataset_dir, "articles.parquet"))
    val_df = pl.read_parquet(os.path.join(dataset_dir, "val", "behaviors.parquet"))
    
    # 1. Build Inverted Index
    lang = "da" if args.dataset == "EBNERD" else "en"
    retriever = BM25Retriever(language=lang)
    retriever.fit(articles_df)
    
    # 2. Query Construction
    print("Constructing queries from click history...")
    val_df = build_queries_for_users(val_df, articles_df)
    
    # 3 & 4. Retrieve and Evaluate
    import numpy as np # Ensure np is available for the list conversion
    evaluate_recall(val_df, retriever)

if __name__ == "__main__":
    main()
