import os
import polars as pl
import argparse

def test_future_click_leakage(feature_store, dataset):
    dataset_dir = os.path.join(feature_store, dataset)
    print(f"\nRunning Future-Click Leakage Test on {dataset}...")
    
    val_df_path = os.path.join(dataset_dir, "val", "behaviors.parquet")
    if not os.path.exists(val_df_path):
        print("Validation data not found.")
        return
        
    val_df = pl.read_parquet(val_df_path)
    
    # We need to check if any article in history_article_ids was actually published AFTER the impression time
    # This requires joining with articles.parquet to get publication times.
    # For a strict Anti-Gaming test, we just check that the 'timestamp' of the impression is strictly 
    # greater than all known click/interaction timestamps in the user's history, but since we don't 
    # explicitly store individual click times in this schema, we verify against the article publication time.
    
    articles_df = pl.read_parquet(os.path.join(dataset_dir, "articles.parquet"))
    
    if "published_time" not in articles_df.columns:
        print("⚠️  'published_time' not found in articles schema. Performing a basic logical split test instead.")
        # Basic check: verify that no impression ID from train exists in val
        train_df_path = os.path.join(dataset_dir, "train", "behaviors.parquet")
        if os.path.exists(train_df_path):
            train_df = pl.read_parquet(train_df_path)
            train_ids = set(train_df.get_column("impression_id").to_list())
            val_ids = set(val_df.get_column("impression_id").to_list())
            
            # Since MIND impression IDs are just line numbers (1, 2, 3...) they are not globally unique.
            # Instead of checking ID overlap, we check the temporal boundary directly.
            if "timestamp" in train_df.columns and "timestamp" in val_df.columns:
                train_max = train_df.select(pl.max("timestamp")).item()
                val_min = val_df.select(pl.min("timestamp")).item()
                if train_max is not None and val_min is not None:
                    assert train_max < val_min, f"Future-click leakage detected! Train ends {train_max} but Val starts {val_min}."
                    print("✅ Pass: Strict temporal boundary maintained between Train and Val sets.")
            else:
                print("⚠️  'timestamp' not found, unable to perform logical split test.")
        return
        
    # Strict temporal leakage test (if published_time is available)
    pub_times = dict(zip(articles_df.get_column("article_id").to_list(), articles_df.get_column("published_time").to_list()))
    
    leakage_count = 0
    for row in val_df.iter_rows(named=True):
        imp_time = row.get("impression_time")
        if imp_time is None: continue
        
        history = row.get("history_article_ids")
        if not history: continue
        
        if hasattr(history, "to_list"): history = history.to_list()
        
        for aid in history:
            pub_time = pub_times.get(str(aid))
            if pub_time and pub_time > imp_time:
                leakage_count += 1
                
    assert leakage_count == 0, f"Future-click leakage detected! {leakage_count} historical clicks occurred after the impression time."
    print("✅ Pass: No future-click leakage. All historical clicks occurred before the impression time.")

def main():
    parser = argparse.ArgumentParser(description="Future-Click Leakage Test")
    parser.add_argument("--feature_store", type=str, default="feature_store")
    parser.add_argument("--dataset", type=str, default="all", choices=["EBNERD", "MIND", "all"])
    args = parser.parse_args()
    
    datasets = ["EBNERD", "MIND"] if args.dataset == "all" else [args.dataset]
    for d in datasets:
        test_future_click_leakage(args.feature_store, d)

if __name__ == "__main__":
    main()
