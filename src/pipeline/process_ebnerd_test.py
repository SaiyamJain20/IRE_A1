import polars as pl
import os
import shutil

def main():
    print("Processing EBNERD testset (Fast Mode - No Join)...")
    data_dir = "data/ebnerd_testset"
    out_dir = "feature_store/EBNERD"

    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)

    # 1. Just copy history.parquet as-is so predict.py can load it as a dict!
    history_src = os.path.join(data_dir, "ebnerd_testset", "test", "history.parquet")
    if os.path.exists(history_src):
        shutil.copy(history_src, os.path.join(out_dir, "test", "history.parquet"))
        print("Copied history.parquet")

    # 2. Process behaviors (No Join = NO OOM!)
    print("Formatting test behaviors...")
    behaviors_path = os.path.join(data_dir, "ebnerd_testset", "test", "behaviors.parquet")
    if not os.path.exists(behaviors_path):
        print(f"Test data not found at {behaviors_path}")
        return
        
    behaviors_df = pl.read_parquet(behaviors_path)

    behaviors_unified = behaviors_df.with_columns(
        pl.col("article_ids_inview").cast(pl.List(pl.Int64)),
        pl.Series("article_ids_clicked", [[]]*len(behaviors_df), dtype=pl.List(pl.Int64)),
        pl.lit("EBNERD").alias("dataset")
    )
    behaviors_unified.write_parquet(os.path.join(out_dir, "test", "behaviors.parquet"))

    # 3. Process test articles
    test_articles_path = os.path.join(data_dir, "ebnerd_testset", "articles.parquet")
    if os.path.exists(test_articles_path):
        print("Reading test articles...")
        new_articles = pl.read_parquet(test_articles_path)
        existing = pl.read_parquet(os.path.join(out_dir, "articles.parquet"))
        combined = pl.concat([existing, new_articles], how="diagonal_relaxed").unique(subset=["article_id"])
        combined.write_parquet(os.path.join(out_dir, "articles.parquet"))

    print("Done processing EBNERD testset!")

if __name__ == "__main__":
    main()
