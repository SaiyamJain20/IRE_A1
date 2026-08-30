import polars as pl
import os
import shutil

def main():
    print("Processing MINDlarge_test...")
    data_dir = "data/MINDlarge_test/MINDlarge_test"
    out_dir = "feature_store/MINDlarge"

    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)

    print("Reading news.tsv...")
    news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    
    if not os.path.exists(os.path.join(data_dir, "news.tsv")):
        print(f"Test data not found at {data_dir}")
        return
        
    news_df = pl.read_csv(os.path.join(data_dir, "news.tsv"), separator="\t", has_header=False, new_columns=news_cols, ignore_errors=True, quote_char=None)

    articles_unified = news_df.select([
        pl.col("article_id").alias("article_id"),
        pl.col("title").fill_null(""),
        pl.col("abstract").fill_null(""),
        pl.col("category"),
        pl.lit("en").alias("language"),
        pl.lit("").alias("body")
    ])
    articles_unified.write_parquet(os.path.join(out_dir, "articles.parquet"))

    print("Reading behaviors.tsv...")
    behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
    behaviors_df = pl.read_csv(os.path.join(data_dir, "behaviors.tsv"), separator="\t", has_header=False, new_columns=behaviors_cols, ignore_errors=True, quote_char=None)

    behaviors_unified = behaviors_df.with_columns(
        pl.col("time").str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False).alias("timestamp"),
        pl.col("history").str.split(" ").alias("history_article_ids"),
        pl.lit("MINDlarge").alias("dataset")
    )
    behaviors_unified.write_parquet(os.path.join(out_dir, "test", "behaviors.parquet"))

    print("Done processing MINDlarge_test!")

if __name__ == "__main__":
    main()
