import polars as pl
import os

def load_mind_data(data_dir):
    """
    Load and clean MIND dataset (news.tsv and behaviors.tsv)
    Loads both train and dev splits.
    """
    def load_split(split):
        news_path = os.path.join(data_dir, f"MINDsmall_{split}", "news.tsv")
        behaviors_path = os.path.join(data_dir, f"MINDsmall_{split}", "behaviors.tsv")
        
        if not os.path.exists(news_path) or not os.path.exists(behaviors_path):
            print(f"MIND {split} data not found at {news_path}. Please extract it.")
            return None, None
            
        news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        news_df = pl.read_csv(news_path, separator="\t", has_header=False, new_columns=news_cols, ignore_errors=True, quote_char=None)
        
        # Unified article schema
        news_unified = news_df.select([
            pl.col("article_id").alias("article_id"),
            pl.col("title"),
            pl.col("abstract"),
            pl.col("category"),
            pl.lit("en").alias("language"),
            pl.lit("").alias("body") 
        ])
        
        behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        behaviors_df = pl.read_csv(behaviors_path, separator="\t", has_header=False, new_columns=behaviors_cols, ignore_errors=True, quote_char=None)
        
        behaviors_unified = behaviors_df.with_columns(
            pl.col("time").str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False).alias("timestamp"),
            pl.col("history").str.split(" ").alias("history_article_ids"),
            pl.lit("MIND").alias("dataset")
        )
        return news_unified, behaviors_unified

    train_news, train_behaviors = load_split("train")
    dev_news, dev_behaviors = load_split("dev")
    
    if train_news is not None and dev_news is not None:
        articles = pl.concat([train_news, dev_news]).unique(subset=["article_id"])
    else:
        articles = train_news if train_news is not None else dev_news
        
    return articles, {"train": train_behaviors, "val": dev_behaviors}

def load_ebnerd_data(data_dir, split="demo"):
    """
    Load and clean EB-NeRD dataset (train and validation splits)
    """
    base_dir = os.path.join(data_dir, f"ebnerd_{split}")
    articles_path = os.path.join(base_dir, "articles.parquet")
    
    if not os.path.exists(articles_path):
        print(f"EB-NeRD {split} data not found at {articles_path}. Please extract it.")
        return None, None
        
    articles_df = pl.read_parquet(articles_path)
    
    articles_unified = articles_df.select([
        pl.col("article_id").cast(pl.Utf8),
        pl.col("title"),
        pl.col("subtitle").alias("abstract"),
        pl.col("category_str").alias("category"),
        pl.lit("da").alias("language"),
        pl.col("body").fill_null("")
    ])
    
    def load_behaviors(sub_split):
        history_path = os.path.join(base_dir, sub_split, "history.parquet")
        behaviors_path = os.path.join(base_dir, sub_split, "behaviors.parquet")
        if not os.path.exists(behaviors_path):
            return None
        behaviors_df = pl.read_parquet(behaviors_path)
        if os.path.exists(history_path):
            history_df = pl.read_parquet(history_path)
            behaviors_df = behaviors_df.join(history_df, on="user_id", how="left")
            
        return behaviors_df.with_columns([
            pl.col("impression_time").alias("timestamp"),
            pl.col("article_id_fixed").alias("history_article_ids"),
            pl.lit("EBNERD").alias("dataset")
        ])
        
    train_behaviors = load_behaviors("train")
    val_behaviors = load_behaviors("validation")
    
    return articles_unified, {"train": train_behaviors, "val": val_behaviors}

def build_unified_schema(data_dir, ebnerd_split="demo"):
    mind_articles, mind_splits = load_mind_data(data_dir)
    ebnerd_articles, ebnerd_splits = load_ebnerd_data(data_dir, ebnerd_split)
    
    return {
        "MIND": {"articles": mind_articles, "behaviors": mind_splits},
        "EBNERD": {"articles": ebnerd_articles, "behaviors": ebnerd_splits}
    }
