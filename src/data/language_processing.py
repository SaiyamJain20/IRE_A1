import polars as pl
import string

def process_text_features(articles_df, language="en"):
    """
    Apply language-aware processing to article text.
    For English (MIND) and Danish (EB-NeRD):
    - Lowercase text
    - Remove punctuation
    - Remove basic language-specific stopwords
    """
    
    # 1. Lowercase
    processed_df = articles_df.with_columns([
        pl.col("title").str.to_lowercase().alias("title_processed"),
        pl.col("abstract").str.to_lowercase().alias("abstract_processed")
    ])
    
    # 2. Remove punctuation
    # We replace all non-word, non-space characters with a space using regex
    punct_pattern = r"[^\w\s]"
    processed_df = processed_df.with_columns([
        pl.col("title_processed").str.replace_all(punct_pattern, " "),
        pl.col("abstract_processed").str.replace_all(punct_pattern, " ")
    ])
    
    # 3. Language-specific stopword removal
    if language == "en":
        # Common English stopwords
        en_stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "of", "is", "it"]
        stop_pattern = r"\b(" + "|".join(en_stopwords) + r")\b"
        
        processed_df = processed_df.with_columns([
            pl.col("title_processed").str.replace_all(stop_pattern, "").str.strip_chars(),
            pl.col("abstract_processed").str.replace_all(stop_pattern, "").str.strip_chars()
        ])
        
    elif language == "da":
        # Common Danish stopwords
        da_stopwords = ["i", "jeg", "det", "at", "en", "den", "til", "er", "som", "på", "de", "med", "han", "af", "for", "ikke"]
        stop_pattern = r"\b(" + "|".join(da_stopwords) + r")\b"
        
        processed_df = processed_df.with_columns([
            pl.col("title_processed").str.replace_all(stop_pattern, "").str.strip_chars(),
            pl.col("abstract_processed").str.replace_all(stop_pattern, "").str.strip_chars()
        ])
        
    # 4. Cleanup: Replace multiple spaces with a single space
    processed_df = processed_df.with_columns([
        pl.col("title_processed").str.replace_all(r"\s+", " "),
        pl.col("abstract_processed").str.replace_all(r"\s+", " ")
    ])
        
    return processed_df
