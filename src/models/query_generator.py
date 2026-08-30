import polars as pl
from collections import Counter

def construct_query(history_article_ids, articles_df):
    """
    Given a list of clicked article IDs from the user's history,
    construct a text query by concatenating their titles.
    """
    if not history_article_ids:
        return ""
        
    # Get the titles of the articles in the history
    clicked_articles = articles_df.filter(pl.col("article_id").is_in(history_article_ids))
    titles = clicked_articles.get_column("title").to_list()
    
    # Optional: We could also use abstract, but usually Title concatenation is sufficient for lexical queries
    query = " ".join([t for t in titles if t is not None])
    return query

def build_queries_for_users(behaviors_df, articles_df, idf_dict=None, dataset="MIND"):
    """
    Build queries for an entire behaviors dataframe.
    Uses a multi-signal approach conditionally based on dataset:
      1. Concatenate last-5 clicked titles (recency focus)
      2. If EBNERD, keep only the top-15 highest-IDF terms (distinctiveness)
      3. If EBNERD, prepend the user's dominant category for topical boosting
    """
    # Create lookup dicts
    article_ids = articles_df.get_column("article_id").to_list()
    titles = articles_df.get_column("title").fill_null("").to_list()
    categories = articles_df.get_column("category").fill_null("").to_list()
    
    title_dict = dict(zip(article_ids, titles))
    category_dict = dict(zip(article_ids, categories))
    
    queries = []
    for row in behaviors_df.iter_rows(named=True):
        history = row.get("history_article_ids")
        if not history:
            queries.append("")
            continue
            
        # --- Signal 1: Last-5 title concatenation ---
        recent_history = history[-5:]
        raw_titles = [title_dict.get(str(aid), "") for aid in recent_history]
        raw_query = " ".join([t for t in raw_titles if t])
        
        # --- Signal 2: TF-IDF term selection (EBNERD only) ---
        if dataset == "EBNERD" and idf_dict and raw_query:
            # Tokenize naively (splitting on whitespace); BM25 will re-tokenize properly later
            terms = raw_query.lower().split()
            # Score each term by IDF and keep the top 15 most distinctive
            scored = sorted(set(terms), key=lambda t: idf_dict.get(t, 0.0), reverse=True)
            raw_query = " ".join(scored[:15])
        
        # --- Signal 3: Category boost (EBNERD only) ---
        if dataset == "EBNERD":
            hist_cats = [category_dict.get(str(aid), "") for aid in history if category_dict.get(str(aid), "")]
            if hist_cats:
                dominant_cat = Counter(hist_cats).most_common(1)[0][0]
                raw_query = f"{dominant_cat} {raw_query}"
        
        queries.append(raw_query)
        
    return behaviors_df.with_columns(pl.Series(name="generated_query", values=queries))
