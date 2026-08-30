# Information Retrieval & Extraction - Assignment 1
**Lexical & Semantic Retrieval on EB-NeRD and MIND**

This repository contains the complete implementation for Assignment 1, demonstrating a robust news recommendation retrieval and evaluation pipeline. 

It covers both **Lexical (BM25)** and **Semantic (Sentence-Transformers + FAISS)** retrieval, alongside a rigorous offline evaluation harness that computes Accuracy and Beyond-Accuracy metrics.

---

## 🔍 Detailed Phase Breakdown

### Phase 1 & 2: Reproducible Data Pipeline (`main.py build`)
In these initial phases, we transition from raw data formats into a highly structured **Feature Store** that serves as the backbone for the rest of the project.
* **Unified Schema Generation:** The raw data for EB-NeRD (Danish) and MIND (English) arrive in entirely different formats. We parse, clean, and merge these into a single unified schema so that downstream models don't need to write dataset-specific logic.
* **Temporal Splitting:** We enforce strict time-based Train/Validation/Test splits (e.g., using the last N days as testing data, and the preceding M days as validation data). We explicitly avoid random splitting to prevent "future-click leakage", ensuring that the model never accidentally trains on future interactions.
* **Feature Store Output:** The cleaned articles (with concatenated Title + Abstract) and user behaviors (impressions, history) are saved as `parquet` files for rapid loading.

### Phase 3: Lexical Candidate Generation (`main.py bm25`)
This phase focuses on keyword-based retrieval. We aim to find news articles that explicitly share words with the articles the user has clicked on in the past.
* **Inverted Indexing:** We construct an inverted index over the entire corpus of article texts (titles and abstracts) using the `rank_bm25` library, utilizing language-specific tokenization, stopword removal, and Snowball stemming (NLTK) for Danish and English.
* **Query Generation:** For every user, we simulate a "search query" by gathering the last 5 articles they clicked on and concatenating their titles into a focused string.
* **Retrieval & Recall:** We query the BM25 index with this string, retrieve the top 50, 100, and 200 candidates, and calculate a strict `Recall@K` to determine how often the true clicked article appeared in our candidate pool.

### Phase 4: Semantic Candidate Generation (`main.py semantic`)
This phase focuses on meaning-based retrieval. We aim to find news articles that have the same *context* or *topic* as the user's history, even if they don't share exact keywords.
* **Article Embeddings:** We use language-appropriate HuggingFace Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2` for Danish/EB-NeRD, `all-MiniLM-L6-v2` for English/MIND) to convert every article's text into a dense mathematical vector.
* **FAISS Indexing:** To avoid comparing a user against 50,000+ articles one-by-one, we build an Approximate Nearest Neighbor (ANN) index using FAISS (`IndexFlatIP`), optimizing for Maximum Inner Product Search (Cosine Similarity).
* **User Representation:** We build a profile for each user by computing a recency-weighted average (exponential decay) of the embeddings of their historically clicked articles, giving more weight to recent interactions.
* **Retrieval & Recall:** We perform a rapid vector search against the FAISS index to find the closest matching semantic articles, generating candidates and calculating `Recall@K`.

### Phase 5: Offline Evaluation Harness (`main.py evaluate`)
Instead of just measuring how well we retrieve candidates from the entire internet, we rigorously test how well our models *rank* a specific subset of candidates (an `impression`) presented to the user at a specific point in time. 
* **Optimized Scoring:** We wrote highly optimized scoring algorithms that parse the exact `inview` candidates for a user, calculate the BM25 or Semantic similarity on the fly, and rank them.
* **Accuracy Metrics:** We calculate standard ranking metrics including Area Under the ROC Curve (AUC), Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG@5, nDCG@10).
* **Beyond-Accuracy Metrics:** A good recommender shouldn't just be accurate; it should be diverse. We calculate:
  * **Intra-List Diversity (ILD):** The average pairwise cosine distance between the recommended items.
  * **Novelty:** How unpopular the recommended items generally are (rewarding algorithms that surface hidden gems).
  * **Catalog Coverage:** The percentage of the entire article database that is recommended to at least one user.
* **Data Slicing:** We evaluate the metrics across quartile-based history-length groups (short vs. long history users) for more granular analysis.
* **Confidence Intervals:** We compute Bootstrap 95% Confidence Intervals for AUC, MRR, nDCG@5, and nDCG@10 to ensure our results are statistically significant.
* **Hybrid Model:** We also support a `hybrid` mode that linearly combines normalized BM25 and Semantic scores for improved ranking.
* **Serving-Time Ablation:** With `--no_future_features`, we evaluate performance when abstracts/body text are unavailable (simulating real-time serving constraints).

---

## 📂 Source Library (`src/`)
* **`src/models/bm25_retriever.py`**: Implementation of the `BM25Retriever` class using NLTK and `rank_bm25`.
* **`src/models/semantic_retriever.py`**: Implementation of the `SemanticRetriever` class utilizing HuggingFace and FAISS.
* **`src/models/query_generator.py`**: Utilities for constructing lexical queries out of raw user history.
* **`src/evaluation/metrics.py`**: Highly optimized formulas for AUC, MRR, nDCG, Diversity, Novelty, Coverage, and Bootstrapping.

## 🚀 How to Run (One-Command Reproduce)
To rebuild the entire pipeline from scratch, ensure your virtual environment is active and run:
```bash
# 1. Build the unified feature store (Automatically downloads and extracts datasets)
python main.py build

# 3. Evaluate both models across both datasets simultaneously
python main.py evaluate --dataset all --model_type all

# 4. Generate Codabench Submissions (Saved to outputs/)
python main.py predict --dataset MIND
```
