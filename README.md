# Information Retrieval & Extraction - Assignment 1

This repository contains the complete implementation for Assignment 1, demonstrating a robust news recommendation retrieval and evaluation pipeline using Lexical (BM25) and Semantic (Sentence-Transformers + FAISS) models on the EB-NeRD and MIND datasets.

## 🚀 How to Run

Ensure your virtual environment is active and run the following commands to execute the end-to-end pipeline:

```bash
# 1. Build the unified feature store (Automatically downloads and extracts datasets)
python main.py build

# 2. Evaluate both models across both datasets simultaneously
python main.py evaluate --dataset all --model_type all

# 3. Generate Codabench Submissions (Saved to outputs/)
python main.py predict --dataset MIND
```

## 📂 Source Library (`src/`)

* **`src/pipeline/`**: Core logic for building feature stores, offline evaluation, and Codabench predictions.
* **`src/models/bm25_retriever.py`**: Implementation of lexical retrieval using `rank_bm25` and NLTK.
* **`src/models/semantic_retriever.py`**: Implementation of semantic retrieval using HuggingFace transformers and FAISS indexing.
* **`src/models/query_generator.py`**: Utilities for constructing lexical queries out of raw user history.
* **`src/evaluation/metrics.py`**: Highly optimized formulas for Accuracy metrics (AUC, MRR, nDCG) and Beyond-Accuracy metrics (Diversity, Novelty, Coverage).
