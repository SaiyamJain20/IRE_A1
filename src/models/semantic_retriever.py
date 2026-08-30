import polars as pl
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

class SemanticRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None, dataset="MIND"):
        """
        Initialize the Semantic Retriever using Sentence-Transformers and FAISS.
        """
        self.dataset = dataset
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Semantic Model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.article_ids = []
        self.article_embeddings = {}
        
    def fit(self, articles_df, embedding_cache_path=None):
        """
        Compute or load embeddings for articles and build the FAISS index.
        """
        self.article_ids = articles_df.get_column("article_id").cast(pl.Utf8).to_list()
        titles = articles_df.get_column("title").fill_null("").to_list()
        abstracts = articles_df.get_column("abstract").fill_null("").to_list()
        
        # Combine title and abstract for embedding
        texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
        
        # Check if embeddings are cached
        if embedding_cache_path and os.path.exists(embedding_cache_path):
            print(f"Loading cached embeddings from {embedding_cache_path}...")
            embeddings = np.load(embedding_cache_path)
        else:
            print(f"Computing embeddings for {len(texts)} articles...")
            embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            if embedding_cache_path:
                print(f"Saving embeddings to {embedding_cache_path}...")
                os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
                np.save(embedding_cache_path, embeddings)
                
        # Store embeddings in a dict for fast user-representation lookup
        print("Caching article embeddings for mean-pooling...")
        for aid, emb in zip(self.article_ids, embeddings):
            self.article_embeddings[aid] = emb
            
        # Build FAISS Index
        print("Building FAISS Index...")
        dimension = embeddings.shape[1]
        
        # Normalize for Cosine Similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print("FAISS Index built successfully.")
        
    def get_user_representation(self, history_article_ids):
        """
        Mean-pool the embeddings of previously clicked articles.
        """
        valid_embs = [self.article_embeddings[aid] for aid in history_article_ids if aid in self.article_embeddings]
        
        if not valid_embs:
            return None
            
        # Exponential decay is very effective for breaking news (EBNERD)
        # but hurts long-term interest modeling for MIND.
        if self.dataset == "EBNERD":
            n = len(valid_embs)
            weights = np.exp(np.linspace(-2.0, 0.0, n))  # decay factor
            weights /= weights.sum()
            user_emb = np.average(valid_embs, axis=0, weights=weights)
        else:
            user_emb = np.mean(valid_embs, axis=0)
        
        # Normalize the user vector for Cosine Similarity
        user_emb = user_emb / np.linalg.norm(user_emb)
        return user_emb
        
    def retrieve(self, history_article_ids, top_k=200):
        """
        Retrieve top_k candidates for a given user history using FAISS.
        """
        if not history_article_ids or self.index is None:
            return []
            
        user_emb = self.get_user_representation(history_article_ids)
        if user_emb is None:
            return []
            
        user_emb = np.expand_dims(user_emb, axis=0).astype(np.float32)
        
        # Search FAISS
        distances, indices = self.index.search(user_emb, top_k)
        
        # Return corresponding article IDs
        return [self.article_ids[idx] for idx in indices[0] if idx != -1]
