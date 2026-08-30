import nltk
from rank_bm25 import BM25Okapi
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Ensure punkt is downloaded for tokenization
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

class BM25Retriever:
    def __init__(self, language="english"):
        self.bm25 = None
        self.article_ids = []
        self.language = language
        
        if language == "da":
            self.nltk_lang = "danish"
        else:
            self.nltk_lang = "english"
            
        self.stopwords = set(stopwords.words(self.nltk_lang))
        self.stemmer = SnowballStemmer(self.nltk_lang)
            
    def _tokenize(self, text):
        if not text:
            return []
        tokens = nltk.word_tokenize(text.lower(), language=self.nltk_lang)
        return [self.stemmer.stem(t) for t in tokens if t.isalnum() and t not in self.stopwords]

    def fit(self, articles_df):
        """
        Build inverted index over article Title + Abstract
        """
        print(f"Building BM25 Index for {len(articles_df)} articles...")
        self.article_ids = articles_df.get_column("article_id").to_list()
        titles = articles_df.get_column("title").fill_null("").to_list()
        abstracts = articles_df.get_column("abstract").fill_null("").to_list()
        
        # Combine title and abstract
        corpus = [f"{t} {a}" for t, a in zip(titles, abstracts)]
        
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Build inverted index for blazing fast retrieval
        print("Building inverted index for O(1) term lookups...")
        self.inverted_index = {}
        for idx, doc_freq in enumerate(self.bm25.doc_freqs):
            for word, freq in doc_freq.items():
                if word not in self.inverted_index:
                    self.inverted_index[word] = {}
                self.inverted_index[word][idx] = freq
                
        # Cache lengths for fast numpy computation
        self.doc_len_np = np.array(self.bm25.doc_len)
        
        print("BM25 Index built successfully.")

    def retrieve(self, query, top_k=200):
        """
        Retrieve top-K candidates for a given query
        """
        if not query or not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        
        # Fast vectorized scoring using inverted index
        score = np.zeros(self.bm25.corpus_size)
        
        for q in set(tokenized_query): # use set to avoid processing same word multiple times
            if q not in self.inverted_index:
                continue
                
            doc_dict = self.inverted_index[q]
            if not doc_dict:
                continue
                
            indices = list(doc_dict.keys())
            freqs = list(doc_dict.values())
            
            indices_np = np.array(indices)
            freqs_np = np.array(freqs)
            
            idf = self.bm25.idf.get(q, 0.0)
            numerator = freqs_np * (self.bm25.k1 + 1)
            denominator = freqs_np + self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * self.doc_len_np[indices_np] / self.bm25.avgdl)
            
            score[indices_np] += idf * (numerator / denominator)
            
        top_k_indices = np.argsort(score)[::-1][:top_k]
        
        return [self.article_ids[i] for i in top_k_indices]
