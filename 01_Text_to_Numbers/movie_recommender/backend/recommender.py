"""
recommender.py — Text embedding methods for movie recommendation.

Every method converts text → numbers, then uses cosine similarity to find similar movies.
That's the one constant: representation changes, similarity measure stays the same.

Methods covered: BoW · TF-IDF · Word2Vec (CBOW) · GloVe · FastText
Note: Transformer-based embeddings (SBERT/BERT) are covered in the Attention session.
"""

import re
import os
import ssl
import certifi
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# Fix macOS SSL certificate issue so gensim can download pre-trained models
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ─────────────────────────────────────────────────────────
# Algorithm metadata shown in the frontend
# ─────────────────────────────────────────────────────────
ALGORITHM_INFO = {
    "bow": {
        "name": "Bag of Words",
        "year": "1954",
        "color": "#ef4444",
        "tagline": "Counts word occurrences — order doesn't matter",
        "pro": "Simple, fast, interpretable",
        "con": "Ignores word order and semantics; 'dog bites man' = 'man bites dog'",
        "formula": "V[word] = count of word in document",
    },
    "tfidf": {
        "name": "TF-IDF",
        "year": "1972",
        "color": "#f59e0b",
        "tagline": "Rare words matter more than common ones",
        "pro": "Weights important/unique words higher automatically",
        "con": "Still bag-of-words; no semantic understanding",
        "formula": "TF-IDF = TF(t,d) × log(N / df(t))",
    },
    "word2vec": {
        "name": "Word2Vec (Google News)",
        "year": "2013",
        "color": "#10b981",
        "tagline": "Pre-trained on 100B Google News words — 3M vocab, 300d",
        "pro": "'king - man + woman ≈ queen' — rich real-world semantics",
        "con": "No subword model — OOV words get zero vector",
        "formula": "P(center | context) → trained weights become vectors",
    },
    "glove": {
        "name": "GloVe",
        "year": "2014",
        "color": "#6366f1",
        "tagline": "Global co-occurrence statistics + local context",
        "pro": "Stable, rich pre-trained vectors from 6B words",
        "con": "One vector per word — polysemy not handled ('bank')",
        "formula": "u·v ≈ log P(word_i | word_j)",
    },
    "fasttext": {
        "name": "FastText (Wiki News)",
        "year": "2016",
        "color": "#ec4899",
        "tagline": "Pre-trained on Wikipedia + news — 1M vocab, 300d subwords",
        "pro": "Handles OOV words via character n-grams — typos still work!",
        "con": "Subword noise can hurt very short movie descriptions",
        "formula": "word_vector = avg of character n-gram vectors",
    },
}


# ─────────────────────────────────────────────────────────
# Core Recommender Class
# ─────────────────────────────────────────────────────────
class MovieRecommender:
    """
    Builds 5 different text representations of 1000 movies.
    Then lets you compare how each one ranks similar movies.
    """

    def __init__(self, csv_path: str):
        print("📂 Loading movie data...")
        self.df = pd.read_csv(csv_path)
        self.titles = self.df["Series_Title"].tolist()
        self._prepare_text()
        self._build_all_models()
        print("✅ All models ready — recommender is live!")

    # ─── Text preprocessing ───────────────────────────────

    def _clean(self, text: str) -> str:
        """Lowercase, remove punctuation/numbers, strip extra whitespace."""
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_text(self):
        """Combine relevant metadata into one text field per movie."""
        self.df["combined"] = (
            self.df["Overview"].fillna("") + " " +
            self.df["Genre"].fillna("") + " " +
            self.df["Director"].fillna("") + " " +
            self.df["Star1"].fillna("") + " " +
            self.df["Star2"].fillna("")
        )
        self.df["text_clean"] = self.df["combined"].apply(self._clean)
        self.corpus = self.df["text_clean"].tolist()
        # Tokenized version for Word2Vec / FastText
        self.tokenized = [doc.split() for doc in self.corpus]

    # ─── Build all similarity matrices ───────────────────

    def _build_all_models(self):
        self._build_bow()
        self._build_tfidf()
        self._build_word2vec()
        self._build_glove()
        self._build_fasttext()

    def _build_bow(self):
        print("  [1/5] Building Bag-of-Words model...")
        vec = CountVectorizer(max_features=8000, stop_words="english")
        matrix = vec.fit_transform(self.corpus)
        self.sim_bow = cosine_similarity(matrix)

    def _build_tfidf(self):
        print("  [2/5] Building TF-IDF model...")
        vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(self.corpus)
        self.sim_tfidf = cosine_similarity(matrix)

    def _avg_vectors(self, model_wv, size: int) -> np.ndarray:
        """Average word vectors across each document."""
        vecs = []
        for tokens in self.tokenized:
            token_vecs = [model_wv[w] for w in tokens if w in model_wv]
            vecs.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(size))
        return np.array(vecs)

    def _build_word2vec(self):
        print("  [3/5] Loading pre-trained Google News Word2Vec (300d, ~1.6 GB, cached after first run)...")
        try:
            wv = api.load("word2vec-google-news-300")  # KeyedVectors, 3M vocab
            matrix = self._avg_vectors(wv, 300)
            self.sim_w2v = cosine_similarity(matrix)
        except Exception as e:
            print(f"  ⚠️  Google News Word2Vec download failed ({e}). Falling back to TF-IDF.")
            self.sim_w2v = self.sim_tfidf.copy()

    def _build_glove(self):
        print("  [4/5] Loading pre-trained GloVe vectors (~66 MB, cached after first run)...")
        try:
            glove = api.load("glove-wiki-gigaword-50")
            matrix = self._avg_vectors(glove, 50)
            self.sim_glove = cosine_similarity(matrix)
        except Exception as e:
            print(f"  ⚠️  GloVe download failed ({e}). Falling back to TF-IDF for GloVe slot.")
            self.sim_glove = self.sim_tfidf.copy()

    def _build_fasttext(self):
        print("  [5/5] Loading pre-trained FastText wiki-news vectors (300d, ~958 MB, cached after first run)...")
        try:
            wv = api.load("fasttext-wiki-news-subwords-300")  # KeyedVectors, 1M vocab + subwords
            matrix = self._avg_vectors(wv, 300)
            self.sim_fasttext = cosine_similarity(matrix)
        except Exception as e:
            print(f"  ⚠️  FastText pre-trained download failed ({e}). Falling back to TF-IDF.")
            self.sim_fasttext = self.sim_tfidf.copy()

    # ─── Recommendation logic ─────────────────────────────

    _SIM_MAP = {
        "bow":      lambda self: self.sim_bow,
        "tfidf":    lambda self: self.sim_tfidf,
        "word2vec": lambda self: self.sim_w2v,
        "glove":    lambda self: self.sim_glove,
        "fasttext": lambda self: self.sim_fasttext,
    }

    def _movie_to_dict(self, row, similarity: float) -> dict:
        # Upgrade the tiny 67×98 thumbnail to a full-quality poster (~400px wide)
        poster = str(row.get("Poster_Link", ""))
        poster = re.sub(r'_V1_.*\.jpg$', '_V1_SX400_.jpg', poster)
        return {
            "title":      str(row["Series_Title"]),
            "year":       str(row.get("Released_Year", "N/A")),
            "rating":     str(row.get("IMDB_Rating", "N/A")),
            "genre":      str(row.get("Genre", "N/A")),
            "overview":   str(row.get("Overview", "No overview available")),
            "director":   str(row.get("Director", "N/A")),
            "runtime":    str(row.get("Runtime", "N/A")),
            "poster":     poster,
            "stars":      [
                str(row.get(f"Star{i}", ""))
                for i in range(1, 5)
                if str(row.get(f"Star{i}", "")) not in ("", "nan")
            ],
            "similarity": round(float(similarity), 4),
        }

    def recommend(self, movie_title: str, method: str = "tfidf", top_n: int = 10) -> list:
        """Return top_n similar movies using the chosen method."""
        if method not in self._SIM_MAP:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(self._SIM_MAP)}")
        try:
            idx = self.titles.index(movie_title)
        except ValueError:
            return []

        sim_matrix = self._SIM_MAP[method](self)
        scores = sim_matrix[idx]
        ranked = np.argsort(-scores)
        top_indices = [i for i in ranked if i != idx][:top_n]

        return [self._movie_to_dict(self.df.iloc[i], scores[i]) for i in top_indices]

    def compare_all(self, movie_title: str, top_n: int = 8) -> dict:
        """Run all 5 methods and return their top recommendations."""
        return {
            method: self.recommend(movie_title, method=method, top_n=top_n)
            for method in self._SIM_MAP
        }

    def get_titles(self) -> list:
        return self.titles

    def algorithm_info(self) -> dict:
        return ALGORITHM_INFO
