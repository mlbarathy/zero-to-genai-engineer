"""
train_models.py — Train custom Word2Vec and FastText models on the IMDB movie corpus.

WHY TRAIN FROM SCRATCH?
  - Tiny model (~2 MB) vs ~1.6 GB for pre-trained Google News Word2Vec
  - Domain-specific vocabulary ("nolan", "blockbuster", genre names)
  - Trains in seconds on 1,000 movies
  - Best for learning: you see exactly how the algorithm works

DOWNSIDE:
  - Trained on only 1,000 short descriptions vs billions of web sentences
  - Pre-trained vectors from gensim.downloader have much richer semantics
  - Use pre-trained for production; custom-trained for learning and experimentation

USAGE:
    cd backend
    python train_models.py

OUTPUT:
    backend/models/word2vec_movies.model   ← saved Word2Vec
    backend/models/fasttext_movies.model  ← saved FastText

PRE-TRAINED MODELS (used by the recommender on startup):
    These are downloaded automatically via gensim.downloader and cached in ~/gensim-data/
    - GloVe:      glove-wiki-gigaword-50            (~66 MB)
    - FastText:   fasttext-wiki-news-subwords-300   (~958 MB)
    - Word2Vec:   word2vec-google-news-300          (~1.6 GB)
"""

import os
import re
import pandas as pd
from gensim.models import Word2Vec, FastText
import gensim.downloader as api


# ─────────────────────────────────────────────────────────
# STEP 1 — Load and preprocess the movie dataset
# ─────────────────────────────────────────────────────────

# Resolve the CSV path relative to this script (works from any working directory)
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "imdb_top_1000.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n\n❌ Dataset not found at: {CSV_PATH}\n"
        "   Download imdb_top_1000.csv from Kaggle and place it in backend/data/\n"
        "   https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows\n"
    )

print("📂 Loading movie data...")
df = pd.read_csv(CSV_PATH)
print(f"   Loaded {len(df)} movies")


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, strip extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters and spaces
    return re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces


# Combine overview + genre + director + top cast into one text blob per movie
# (Same preprocessing as recommender.py — keeps custom model consistent)
df["combined"] = (
    df["Overview"].fillna("") + " " +
    df["Genre"].fillna("") + " " +
    df["Director"].fillna("") + " " +
    df["Star1"].fillna("") + " " +
    df["Star2"].fillna("")
)
df["text_clean"] = df["combined"].apply(clean_text)

# gensim Word2Vec and FastText both expect:
#   sentences = list of lists of string tokens
#   e.g. [["the", "dark", "knight", "rises", ...], ["inception", ...], ...]
corpus = [doc.split() for doc in df["text_clean"].tolist()]

print(f"   Sample tokens from first movie: {corpus[0][:10]}")
print(f"   Total documents: {len(corpus)}")


# ─────────────────────────────────────────────────────────
# STEP 2 — Create the output directory
# ─────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)  # creates the folder if it doesn't exist yet


# ─────────────────────────────────────────────────────────
# STEP 3 — Train Word2Vec from scratch
# ─────────────────────────────────────────────────────────
# Word2Vec learns word meanings by predicting context words.
# Core idea: "You shall know a word by the company it keeps." — Firth (1957)
#
# Two training modes:
#   sg=0  →  CBOW: predict the CENTER word from surrounding context
#   sg=1  →  Skip-gram: predict CONTEXT words from the center word
#            (Skip-gram works better for rare words)

print("\n[1/2] Training Word2Vec on movie corpus...")

w2v_model = Word2Vec(
    sentences=corpus,   # list-of-lists of tokens (our 1,000 movie descriptions)
    vector_size=100,    # number of dimensions for each word vector
                        # (100 = lightweight; use 300 to match pre-trained Google News)
    window=5,           # context window: how many words left/right to consider
                        # e.g. window=5 means 5 words before and 5 words after
    min_count=1,        # ignore words that appear fewer than this many times
                        # (set to 1 to keep all words in our small corpus)
    workers=4,          # parallel CPU threads for training
    epochs=20,          # how many full passes over the corpus during training
                        # more epochs = slower but potentially better quality
    sg=0,               # 0 = CBOW (faster); 1 = Skip-gram (better for rare words)
    seed=42,            # fixed seed for reproducible results
)

# Save the full model (preserves training state; can resume training later)
W2V_PATH = os.path.join(MODELS_DIR, "word2vec_movies.model")
w2v_model.save(W2V_PATH)
print(f"   ✅ Saved → {W2V_PATH}")

# Quick sanity check — verify the model learned something
try:
    sample_vec = w2v_model.wv["action"]
    print(f"   'action' vector shape: {sample_vec.shape} (first 5 dims: {sample_vec[:5].round(3)})")
    similar = w2v_model.wv.most_similar("thriller", topn=5)
    print(f"   Words most similar to 'thriller': {[w for w, _ in similar]}")
except KeyError as e:
    print(f"   (Word not in corpus vocabulary: {e})")


# ─────────────────────────────────────────────────────────
# STEP 4 — Train FastText from scratch
# ─────────────────────────────────────────────────────────
# FastText extends Word2Vec with character n-gram subword embeddings.
#
# KEY ADVANTAGE: can represent ANY word, even ones never seen during training.
# How? "cinematographic" is built from overlapping character fragments:
#   min_n=3:  "cin", "ine", "nem", "ema", "mat", ...
#   max_n=6:  "cinema", "inemat", "nemato", ...
# The word vector = average of all its character n-gram vectors.
#
# This is why FastText handles typos and rare words so well.

print("\n[2/2] Training FastText on movie corpus...")

ft_model = FastText(
    sentences=corpus,   # same tokenized corpus as Word2Vec
    vector_size=100,    # embedding dimensions (match Word2Vec for fair comparison)
    window=5,           # context window size
    min_count=1,        # minimum word frequency
    workers=4,          # parallel CPU threads
    epochs=20,          # training passes over the corpus
    min_n=3,            # minimum character n-gram length (e.g. "act", "ion")
    max_n=6,            # maximum character n-gram length (e.g. "action")
                        # larger max_n = more subword fragments = slower but handles
                        # longer rare words better
    seed=42,
)

FT_PATH = os.path.join(MODELS_DIR, "fasttext_movies.model")
ft_model.save(FT_PATH)
print(f"   ✅ Saved → {FT_PATH}")

try:
    print(f"   'drama' vector shape: {ft_model.wv['drama'].shape}")
    # FastText's killer feature: it can handle out-of-vocabulary words
    # "cinematographic" was never in our 1,000 movie descriptions, but
    # FastText builds its vector from character n-grams it has seen
    oov_vec = ft_model.wv["cinematographic"]
    print(f"   OOV test — 'cinematographic' (not in corpus): vector shape = {oov_vec.shape} ✅")
except Exception as e:
    print(f"   (Test skipped: {e})")


# ─────────────────────────────────────────────────────────
# HOW TO RELOAD A SAVED MODEL
# ─────────────────────────────────────────────────────────
# After running this script, you can reload models in any Python session:
#
#   from gensim.models import Word2Vec, FastText
#
#   # Load Word2Vec
#   w2v = Word2Vec.load("models/word2vec_movies.model")
#   wv  = w2v.wv          # wv = KeyedVectors (the actual embedding table)
#   print(wv["action"])   # 100-dimensional numpy array
#   print(wv.most_similar("thriller", topn=5))
#
#   # Load FastText
#   ft  = FastText.load("models/fasttext_movies.model")
#   ftv = ft.wv
#   print(ftv["cinematographic"])  # works even for OOV words

print("\n--- Reload test ---")
loaded_w2v = Word2Vec.load(W2V_PATH)
wv = loaded_w2v.wv   # KeyedVectors: the actual lookup table
print(f"   Word2Vec vocab size: {len(wv)}")
try:
    print(f"   'director' most similar: {wv.most_similar('director', topn=3)}")
except KeyError:
    print("   ('director' not in vocabulary)")

loaded_ft = FastText.load(FT_PATH)
ftv = loaded_ft.wv
print(f"   FastText vocab size: {len(ftv)}")


# ─────────────────────────────────────────────────────────
# HOW TO DOWNLOAD AND USE PRE-TRAINED GLOVE / FASTTEXT
# ─────────────────────────────────────────────────────────
# The main recommender.py uses gensim.downloader automatically on startup.
# To use pre-trained vectors in your own code:
#
#   import gensim.downloader as api
#
#   # GloVe — Stanford, 50d, trained on 6B Wikipedia + Gigaword words (~66 MB)
#   glove = api.load("glove-wiki-gigaword-50")
#   print(glove["action"])                        # 50-dim vector
#   print(glove.most_similar("hero", topn=5))
#
#   # FastText — Facebook, 300d + subwords, trained on Wiki + News (~958 MB)
#   fasttext = api.load("fasttext-wiki-news-subwords-300")
#   print(fasttext["cinematography"])             # works for rare words
#   print(fasttext.most_similar("thriller", topn=5))
#
#   # Word2Vec — Google News, 300d, 3M vocab, 100B words (~1.6 GB)
#   word2vec = api.load("word2vec-google-news-300")
#   print(word2vec.most_similar("king", topn=5))
#
# All models cache to ~/gensim-data/ after first download.
# List all available models: print(list(api.info()["models"].keys()))


# ─────────────────────────────────────────────────────────
# HOW TO USE A CUSTOM MODEL IN THE RECOMMENDER
# ─────────────────────────────────────────────────────────
# To swap the pre-trained Word2Vec for your custom-trained model,
# edit recommender.py and replace _build_word2vec() with:
#
#   def _build_word2vec(self):
#       from gensim.models import Word2Vec as GensimW2V
#       model = GensimW2V.load("models/word2vec_movies.model")
#       # IMPORTANT: vector_size must match what you trained (100 here, not 300)
#       matrix = self._avg_vectors(model.wv, size=100)
#       self.sim_w2v = cosine_similarity(matrix)
#
# And similarly for FastText:
#
#   def _build_fasttext(self):
#       from gensim.models import FastText as GensimFT
#       model = GensimFT.load("models/fasttext_movies.model")
#       matrix = self._avg_vectors(model.wv, size=100)
#       self.sim_fasttext = cosine_similarity(matrix)
#
# KEY RULE: the `size` argument to _avg_vectors() must match vector_size from training.


print("\n" + "="*60)
print("✅ Training complete!")
print(f"   Word2Vec  → {W2V_PATH}")
print(f"   FastText  → {FT_PATH}")
print("\nTo start the recommender with pre-trained models:")
print("   uvicorn main:app --reload --port 8000")
print("="*60)
