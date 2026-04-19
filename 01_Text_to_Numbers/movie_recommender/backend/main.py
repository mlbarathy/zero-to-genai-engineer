"""
main.py — FastAPI backend for the CineMatch movie recommender.

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from recommender import MovieRecommender, ALGORITHM_INFO
import os

# ─────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="CineMatch API",
    description="Movie recommendations using 5 different text embedding methods",
    version="1.0.0",
)

# Allow React frontend to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Load recommender on startup (builds all 6 models)
# ─────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "imdb_top_1000.csv")
recommender = MovieRecommender(CSV_PATH)


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    movie: str
    method: str = "tfidf"
    top_n: int = 10


class CompareRequest(BaseModel):
    movie: str
    top_n: int = 8


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": len(recommender.get_titles())}


@app.get("/movies")
def list_movies():
    """Return all movie titles (for the search dropdown)."""
    return {"movies": recommender.get_titles()}


@app.get("/algorithms")
def list_algorithms():
    """Return metadata for each algorithm (displayed in the UI)."""
    return recommender.algorithm_info()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Get top-N movie recommendations for a given movie using one method.

    Methods: bow | tfidf | word2vec | glove | fasttext
    """
    valid_methods = ["bow", "tfidf", "word2vec", "glove", "fasttext"]
    if req.method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"method must be one of {valid_methods}")

    results = recommender.recommend(req.movie, method=req.method, top_n=req.top_n)
    if not results:
        raise HTTPException(status_code=404, detail=f"Movie '{req.movie}' not found")

    return {
        "query": req.movie,
        "method": req.method,
        "algorithm": ALGORITHM_INFO[req.method]["name"],
        "recommendations": results,
    }


@app.post("/compare")
def compare(req: CompareRequest):
    """
    Run ALL 5 methods on the same movie and return all results for comparison.
    """
    all_results = recommender.compare_all(req.movie, top_n=req.top_n)
    if not any(all_results.values()):
        raise HTTPException(status_code=404, detail=f"Movie '{req.movie}' not found")

    return {
        "query": req.movie,
        "results": all_results,
        "algorithms": ALGORITHM_INFO,
    }
