"""In-process RAG over the documents table."""

from __future__ import annotations

import json
from typing import Any

from dining_bot.config import RAG_SCORE_THRESHOLD
from dining_bot.db.connections import connect_readonly, connect_write

class DiningRAG:
    """Tiny in-process RAG over the `documents` table (S10 ideas, one file)."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.embeddings = None  # numpy array (n, d)
        self.model = None

    def build(self) -> "DiningRAG":
        import numpy as np
        from sentence_transformers import SentenceTransformer

        con = connect_readonly()
        try:
            rows = con.execute(
                "SELECT id, name, section, version, chunk, source_type, source_file "
                "FROM documents ORDER BY id"
            ).fetchall()
        finally:
            con.close()

        self.docs = [dict(r) for r in rows]
        if not self.docs:
            raise RuntimeError("RETRIEVAL_EMPTY: no documents in dining_bot.db")

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [f"{d['name']} — {d['section']}: {d['chunk']}" for d in self.docs]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        self.embeddings = np.asarray(vectors, dtype="float32")

        # Optional: persist embeddings back (nullable column in schema).
        try:
            w = connect_write()
            for d, vec in zip(self.docs, self.embeddings):
                w.execute(
                    "UPDATE documents SET embedding = ? WHERE id = ?",
                    (json.dumps(vec.tolist()), d["id"]),
                )
            w.commit()
            w.close()
        except Exception:  # noqa: BLE001
            pass  # read-only demo still works from memory
        return self

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        import numpy as np

        assert self.model is not None and self.embeddings is not None
        q = self.model.encode([query], normalize_embeddings=True)
        scores = (self.embeddings @ np.asarray(q, dtype="float32").T).ravel()
        order = np.argsort(-scores)[:k]
        hits = []
        for i in order:
            score = float(scores[i])
            if score < RAG_SCORE_THRESHOLD:
                continue
            d = dict(self.docs[int(i)])
            d["score"] = score
            hits.append(d)
        return hits


# Citations are built HERE in Python — never by the LLM inventing sources (FR-8).
def format_citations(hits: list[dict[str, Any]]) -> list[str]:
    cites = []
    for h in hits:
        src = h.get("source_type") or "md"
        file = h.get("source_file") or "unknown"
        cites.append(
            f"{h['name']} · {h['section']} · {h['version']} · "
            f"{src}:{file} (score={h['score']:.2f})"
        )
    return cites
