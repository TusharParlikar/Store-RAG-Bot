"""The search index: build it once, then search it for every message.

Every chunk (one product, or one policy section) is stored as a vector in a FAISS index.
A search turns the question into a vector and returns the chunks closest to it.

  python -m rag.index    rebuild index/ and print a few sample searches
"""

import json
from pathlib import Path

import faiss

from nlp.chunks import embed, load_chunks

# Where the built index is kept. Not in git: it is rebuilt from data/ when missing.
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# Length of an all-MiniLM-L6-v2 vector.
VECTOR_SIZE = 384

# The index and chunks, loaded on the first search and kept in memory.
_cache = {}


# --------------------------------------------------------------------------------------
# Build and load
# --------------------------------------------------------------------------------------


def build():
    """Embed every chunk and save the index. Returns (index, chunks)."""
    chunks = load_chunks()

    # "IP" is inner product. The vectors have length 1, so inner product = cosine similarity.
    index = faiss.IndexFlatIP(VECTOR_SIZE)
    index.add(embed([chunk["text"] for chunk in chunks]))

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks), encoding="utf-8")
    return index, chunks


def load():
    """Read the saved index, or build it when there is none yet. Returns (index, chunks)."""
    if not INDEX_FILE.exists():
        return build()
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------


def search(query: str, k: int = 5, rules: int = 2) -> list[dict]:
    """The chunks closest to the query, best first.

    Returns the top `k` chunks of any kind, plus the best `rules` policy sections.
    Each result is the chunk dict with a `score` added (cosine similarity, higher is closer).

    Policy sections are searched a second time on their own, because about 3,000
    product rows would otherwise crowd the handful of policy sections out.
    """
    if not _cache:
        _cache["index"], _cache["chunks"] = load()
        # Policy chunks are stored first, so they are ids 0 .. n_rules - 1.
        _cache["n_rules"] = sum(chunk["kind"] == "rule" for chunk in _cache["chunks"])

    index = _cache["index"]
    chunks = _cache["chunks"]
    vector = embed([query])

    # Search 1: everything.
    scores, ids = index.search(vector, k)
    found = dict(zip(ids[0], scores[0]))

    # Search 2: policy sections only.
    only_rules = faiss.SearchParameters(sel=faiss.IDSelectorRange(0, _cache["n_rules"]))
    scores, ids = index.search(vector, rules, params=only_rules)
    found.update(zip(ids[0], scores[0]))

    # FAISS gives id -1 when it has fewer results than asked for.
    hits = [{**chunks[i], "score": float(score)} for i, score in found.items() if i >= 0]
    return sorted(hits, key=lambda hit: -hit["score"])


if __name__ == "__main__":
    build()
    samples = [
        "warranty on the chair",
        "back pain",
        "coffee table",
        "who won the football world cup",
    ]
    for query in samples:
        print(f"\n> {query}")
        for hit in search(query):
            print(f"  {hit['score']:.2f}  [{hit['source']}] {hit['text'][:90]}")
