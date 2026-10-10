"""Build the FAISS index and search it.

`python -m rag.index` rebuilds index/ and runs a few sample searches.
"""
import json
from pathlib import Path

import faiss

from nlp.chunks import embed, load_chunks

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"


def build():
    chunks = load_chunks()
    index = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 vector size; inner product of unit vectors = cosine
    index.add(embed([c["text"] for c in chunks]))
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return index, chunks


def load():
    if not (INDEX_DIR / "faiss.index").exists():
        return build()
    return faiss.read_index(str(INDEX_DIR / "faiss.index")), json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))


_cache = {}


def search(query: str, k: int = 5, rules: int = 2) -> list[dict]:
    """Top k chunks plus the best `rules` policy sections, each with a `score` (cosine, higher is closer).

    Policy sections are searched separately because ~3,000 product rows would otherwise crowd them out.
    """
    if not _cache:
        _cache["index"], _cache["chunks"] = load()
        _cache["n_rules"] = sum(c["kind"] == "rule" for c in _cache["chunks"])  # rules are stored first
    index, chunks, vec = _cache["index"], _cache["chunks"], embed([query])
    scores, ids = index.search(vec, k)
    found = dict(zip(ids[0], scores[0]))
    only_rules = faiss.SearchParameters(sel=faiss.IDSelectorRange(0, _cache["n_rules"]))
    scores, ids = index.search(vec, rules, params=only_rules)
    found.update(zip(ids[0], scores[0]))
    hits = [{**chunks[i], "score": float(s)} for i, s in found.items() if i >= 0]
    return sorted(hits, key=lambda h: -h["score"])


if __name__ == "__main__":
    build()
    for q in ["warranty on the chair", "back pain", "coffee table", "who won the football world cup"]:
        print(f"\n> {q}")
        for hit in search(q):
            print(f"  {hit['score']:.2f}  [{hit['source']}] {hit['text'][:90]}")
