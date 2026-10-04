"""Step 2: read data/, split into chunks, turn chunks into vectors.

One product row = one chunk. One policy section (## heading) = one chunk.
Run `python -m nlp.chunks` to print a few chunks.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

DATA = Path(__file__).resolve().parent.parent / "data"


def product_chunks() -> list[dict]:
    df = pd.read_csv(DATA / "products" / "products.csv", dtype={"shelf_life": str}).fillna("")
    chunks = []
    for r in df.itertuples():
        text = (
            f"{r.name}. Category: {r.category}. Price: {r.price:g} SAR. "
            f"Warranty: {r.warranty_months} months. Benefit: {r.benefit}. "
            f"Good for: {r.good_for}. Goes with: {r.goes_with}."
        )
        if r.shelf_life:
            text += f" Shelf life: {r.shelf_life}."
        chunks.append({"text": text, "source": f"products.csv #{r.item_id}", "kind": "product",
                       "item_id": int(r.item_id), "name": r.name, "price": float(r.price),
                       "warranty_months": int(r.warranty_months),
                       "category": r.category, "link": r.link})
    return chunks


def rule_chunks() -> list[dict]:
    chunks = []
    for path in sorted((DATA / "rules").glob("*.md")):
        title, *sections = path.read_text(encoding="utf-8").split("\n## ")
        title = title.lstrip("# ").strip()
        for section in sections:
            heading, _, body = section.partition("\n")
            chunks.append({"text": f"{title} - {heading.strip()}: {' '.join(body.split())}",
                           "source": f"{path.name} > {heading.strip()}", "kind": "rule"})
    return chunks


def load_chunks() -> list[dict]:
    return rule_chunks() + product_chunks()


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]):
    """Unit-length float32 vectors, so inner product = cosine similarity."""
    return model().encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                          batch_size=64, show_progress_bar=len(texts) > 100).astype("float32")


if __name__ == "__main__":
    chunks = load_chunks()
    rules = [c for c in chunks if c["kind"] == "rule"]
    print(f"{len(chunks)} chunks ({len(rules)} rule, {len(chunks) - len(rules)} product)\n")
    for c in rules + chunks[len(rules):len(rules) + 3]:
        print(f"[{c['source']}] {c['text']}\n")
    assert all(c["text"].strip() for c in chunks), "empty chunk"
