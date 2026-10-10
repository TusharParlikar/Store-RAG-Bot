"""Turn the store's data into chunks, and chunks into vectors.

A chunk is one piece of text the bot can search for and answer from:
  - one product row of data/products/products.csv
  - one section (## heading) of a policy file in data/rules/

  python -m nlp.chunks    print a few chunks and check none is empty
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

DATA = Path(__file__).resolve().parent.parent / "data"


def inr(amount: float) -> str:
    """Indian rupee format with lakh grouping: 123456 -> "₹1,23,456".

    The last three digits form one group; every group before that has two digits.
    """
    digits = str(round(amount))
    if len(digits) > 3:
        head, last_three = digits[:-3], digits[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        digits = f"{head},{last_three}"
    return "₹" + digits


# --------------------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------------------


def product_chunks() -> list[dict]:
    """One chunk per product. `text` is what gets searched; the other keys are exact facts."""
    table = pd.read_csv(DATA / "products" / "products.csv", dtype={"shelf_life": str}).fillna("")

    chunks = []
    for row in table.itertuples():
        text = (
            f"{row.name}. Category: {row.category}. Price: {inr(row.price)}. "
            f"Warranty: {row.warranty_months} months. Benefit: {row.benefit}. "
            f"Good for: {row.good_for}. Goes with: {row.goes_with}."
        )
        if row.shelf_life:
            text += f" Shelf life: {row.shelf_life}."

        chunks.append(
            {
                "text": text,
                "source": f"products.csv #{row.item_id}",
                "kind": "product",
                "item_id": int(row.item_id),
                "name": row.name,
                "price": float(row.price),
                "warranty_months": int(row.warranty_months),
                "category": row.category,
                "link": row.link,
                "benefit": row.benefit,
                "good_for": row.good_for,
                "goes_with": row.goes_with,
            }
        )
    return chunks


def rule_chunks() -> list[dict]:
    """One chunk per "## section" of every policy file."""
    chunks = []
    for path in sorted((DATA / "rules").glob("*.md")):
        # The file is "# Title", then sections that each start with "## Heading".
        title, *sections = path.read_text(encoding="utf-8").split("\n## ")
        title = title.lstrip("# ").strip()

        for section in sections:
            heading, _, body = section.partition("\n")
            heading = heading.strip()
            body = " ".join(body.split())  # one line, single spaces
            chunks.append(
                {
                    "text": f"{title} - {heading}: {body}",
                    "source": f"{path.name} > {heading}",
                    "kind": "rule",
                }
            )
    return chunks


def load_chunks() -> list[dict]:
    """All chunks. Policy chunks come first: rag/index.py relies on that order."""
    return rule_chunks() + product_chunks()


# --------------------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def model():
    """The embedding model, loaded once. Imported here because the import itself is slow."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]):
    """One vector per text: float32, length 1, so inner product = cosine similarity."""
    vectors = model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
    )
    return vectors.astype("float32")


if __name__ == "__main__":
    amounts = (99, 1000, 23379, 123456, 1234567)
    expected = ["₹99", "₹1,000", "₹23,379", "₹1,23,456", "₹12,34,567"]
    assert [inr(amount) for amount in amounts] == expected

    chunks = load_chunks()
    rules = [chunk for chunk in chunks if chunk["kind"] == "rule"]
    print(f"{len(chunks)} chunks ({len(rules)} rule, {len(chunks) - len(rules)} product)\n")
    for chunk in rules + chunks[len(rules) : len(rules) + 3]:
        print(f"[{chunk['source']}] {chunk['text']}\n")
    assert all(chunk["text"].strip() for chunk in chunks), "empty chunk"
