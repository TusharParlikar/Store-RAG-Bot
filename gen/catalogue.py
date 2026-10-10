"""What the store sells. Plain Python, no LLM.

Three jobs:
  1. Stock check     is_missing("laptop")           -> True, the store sells no laptops
  2. Product names   named_products("the MALM bed") -> the MALM products, to link to
  3. Small helpers   plain(), distinct(), and the word lists other modules share
"""

import re
import unicodedata
from functools import lru_cache

from nlp.chunks import inr, product_chunks
from rag.index import search

# --------------------------------------------------------------------------------------
# Word lists (written by hand; extend them when the request log shows a wrong reply)
# --------------------------------------------------------------------------------------

# fmt: off

# Words the model sometimes returns as the "item" that are needs or policies.
# They must never be reported as a product the store does not sell.
NOT_PRODUCTS = {
    "none", "pain", "ache", "hurt", "relief", "comfort", "support",
    "warranty", "guarantee", "return", "returns", "refund", "window", "policy",
    "delivery", "price", "cost", "order", "assembly", "furniture", "product",
}

# Body parts and medical items. Some are also names of furniture parts ("Leg", "Arm"),
# so they are removed before searching for furniture that helps with a need.
NOT_FURNITURE = {
    "leg", "legs", "arm", "arms", "foot", "feet", "hand", "hands",
    "knee", "ankle", "wrist", "neck",
    "brace", "cast", "crutch", "crutches", "bandage", "splint", "support", "rest",
}

# Other words customers use for product types the catalogue names differently.
SYNONYMS = {
    "crib": "cot",
    "couch": "sofa",
    "closet": "wardrobe",
    "almirah": "wardrobe",
    "cupboard": "cabinet",
}

# Product names that are also everyday words or first names. They count as a product
# name only when typed in capitals ("LACK table"), so "I lack space" finds nothing.
COMMON_WORD_NAMES = {
    "lack", "hallo", "urban", "utter", "harry", "erik", "glenn", "len", "hol", "pax",
    "stig", "jules", "micke", "nisse", "cilla", "terje", "bror", "rast", "olov", "olaus",
}

# fmt: on

# Product types that are spare parts ("STOCKSUND - Legs for armchair").
# They are no help to someone describing a need, so need searches skip them.
PART = re.compile(
    r"(legs?|supporting leg|armrest|backrest|back rest|cover|slipcover|knob|handle|door"
    r"|drawer|shelf|hinge|frame|cushion cover|glass door|plinth|rail)\b",
    re.I,
)

# Words that end the main part of a product description: "bunk bed WITH a slide".
AFTER_MAIN_PART = r"\b(?:with|for|in|that|which)\b"

# Limits for the product page links under a reply.
LINKS_PER_NAME = 3
LINKS_IN_ALL = 6


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def without_accents(text: str) -> str:
    """ "HATTEFJÄLL" -> "HATTEFJALL". Case is kept."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def plain(text: str) -> str:
    """Lowercase without accents, so "hattefjall" matches "HATTEFJÄLL"."""
    return without_accents(text).lower()


def distinct(hits) -> list[dict]:
    """Drop repeats, keeping the first of each.

    The catalogue lists some products twice: same name and price, different item id.
    """
    seen = set()
    unique = []
    for hit in hits:
        key = (hit.get("name") or hit["source"], hit.get("price"))
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique


# --------------------------------------------------------------------------------------
# 1. Stock check
# --------------------------------------------------------------------------------------


def head_noun(kind: str) -> str:
    """The main word of a product type.

    "Office chair with armrests" -> "chair"
    "Laptop table, 100x36 cm"    -> "table"
    """
    main_part = re.split(AFTER_MAIN_PART + "|,", kind.lower())[0]
    words = re.findall(r"[a-z]+", main_part)
    return words[-1] if words else ""


@lru_cache(maxsize=1)
def catalogue_words() -> frozenset[str]:
    """Every kind of product the store has, as single words.

    That is the main word of every product type, plus the words of the category names.
    Main words only: a "Laptop table" must not make the store look like it sells laptops.
    """
    products = product_chunks()
    main_words = {head_noun(product["name"].partition(" - ")[2]) for product in products}
    categories = " ".join(product["category"] for product in products).lower()
    category_words = set(re.findall(r"[a-z]+", categories))
    return frozenset(main_words | category_words)


def in_catalogue(word: str) -> bool:
    """Does the store sell this kind of thing? Singular and plural both count."""
    word = SYNONYMS.get(word, word)
    forms = (word, word + "s", word.removesuffix("s"), word.removesuffix("es"))
    words = catalogue_words()
    return any(form in words for form in forms)


def is_missing(kind: str) -> bool:
    """True when the store has nothing of this kind.

    "desk lamp"              -> True  (no product type ends in "lamp")
    "bunk bed with a slide"  -> False (main word "bed" is sold)
    "warranty"               -> False (not a product at all)
    """
    all_words = set(re.findall(r"[a-z]+", kind))
    if NOT_PRODUCTS & all_words:
        return False

    main_part = re.split(AFTER_MAIN_PART, kind)[0]
    main_words = re.findall(r"[a-z]+", main_part)
    if not main_words:
        return False

    return not in_catalogue(main_words[-1])


# --------------------------------------------------------------------------------------
# 2. Product names and links
# --------------------------------------------------------------------------------------


def product_names(product: dict) -> list[str]:
    """The plain names in a product's title.

    "STENSELE / RÖNNINGE - Table and 2 chairs" -> ["stensele", "ronninge"]
    """
    title = product["name"].partition(" - ")[0]
    return [plain(name.strip()) for name in title.split("/") if name.strip()]


@lru_cache(maxsize=1)
def catalogue_by_name() -> dict[str, list[dict]]:
    """Every product, grouped by plain name: {"malm": [all MALM products], ...}."""
    by_name = {}
    for product in product_chunks():
        for name in product_names(product):
            by_name.setdefault(name, []).append(product)
    return by_name


def names_in(question: str) -> list[str]:
    """The product names the customer typed."""
    lowered = plain(question)
    as_typed = without_accents(question)

    found = []
    for name in catalogue_by_name():
        if not re.search(rf"\b{re.escape(name)}\b", lowered):
            continue
        # An everyday word counts only when typed in capitals.
        if name in COMMON_WORD_NAMES and not re.search(rf"\b{re.escape(name.upper())}\b", as_typed):
            continue
        found.append(name)
    return found


def named_products(question: str) -> list[dict]:
    """The products the customer mentions by name, to link to under the reply.

    For each name: the best search matches first, then the rest of the catalogue.
    At most LINKS_PER_NAME per name and LINKS_IN_ALL in total.
    """
    names = names_in(question)
    if not names:
        return []

    best_matches = [hit for hit in search(question, k=20) if hit["kind"] == "product"]

    products = []
    titles_used = set()
    for name in names:
        candidates = [hit for hit in best_matches if name in product_names(hit)]
        candidates += catalogue_by_name()[name]

        taken = 0
        for product in candidates:
            # Colour variants share a title: one link is enough.
            if taken < LINKS_PER_NAME and product["name"] not in titles_used:
                titles_used.add(product["name"])
                products.append(product)
                taken += 1

    return products[:LINKS_IN_ALL]


def product_links(products: list[dict]) -> str:
    """A "Product pages" block for the end of a reply. Empty when there is nothing to link."""
    rows = [
        f"- [{product['name']}]({product['link']}) – {inr(product['price'])}"
        for product in products
        if product.get("link")
    ]
    if not rows:
        return ""
    return "\n\n**Product pages:**\n" + "\n".join(rows)
