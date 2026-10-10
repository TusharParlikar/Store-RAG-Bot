"""Work out which product the customer means.

"The second one", "1st", "number 3", "the cheapest", "both", or a product name.
This is plain Python on purpose: the small model often reads "ok I'll get that"
as the earlier need again and starts a new search.

  listed()         which products the last reply showed, in the order it showed them
  BUY              does this message pick something?
  pick_product()   the one product meant
  pick_products()  every product meant, when the customer picks several at once
"""

import re

from gen.catalogue import names_in, plain, product_names
from nlp.chunks import inr
from rag.index import search

# --------------------------------------------------------------------------------------
# Does the message pick a product?
# --------------------------------------------------------------------------------------

# Each line is one way of saying "I'll take it". Used only right after the bot listed products.
BUYING_PHRASES = [
    r"\bi'?ll (take|get|buy|have)\b",  # "I'll take ...", "ill get ..."
    r"\bi ?will (take|get|buy)\b",  # "I will buy ..."
    r"\b(buy|take|get|want) (it|this|that|this one|that one)\b",  # "I want this one"
    r"\b(that|this|the (first|second|third)) one\b",  # "that one", "the second one"
    r"\badd (it|this|that)\b",  # "add it"
    r"\b(number|option|#)\s*[1-3]\b",  # "number 2", "option 1"
    # "I want to take the 1st chair": a buying word, then a position soon after
    r"\b(take|get|buy|want|choose|pick|like)\b[^.?!]{0,20}\b(first|second|third|1st|2nd|3rd)\b",
    r"^\s*[1-3]\s*\.?\s*$",  # just the number: "2"
    r"^\s*(the\s+)?(first|second|third|1st|2nd|3rd)(\s+one)?\s*[.!]?\s*$",  # just "1st"
    r"\b(the\s+)?(cheapest|least expensive|most expensive|priciest)(\s+one)?\b",
    r"\b(the last|last one)\b",  # not "last" alone: "I bought it last year"
    r"\b(take|get|buy|want|add)\b[^.?!]{0,20}\b(both|all)\b",  # "I'll take both"
    r"^\s*both\b",
    r"\badd\b.{0,40}\b(cart|basket)\b",  # "add the first one to my cart"
]
BUY = re.compile("|".join(BUYING_PHRASES), re.I)

# A position in the list, in the three ways customers write it.
ORDINAL = re.compile(
    r"\b(?P<word>first|1st|second|2nd|third|3rd)\b"  # "second", "2nd"
    r"|(?:number|no\.?|#|option)\s*(?P<labelled>[1-3])\b"  # "number 2"
    r"|^\s*(?P<bare>[1-3])\s*\.?\s*$"  # "2"
)
POSITION_OF_WORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}


# A clear wish to buy. With a product name from the catalogue, this is a pick even when
# nothing was listed before: "I want to buy the POÄNG rocking-chair".
WANTS_TO_BUY = re.compile(
    r"\b(buy|purchase)\b" r"|\bi('?ll| will) (take|get|have)\b" r"|\badd\b.{0,40}\b(cart|basket)\b",
    re.I,
)


def buys_by_name(question: str) -> bool:
    """True for a buying wish that names a product the store has."""
    return bool(WANTS_TO_BUY.search(question)) and bool(names_in(question))


def ordinals(text: str) -> str:
    """Join "1 st" into "1st". Customers type it both ways."""
    return re.sub(r"\b([1-3])\s+(st|nd|rd)\b", r"\1\2", text, flags=re.I)


def position(match: re.Match) -> int:
    """The list index (0 = first) that an ORDINAL match points to."""
    if match["word"]:
        return POSITION_OF_WORD[match["word"]]
    return int(match["labelled"] or match["bare"]) - 1


# --------------------------------------------------------------------------------------
# What the last reply listed
# --------------------------------------------------------------------------------------


def listed(reply: str, sources: list[dict]) -> list[dict]:
    """The products a reply actually lists, in the order the reply lists them.

    `sources` is every search result behind the reply; the reply shows only some of them.
    The order matters: "the second one" means the second in the text, and the model
    does not keep the search order.
    """
    shown = [
        source
        for source in sources
        if source.get("kind") == "product"
        and source["name"] in reply
        and inr(source["price"]) in reply
    ]

    def place_in_reply(product: dict):
        # Two products can share a name (colour variants), so the price breaks the tie.
        return reply.find(product["name"]), reply.find(inr(product["price"]))

    return sorted(shown, key=place_in_reply)


# --------------------------------------------------------------------------------------
# Picking
# --------------------------------------------------------------------------------------


def pick_product(question: str, shown: list[dict]) -> dict | None:
    """The one product the customer means, or None when it is not clear.

    Tried in this order:
      1. a product name, in the list just shown or anywhere in the catalogue
      2. "the cheapest", "the most expensive", "the last one"
      3. a position: "2", "1st", "the second one", "number 3"
      4. only one product was shown, so it must be that one
    """
    q = plain(ordinals(question))

    # 1. A product name. Products just shown come first, so the right variant wins.
    in_catalogue = [hit for hit in search(question, k=10) if hit["kind"] == "product"]
    for product in [*shown, *in_catalogue]:
        name = plain(product["name"].partition(" - ")[0])
        if name and re.search(rf"\b{re.escape(name)}\b", q):
            return product

    # 2. By price or by being last. These need a list to choose from.
    if shown:
        if re.search(r"\b(cheapest|least expensive|lowest price)\b", q):
            return min(shown, key=lambda product: product["price"])
        if re.search(r"\b(most expensive|priciest|best quality)\b", q):
            return max(shown, key=lambda product: product["price"])
        if re.search(r"\b(the last|last one)\b", q):
            return shown[-1]

    # 3. By position.
    match = ORDINAL.search(q)
    if match and shown:
        index = position(match)
        return shown[index] if index < len(shown) else None

    # 4. Nothing else to choose from.
    return shown[0] if len(shown) == 1 else None


def pick_products(question: str, shown: list[dict]) -> list[dict]:
    """Every product the customer picks in one message. An empty list when it is not clear.

    "Both", "MALM and HEMNES", "the first and the third". With a single pick this
    gives the same answer as pick_product().
    """
    q = plain(ordinals(question))

    # "Both", "all of them": everything that was shown.
    if shown and re.search(r"\b(both|all of them|all (two|three|3|2)|everything)\b", q):
        return list(shown)

    # Two or more of the shown products named.
    named = [
        product
        for product in shown
        if any(re.search(rf"\b{re.escape(name)}\b", q) for name in product_names(product))
    ]
    if len(named) > 1:
        return named

    # Two or more positions: "the first and the third". Repeats are dropped, order kept.
    indexes = [position(match) for match in ORDINAL.finditer(q)]
    indexes = list(dict.fromkeys(index for index in indexes if index < len(shown)))
    if len(indexes) > 1:
        return [shown[index] for index in indexes]

    # Otherwise it is a single pick.
    one = pick_product(question, shown)
    return [one] if one else []
