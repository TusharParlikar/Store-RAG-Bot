"""The cart: the products the customer has picked so far. Plain Python, no LLM.

A cart is a list of product dicts. The chat page keeps it between messages
and passes it to answer() with every message.

  add()      put picked products in the cart
  command()  handle messages about the cart itself: check out, show, remove, empty
  listing(), size(), added_line()   the texts shown to the customer
"""

import re

from nlp.chunks import inr

# --------------------------------------------------------------------------------------
# What the customer can say about the cart
# --------------------------------------------------------------------------------------

# "Check out" always means checkout.
CHECKOUT = re.compile(
    r"\b(check\s?-?out|place (the|my) order|proceed to (pay|buy|payment|order)"
    r"|ready to (buy|pay|order)|buy (them|these|all))\b",
    re.I,
)

# "That's all" or "I'm done" means checkout only when something is in the cart.
DONE = re.compile(
    r"\b(that'?s all|that is all|that'?s it|i'?m done|i am done|done shopping|nothing else"
    r"|no more)\b",
    re.I,
)

# "Show my cart", or just "cart?".
VIEW = re.compile(
    r"\b(show|see|view|what'?s in|what is in|check)\b.{0,15}\b(cart|basket|bag)\b"
    r"|^\s*(my )?(cart|basket)\s*\??\s*$",
    re.I,
)

# "Empty my cart", "start over".
CLEAR = re.compile(r"\b(empty|clear|reset)\b.{0,15}\b(cart|basket|bag)\b|\bstart over\b", re.I)

# "Remove 2". Not "don't want": "I don't want a big sofa" is a search, not a removal.
REMOVE = re.compile(r"\b(remove|delete|drop|take out)\b", re.I)

# A position in the cart: "2", "second", "2nd".
POSITION = re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|[1-9])\b")
POSITION_WORDS = ("first", "1st", "second", "2nd", "third", "3rd", "fourth", "4th", "fifth", "5th")

# --------------------------------------------------------------------------------------
# Fixed texts
# --------------------------------------------------------------------------------------

MORE = "Tell me what else you need, or say **check out** when you are done."
EMPTY = "Your cart is empty. What are you looking for today?"
EMPTIED = "Your cart is empty now. What are you looking for today?"
THANKS = "Open each link to buy the product on its page. Thank you for shopping with us!"


# --------------------------------------------------------------------------------------
# Adding
# --------------------------------------------------------------------------------------


def key(product: dict):
    """What makes two cart entries the same product."""
    return product.get("item_id") or (product["name"], product["price"])


def add(cart: list[dict], products: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (the new cart, the products that were not in it yet).

    A product already in the cart is not added a second time.
    """
    in_cart = {key(product) for product in cart}
    new = []
    for product in products:
        if key(product) not in in_cart:
            in_cart.add(key(product))
            new.append(product)
    return cart + new, new


# --------------------------------------------------------------------------------------
# Texts
# --------------------------------------------------------------------------------------


def listing(cart: list[dict]) -> str:
    """A numbered list of the cart, each product linked to its page."""
    rows = []
    for number, product in enumerate(cart, 1):
        name = product["name"]
        if product.get("link"):
            name = f"[{name}]({product['link']})"
        rows.append(f"{number}. {name} – {inr(product['price'])}")
    return "\n".join(rows)


def size(cart: list[dict]) -> str:
    """ "2 items, total ₹12,345" """
    count = len(cart)
    total = sum(product["price"] for product in cart)
    return f"{count} item{'s' if count != 1 else ''}, total {inr(total)}"


def added_line(cart: list[dict], new: list[dict]) -> str:
    """The line under a pick: what happened, the cart size, and what to do next."""
    what = "Added to your cart" if new else "That is already in your cart"
    return f"🛒 {what} ({size(cart)}). {MORE}"


def summary(cart: list[dict]) -> str:
    """The cart with its size, or the "empty" text."""
    if not cart:
        return EMPTY
    return f"Your cart ({size(cart)}):\n{listing(cart)}\n\n{MORE}"


# --------------------------------------------------------------------------------------
# Messages about the cart
# --------------------------------------------------------------------------------------


def to_remove(question: str, cart: list[dict]) -> list[dict]:
    """The cart products a "remove ..." message points at: by name, else by position."""
    q = question.lower()

    by_name = [product for product in cart if product["name"].partition(" - ")[0].lower() in q]
    if by_name:
        return by_name

    indexes = []
    for word in POSITION.findall(q):
        if word in POSITION_WORDS:
            indexes.append(POSITION_WORDS.index(word) // 2)
        else:
            indexes.append(int(word) - 1)
    by_position = [cart[index] for index in indexes if index < len(cart)]
    if by_position:
        return by_position

    # "Remove it" with one product in the cart can only mean that one.
    return cart if len(cart) == 1 else []


def command(question: str, cart: list[dict]) -> tuple[str, list[dict]] | None:
    """Handle a message that is about the cart itself.

    Returns (reply, new cart), or None when the message is about something else
    and the normal answer steps should run.
    """
    # Check out: the order with a link to every product. The cart is then empty.
    if CHECKOUT.search(question) or (cart and DONE.search(question)):
        if not cart:
            return EMPTY, cart
        return f"Here is your order ({size(cart)}):\n{listing(cart)}\n\n{THANKS}", []

    # Empty the cart.
    if CLEAR.search(question):
        return EMPTIED, []

    # Show the cart.
    if VIEW.search(question):
        return summary(cart), cart

    # Remove something.
    if cart and REMOVE.search(question):
        gone = to_remove(question, cart)
        if not gone:
            return f"Which one should I remove? Tell me its number.\n{listing(cart)}", cart
        left = [product for product in cart if product not in gone]
        names = ", ".join(product["name"] for product in gone)
        return f"Removed {names}.\n\n{summary(left)}", left

    return None
