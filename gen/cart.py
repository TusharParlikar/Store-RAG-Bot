"""The cart: products the customer has picked so far. Plain Python, no LLM.

The cart is a list of product dicts. The chat page keeps it between messages and passes it to answer().
`python -m gen.cart` runs the checks.
"""
import re

from nlp.chunks import inr

# "check out" always means checkout; "that's all" or "I'm done" only when something is in the cart.
CHECKOUT = re.compile(r"\b(check\s?-?out|place (the|my) order|proceed to (pay|buy|payment|order)|"
                      r"ready to (buy|pay|order)|buy (them|these|all))\b", re.I)
DONE = re.compile(r"\b(that'?s all|that is all|that'?s it|i'?m done|i am done|done shopping|nothing else|no more)\b", re.I)
VIEW = re.compile(r"\b(show|see|view|what'?s in|what is in|check)\b.{0,15}\b(cart|basket|bag)\b|^\s*(my )?(cart|basket)\s*\??\s*$", re.I)
CLEAR = re.compile(r"\b(empty|clear|reset)\b.{0,15}\b(cart|basket|bag)\b|\bstart over\b", re.I)
REMOVE = re.compile(r"\b(remove|delete|drop|take out)\b", re.I)  # not "don't want": "I don't want a big sofa" is a search
POSITION = re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|[1-9])\b")
WORDS = ("first", "1st", "second", "2nd", "third", "3rd", "fourth", "4th", "fifth", "5th")

MORE = "Tell me what else you need, or say **check out** when you are done."
EMPTY = "Your cart is empty. What are you looking for today?"


def key(p: dict):
    return p.get("item_id") or (p["name"], p["price"])


def add(cart: list[dict], products: list[dict]) -> tuple[list[dict], list[dict]]:
    """(new cart, the products that were not in it yet)."""
    have = {key(p) for p in cart}
    new = []
    for p in products:
        if key(p) not in have:
            have.add(key(p))
            new.append(p)
    return cart + new, new


def listing(cart: list[dict]) -> str:
    return "\n".join(f"{i}. [{p['name']}]({p['link']}) – {inr(p['price'])}" if p.get("link")
                     else f"{i}. {p['name']} – {inr(p['price'])}" for i, p in enumerate(cart, 1))


def size(cart: list[dict]) -> str:
    n = len(cart)
    return f"{n} item{'s' if n != 1 else ''}, total {inr(sum(p['price'] for p in cart))}"


def added_line(cart: list[dict], new: list[dict]) -> str:
    what = "Added to your cart" if new else "That is already in your cart"
    return f"🛒 {what} ({size(cart)}). {MORE}"


def command(question: str, cart: list[dict]) -> tuple[str, list[dict]] | None:
    """(reply, new cart) when the message is about the cart itself: check out, show, empty or remove. Else None."""
    if CHECKOUT.search(question) or cart and DONE.search(question):
        if not cart:
            return EMPTY, cart
        return (f"Here is your order ({size(cart)}):\n{listing(cart)}\n\n"
                "Open each link to buy the product on its page. Thank you for shopping with us!"), []
    if CLEAR.search(question):
        return "Your cart is empty now. What are you looking for today?", []
    if VIEW.search(question):
        return (f"Your cart ({size(cart)}):\n{listing(cart)}\n\n{MORE}" if cart else EMPTY), cart
    if cart and REMOVE.search(question):
        q = question.lower()
        gone = [p for p in cart if p["name"].partition(" - ")[0].lower() in q]
        if not gone:
            spots = [WORDS.index(m) // 2 if m in WORDS else int(m) - 1 for m in POSITION.findall(q)]
            gone = [cart[i] for i in spots if i < len(cart)]
        if not gone and len(cart) == 1:
            gone = cart
        if not gone:
            return f"Which one should I remove? Tell me its number.\n{listing(cart)}", cart
        left = [p for p in cart if p not in gone]
        names = ", ".join(p["name"] for p in gone)
        return f"Removed {names}.\n\n" + (f"Your cart ({size(left)}):\n{listing(left)}\n\n{MORE}" if left else EMPTY), left
    return None


if __name__ == "__main__":
    a = {"item_id": 1, "name": "MALM - Bed", "price": 1000, "link": "https://x/a"}
    b = {"item_id": 2, "name": "LACK - Table", "price": 250000, "link": "https://x/b"}
    cart, new = add([], [a, b, a])
    assert cart == [a, b] and new == [a, b] and add(cart, [a]) == (cart, [])
    assert command("do you have a sofa?", cart) is None and command("i'm done", []) is None
    assert command("check out", [])[0] == EMPTY
    text, left = command("ok lets check out", cart)
    assert left == [] and "https://x/a" in text and "https://x/b" in text and "2 items, total ₹2,51,000" in text
    assert command("that's all", cart)[1] == [] and command("show my cart", cart)[1] == cart
    assert command("remove the malm", cart)[1] == [b] and command("remove 2", cart)[1] == [a]
    assert command("remove it", cart)[1] == cart and command("remove it", [a])[1] == []
    assert command("empty my cart", cart)[1] == []
    assert command("I do not want a big sofa", cart) is None
    print("cart checks pass")
