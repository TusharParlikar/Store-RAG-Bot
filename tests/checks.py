"""Fast checks of the plain-Python parts. No LLM needed.

  python -m tests.checks

Each function below checks one module. The warranty maths has its own: python -m gen.warranty
"""

from gen import cart
from gen.answer import said_now
from gen.catalogue import distinct, head_noun, is_missing, named_products, product_names
from gen.picking import BUY, buys_by_name, listed, ordinals, pick_product, pick_products
from gen.understand import is_broken_furniture, parse_understanding


def understanding():
    """gen/understand.py: cleaning the model's JSON, and telling furniture from body parts."""
    # JSON wrapped in code fences, wrong case, a string where a list is expected.
    reply = '```json\n{"intent": "product_search", "item": "Desk Lamp.", "furniture": "desk"}\n```'
    u = parse_understanding(reply)
    assert u["intent"] == "PRODUCT_SEARCH"
    assert u["item"] == "desk lamp"
    assert u["furniture"] == ["desk"]
    assert u["sentiment"] == "neutral"

    # Rubbish and unknown intents fall back to "".
    assert parse_understanding("not json")["intent"] == ""
    assert parse_understanding('{"intent": "HACK"}')["intent"] == ""

    # Furniture that broke is a complaint; a broken leg is not.
    assert is_broken_furniture("the leg of my new table snapped")
    assert is_broken_furniture("the drawer of my wardrobe broke")
    for message in ["my leg is broken", "I broke my arm", "my house got flooded"]:
        assert not is_broken_furniture(message), message

    # Is the problem in this message, or only in an earlier one?
    assert said_now("broken arm", "i have broken arm")
    assert said_now("stress after work", "I'm feeling stressed")
    assert said_now("", "anything")
    assert not said_now("broken leg", "I need something for my room")


def buying_words():
    """gen/picking.py: which messages pick a product."""
    picks = [
        "ok ill get that",
        "I'll take the second one",
        "i want this one",
        "2",
        " 3. ",
        "option 1",
        "1st",
        "the second one",
        "third.",
        "the cheapest one please",
        "actually show me the last one",
        "i'll take both",
        "add the first one to my cart",
    ]
    for message in picks:
        assert BUY.search(message), message

    not_picks = [
        "do you have a desk?",
        "my neck hurts",
        "show me other chairs",
        "tell me more about returns",
        "I have 2 kids",
        "I want 2 chairs",
        "first time buying a sofa",
        "I bought a sofa last year",
    ]
    for message in not_picks:
        assert not BUY.search(message), message

    # "1 st" typed with a space.
    assert BUY.search(ordinals("i want to take 1 st chair and this 1st chair"))

    # A buying wish that names a product is a pick even with no list before it.
    for message in ["I want to buy the POÄNG rocking-chair", "add the HEMNES to my cart"]:
        assert buys_by_name(message), message
    for message in ["How much is the MALM bed?", "I want to buy a sofa", "I lack space"]:
        assert not buys_by_name(message), message


def picking():
    """gen/picking.py: which product a pick means."""
    shown = [
        {"name": "HATTEFJÄLL - Office chair"},
        {"name": "NILSOVE - Chair"},
        {"name": "JÄRVFJÄLLET - Office chair"},
    ]

    # By name, without the accents.
    assert pick_product("I'll buy the hattefjall", shown) is shown[0]

    # By position.
    assert pick_product("tell me more about the second one", shown) is shown[1]
    assert pick_product("option 3 please", shown) is shown[2]
    assert pick_product("2", shown) is shown[1]
    assert pick_product(ordinals("i want to take 1 st chair"), shown) is shown[0]

    # "That" is not clear with three products, and clear with one.
    assert pick_product("ok I'll get that", shown) is None
    assert pick_product("ok I'll get that", shown[:1]) is shown[0]

    # By price, or the last one.
    priced = [
        {"name": "A - x", "price": 30},
        {"name": "B - y", "price": 10},
        {"name": "C - z", "price": 20},
    ]
    assert pick_product("the cheapest one please", priced)["price"] == 10
    assert pick_product("most expensive", priced)["price"] == 30
    assert pick_product("the last one", priced) is priced[2]

    # Several at once.
    assert pick_products("i'll take the first and the third", shown) == [shown[0], shown[2]]
    assert pick_products("both please", shown[:2]) == shown[:2]
    assert pick_products("hattefjall and nilsove", shown) == shown[:2]
    assert pick_products("the second one", shown) == [shown[1]]

    # "The second one" follows the order of the reply text, not of the search results.
    hits = [
        {"kind": "product", "name": "A - x", "price": 10},
        {"kind": "product", "name": "B - y", "price": 20},
        {"kind": "product", "name": "C - z", "price": 30},
    ]
    in_reply = listed("1. **B - y** – ₹20 2. **A - x** – ₹10", hits)
    assert [product["name"] for product in in_reply] == ["B - y", "A - x"]


def catalogue():
    """gen/catalogue.py: stock check, product names, links."""
    assert head_noun("Office chair with armrests") == "chair"
    assert head_noun("Laptop table, 100x36 cm") == "table"

    # Sold, not sold, and not a product at all.
    for kind in ["coffee table", "bunk bed with a slide", "sofa", "couch", "warranty"]:
        assert not is_missing(kind), kind
    for kind in ["laptop", "desk lamp", "piano", "ceiling fan"]:
        assert is_missing(kind), kind

    same = [{"name": "A", "price": 1, "source": "x"}, {"name": "A", "price": 1, "source": "y"}]
    assert len(distinct(same)) == 1

    # A named product is found; an everyday word is not, unless typed in capitals.
    malm = named_products("How much is the MALM bed?")
    assert malm and all("malm" in product_names(product) for product in malm)
    assert named_products("I lack space at home") == []
    assert named_products("hallo") == []
    lack = named_products("What does the LACK coffee table cost?")
    assert any("lack" in product_names(product) for product in lack)


def shopping_cart():
    """gen/cart.py: adding, and every cart command."""
    bed = {"item_id": 1, "name": "MALM - Bed", "price": 1000, "link": "https://x/a"}
    table = {"item_id": 2, "name": "LACK - Table", "price": 250000, "link": "https://x/b"}

    # Adding: the same product is not added twice.
    items, new = cart.add([], [bed, table, bed])
    assert items == [bed, table] and new == [bed, table]
    assert cart.add(items, [bed]) == (items, [])

    # Messages that are not about the cart.
    assert cart.command("do you have a sofa?", items) is None
    assert cart.command("I do not want a big sofa", items) is None
    assert cart.command("i'm done", []) is None

    # Check out: links and total, and the cart is emptied.
    text, left = cart.command("ok lets check out", items)
    assert left == []
    assert "https://x/a" in text and "https://x/b" in text
    assert "2 items, total ₹2,51,000" in text
    assert cart.command("that's all", items)[1] == []
    assert cart.command("check out", [])[0] == cart.EMPTY

    # Show, remove, empty.
    assert cart.command("show my cart", items)[1] == items
    assert cart.command("remove the malm", items)[1] == [table]
    assert cart.command("remove 2", items)[1] == [bed]
    assert cart.command("remove it", items)[1] == items  # not clear with two products
    assert cart.command("remove it", [bed])[1] == []
    assert cart.command("empty my cart", items)[1] == []


if __name__ == "__main__":
    for check in (understanding, buying_words, picking, catalogue, shopping_cart):
        check()
        print("ok", check.__name__)
