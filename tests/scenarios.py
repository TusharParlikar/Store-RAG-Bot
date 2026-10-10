"""End-to-end chat scenarios against the real index and LLM. Run: python -m tests.scenarios

Each scenario: question, expected kind of reply, and text that must appear.
kinds: answer (normal reply), missing (not-available line + substitute), idk (refusal)
A must-have string starting with "!" must NOT appear.
An optional 4th item gives answer() earlier user messages ("history") and, as a search query, the products
the bot just listed ("shown").
"""
import sys
import time

from gen.answer import IDK, answer
from rag.index import search

SCENARIOS = [
    # prices and products in stock
    ("How much is the NORDVIKEN bar table?", "answer", ["₹23,382"]),
    ("What does the LACK coffee table 118x78 cost?", "answer", ["₹3,502"]),
    ("Do you have a coffee table?", "answer", ["1. **", "₹"]),
    ("I need a wardrobe for my bedroom", "answer", ["wardrobe"]),
    ("Do you sell sofas?", "answer", ["sofa"]),
    ("Do you have a bunk bed?", "answer", ["bunk bed"]),
    ("Do you have a hammock for the garden?", "answer", ["hammock"]),
    # policies
    ("Can I return an assembled chair?", "answer", ["180"]),
    ("What is your return window?", "answer", ["365"]),
    ("What does the warranty not cover?", "answer", ["wear"]),
    # needs
    ("I have back pain from sitting all day", "answer", ["₹", "sorry"]),
    # needs and feelings: empathy first, real furniture, no spare parts
    ("my leg is broken suggest me something", "answer", ["sorry", "₹", "!- Leg"]),
    ("I'm feeling stressed and can't sleep well", "answer", ["sorry", "₹"]),
    ("my baby is coming next month", "answer", ["₹", "!sorry"]),
    ("my mother is old and finds it hard to get up from the sofa", "answer", ["₹", "!Armrest**"]),
    # not in stock: honest line + close suggestion
    ("Do you sell desk lamps?", "missing", ["not available"]),
    ("Do you sell ceiling fans?", "missing", ["not available"]),
    ("I need a fridge for my kitchen", "missing", ["not available"]),
    ("I want to buy a piano", "missing", ["not available"]),
    ("Do you sell carpets?", "missing", ["not available"]),
    # off topic
    ("What is the capital of France?", "idk", [IDK]),
    ("Who won the football world cup?", "idk", [IDK]),
    # chit-chat: friendly reply, no products
    ("hi there, I want to build this chatbot", "answer", ["!₹"]),
    # complaint: empathy, then policy
    ("the chair I bought arrived broken and I'm really upset", "answer", ["sorry", "!₹"]),
    # memory: the earlier message explains the new one
    ("I need something for my room", "answer", ["₹", "!- Leg", "!sorry"], {"history": ["my leg is broken"]}),
    # buying: details + next step; ask which one when several were listed
    ("HATTEFJÄLL office chair, I would like to buy this, tell me more", "answer",
     ["HATTEFJÄLL", "Warranty: 60 months", "your cart", "!sorry"]),
    ("ok ill get that", "answer", ["which one", "1. **", "3. **", "!4. **"],
     {"history": ["my neck hurts"], "shown": "office chair with armrests"}),
    ("the second one", "answer", ["Price: ₹", "your cart"], {"shown": "office chair with armrests"}),
    ("2", "answer", ["Price: ₹", "your cart"], {"shown": "office chair with armrests"}),
    # more feelings and life events: empathy first, real furniture, a next step towards buying, no medical promises
    ("my father had knee surgery and comes home next week", "answer", ["sorry", "₹", "would you like", "!- Leg", "!cure"]),
    ("I work from home and my back hurts by evening", "answer", ["sorry", "₹", "would you like", "!cure"]),
    ("we just got married and moved into a new flat", "answer", ["congrat", "₹", "would you like", "!sorry"]),
    ("I feel lonely since I moved to a new city", "answer", ["₹", "would you like"]),
    ("my son starts school next month and needs a place to study", "answer", ["₹", "would you like", "!sorry"]),
    ("my grandmother finds it hard to get out of bed", "answer", ["sorry", "₹", "!cure"]),
    # general shopping: every product list ends with a next step
    ("Do you have a dining table for 6 people?", "answer", ["1. **", "₹", "would you like"]),
    ("Which is better for a small room, a sofa-bed or a daybed?", "answer", ["₹"]),
    ("I need a desk under ₹10,000", "answer", ["₹", "would you like"]),
    # general chat and questions
    ("hello!", "answer", ["!₹"]),
    ("thank you so much", "answer", ["!₹"]),
    ("how do I cook biryani?", "idk", [IDK]),
    ("where is my order ORD-0042?", "answer", ["!₹"]),
    # fresh set: written after the prompts were tuned, to check they generalise
    ("I'm recovering from back surgery and can't bend much", "answer", ["sorry", "₹", "would you like", "!cure"]),
    ("my husband snores and I can't sleep", "answer", ["₹", "would you like"]),
    ("I just got promoted and want a nicer home office", "answer", ["congrat", "₹", "would you like", "!sorry"]),
    ("my wife is pregnant and gets tired standing in the kitchen", "answer", ["₹", "would you like", "!cure"]),
    ("I'm a student with a very small room", "answer", ["₹", "would you like"]),
    ("my elderly parents are moving in with us", "answer", ["₹", "would you like"]),
    ("our house got flooded and we lost our furniture", "answer", ["sorry", "₹", "would you like"]),
    ("What is the cheapest bed you have?", "answer", ["₹"]),
    ("Do you have something to store shoes?", "answer", ["₹", "would you like"]),
    ("Can I get a refund without my order number?", "answer", ["credit", "!₹"]),
    ("Do you sell laptops?", "missing", ["not available"]),
    ("what's the weather like today?", "idk", [IDK]),
    ("good morning", "answer", ["!₹"]),
    ("the drawer of my wardrobe broke after two months", "answer", ["warranty", "!₹"]),
    ("tell me more about the POÄNG armchair, I want to buy it", "answer", ["POÄNG", "Price: ₹", "your cart"]),
    ("I'll take the first one", "answer", ["Price: ₹", "your cart"], {"shown": "comfortable armchair"}),
    ("number 3 please", "answer", ["Price: ₹", "your cart"], {"shown": "dining table"}),
    ("I'm so happy, my daughter got into college!", "answer", ["congrat", "₹", "!sorry"]),
    # fresh set 2: written after fixing fresh set 1, never tuned on
    ("I just adopted a cat and she scratches everything", "answer", ["₹", "would you like"]),
    ("my back is stiff every morning", "answer", ["sorry", "₹", "!cure"]),
    ("we're hosting Diwali dinner for 15 relatives", "answer", ["₹", "would you like"]),
    ("I started night shifts and can't sleep during the day", "answer", ["₹", "would you like"]),
    ("my two kids keep fighting over one desk", "answer", ["₹", "would you like"]),
    ("Do you have a couch for a small living room?", "answer", ["₹", "would you like"]),
    ("Do you sell air conditioners?", "missing", ["not available"]),
    ("Can I return a cut-to-size worktop?", "answer", ["cut", "!₹"]),
    ("the leg of my new table snapped", "answer", ["!₹"]),
    ("who are you?", "answer", ["!₹"]),
    ("write me a poem about the sea", "answer", ["only help", "!₹"]),
    ("ok I'll take it", "answer", ["Price: ₹", "your cart"], {"shown": "POÄNG rocking-chair", "n": 1}),
    # from a real chat: "1 st" typed with a space, and no second "sorry" on a follow-up
    ("i want to take 1 st chair and this 1st chair", "answer", ["Price: ₹", "Product page"],
     {"history": ["i have broken arm, suggest me something", "Show me cafe furniture"], "shown": "cafe chair"}),
    ("Show me cafe furniture", "answer", ["₹", "!sorry"], {"history": ["i have broken arm, suggest me something"]}),
    ("1st", "answer", ["Price: ₹", "Product page"], {"shown": "office chair"}),
    # products named in the message get direct links; everyday words do not
    ("How much is the MALM bed?", "answer", ["MALM", "Product pages:", "https://"]),
    ("I lack space in my bedroom", "answer", ["₹", "!Product pages:"]),
]


def kind_of(r: dict) -> str:
    if r["text"] == IDK:
        return "idk"
    return "missing" if r["missing"] else "answer"


def main(rounds: int = 1):
    fails = 0
    for q, want, must, *extra in SCENARIOS:
        opts = extra[0] if extra else {}
        shown = [h for h in search(opts["shown"]) if h["kind"] == "product"][:opts.get("n", 3)] if "shown" in opts else []
        for _ in range(rounds):
            t = time.time()
            r = answer(q, history=opts.get("history", []), shown=shown)
            got = kind_of(r)
            text = r["text"].lower()
            ok = got == want and all((m[1:].lower() not in text) if m.startswith("!") else (m.lower() in text) for m in must)
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {time.time() - t:5.1f}s  [{got}] {q}")
            if not ok:
                print(f"      want [{want}] containing {must}\n      understood: {r.get('understanding')}\n      got: {r['text'][:300]!r}")
    total = len(SCENARIOS) * rounds
    print(f"\n{total - fails}/{total} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 1) else 0)
