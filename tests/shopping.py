"""Whole shopping trips against the real index and LLM: ask, add, add more, check out. Run: python -m tests.shopping

Each turn: what the customer types, then strings the reply must contain ("!x" = must not contain).
An optional third item is a position: the product added must be the one the last reply listed at that number.
The cart and the products shown last are carried between turns the same way the chat page does it.
"""

import re
import sys
import time

from gen.answer import answer
from gen.picking import listed

TRIPS = {
    "need, add one, add another kind, check out": [
        ("my back hurts after sitting all day", ["sorry", "₹", "add it to your cart"]),
        ("1", ["Added to your cart (1 item", "check out"], 1),
        ("do you have a desk?", ["₹", "add it to your cart"]),
        ("the second one", ["Added to your cart (2 items"], 2),
        ("ok lets check out", ["Here is your order (2 items", "https://", "!Added"]),
        ("show my cart", ["cart is empty"]),
    ],
    "two picks in one message, then done": [
        ("show me coffee tables", ["₹"]),
        ("i'll take the first and the third", ["Good choices", "Added to your cart (2 items"]),
        ("that's all", ["Here is your order (2 items", "https://"]),
    ],
    "add, change mind, remove, check out": [
        ("I need a bookcase", ["₹"]),
        ("add the first one to my cart", ["Added to your cart (1 item"], 1),
        ("show me wardrobes", ["₹"]),
        ("2", ["Added to your cart (2 items"], 2),
        ("show my cart", ["Your cart (2 items", "1. ", "2. "]),
        ("remove 1", ["Removed", "Your cart (1 item"]),
        ("check out", ["Here is your order (1 item", "https://"]),
    ],
    "same product twice, and a named product": [
        ("do you have office chairs?", ["₹"]),
        ("1", ["Added to your cart (1 item"], 1),
        ("add the first one to my cart", ["already in your cart (1 item"]),
        ("I want to buy the POÄNG rocking-chair", ["POÄNG", "Added to your cart (2 items"]),
        ("checkout", ["Here is your order (2 items", "POÄNG"]),
    ],
    "check out with nothing, questions do not touch the cart": [
        ("check out", ["cart is empty"]),
        ("hello", ["!cart is empty", "!₹"]),
        ("What is your return window?", ["365", "!Added"]),
        ("do you sell laptops?", ["not available"]),
        ("I'm done", ["!Here is your order"]),
    ],
}


def main():
    fails = total = 0
    for name, turns in TRIPS.items():
        print(f"\n##### {name}")
        cart, history, last = [], [], {"content": "", "sources": []}
        for q, must, *spot in turns:
            shown = listed(last["content"], last["sources"])  # as app/main.py works it out
            numbered = re.findall(
                r"^\d+\. \*\*(.+?)\*\*", last["content"], re.M
            )  # names as the customer reads them
            t = time.time()
            r = answer(q, history=history, shown=shown, cart=cart)
            cart, last = r["cart"], {"content": r["text"], "sources": r["sources"]}
            history.append(q)
            low = r["text"].lower()
            bad = [
                m
                for m in must
                if (m[1:].lower() in low if m.startswith("!") else m.lower() not in low)
            ]
            if spot and (
                len(numbered) < spot[0] or not cart or cart[-1]["name"] != numbered[spot[0] - 1]
            ):
                bad.append(
                    f"picked {cart[-1]['name'] if cart else None!r}, number {spot[0]} was {numbered[spot[0] - 1:spot[0]]}"
                )
            total += 1
            fails += bool(bad)
            print(
                f"{'PASS' if not bad else 'FAIL'} {time.time() - t:5.1f}s  cart={len(cart)}  {q!r}"
            )
            if bad:
                print(
                    f"      missing/unwanted: {bad}\n      understood: {r['understanding']['intent']}\n      got: {r['text'][:400]!r}"
                )
    print(f"\n{total - fails}/{total} turns passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
