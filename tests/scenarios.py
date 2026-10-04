"""End-to-end chat scenarios against the real index and LLM. Run: python -m tests.scenarios

Each scenario: question, expected kind of reply, and text that must appear.
kinds: answer (normal reply), missing (not-available line + substitute), idk (refusal)
"""
import sys
import time

from gen.answer import IDK, answer

SCENARIOS = [
    # prices and products in stock
    ("How much is the NORDVIKEN bar table?", "answer", ["995"]),
    ("What does the LACK coffee table 118x78 cost?", "answer", ["149"]),
    ("Do you have a coffee table?", "answer", ["coffee table"]),
    ("I need a wardrobe for my bedroom", "answer", ["wardrobe"]),
    ("Do you sell sofas?", "answer", ["sofa"]),
    ("Do you have a bunk bed?", "answer", ["bunk bed"]),
    ("Do you have a hammock for the garden?", "answer", ["hammock"]),
    # policies
    ("Can I return an assembled chair?", "answer", ["180"]),
    ("What is your return window?", "answer", ["365"]),
    ("What does the warranty not cover?", "answer", ["wear"]),
    # needs
    ("I have back pain from sitting all day", "answer", ["SAR"]),
    # not in stock: honest line + close suggestion
    ("Do you sell desk lamps?", "missing", ["not available"]),
    ("Do you sell ceiling fans?", "missing", ["not available"]),
    ("I need a fridge for my kitchen", "missing", ["not available"]),
    ("I want to buy a piano", "missing", ["not available"]),
    ("Do you sell carpets?", "missing", ["not available"]),
    # off topic
    ("What is the capital of France?", "idk", [IDK]),
    ("Who won the football world cup?", "idk", [IDK]),
]


def kind_of(r: dict) -> str:
    if r["text"] == IDK:
        return "idk"
    return "missing" if r["missing"] else "answer"


def main(rounds: int = 1):
    fails = 0
    for q, want, must in SCENARIOS:
        for _ in range(rounds):
            t = time.time()
            r = answer(q)
            got = kind_of(r)
            ok = got == want and all(m.lower() in r["text"].lower() for m in must)
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {time.time() - t:5.1f}s  [{got}] {q}")
            if not ok:
                print(f"      want [{want}] containing {must}\n      got: {r['text'][:300]!r}")
    total = len(SCENARIOS) * rounds
    print(f"\n{total - fails}/{total} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 1) else 0)
