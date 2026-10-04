"""Step 4: build the prompt, call the LLM, return the answer with its sources.

Also handles products the store does not carry: the bot says so, logs the
request in data/requests/requests.csv, and suggests the closest product it has.
"""
import csv
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

import config
from nlp.chunks import product_chunks
from rag.index import search

# Below this best-match score the question is not about the store: answer without the LLM.
# Picked by testing off-topic questions (see `python -m gen.answer`).
MIN_SCORE = 0.30

IDK = "I don't know. Please contact the store and the team will be happy to help."

# Said when a requested product is not in the data. Keep it true: no fake scarcity.
NOT_AVAILABLE = "Sorry, {item} is not available in our store right now. I have passed your request to our team so they can look at adding it."

REQUESTS = Path(__file__).resolve().parent.parent / "data" / "requests" / "requests.csv"

SYSTEM = f"""You are the friendly assistant of a furniture store in India. Prices are in Indian rupees (₹).
Rules:
1. Answer only from the context below.
2. If the context does not answer the question, reply exactly: "{IDK}"
3. Quote prices and warranty lengths exactly as written in the context.
4. Mention which product or policy the answer comes from.
5. Keep answers short.
6. Describe benefits at comfort level only. No medical promises.
7. If a "Warranty check (computed)" line is given, use its status and date as-is. Never calculate dates yourself."""

# Added to SYSTEM only when products are in the context: the small model copies a list template into everything.
PRODUCT_FORMAT = """
Format: start with a short, warm greeting, then put each product on its own numbered line:
the product name in bold, an en dash, the ₹ price, then one friendly sentence on what it is and why it suits the customer.
Example:
Hello! Here is what we have for you:
1. **NAME - Product, size** – ₹price. What it is and why it suits them.
2. **NAME - Product, size** – ₹price. What it is and why it suits them.
List at most 3 products, the best matches first. Copy names and prices exactly from the context."""

POLICY_FORMAT = "\nFormat: answer in 1 to 3 plain sentences. No list, no prices."

# Few-shot: a 1.7b model follows examples far better than instructions.
ROUTE = """Read the customer's message and reply with exactly one line:
PRODUCT: <kind of product>  if they ask for, want or need any kind of item, even one a furniture store may not sell
NEED: <furniture that would help>  if they describe a problem, pain, injury, feeling, life event or situation.
  Name only furniture (chairs, armchairs, sofas, beds, footstools, tables, desks, storage), never medical items.
HAPPY: <furniture that would help>  same as NEED, but for good news (a baby, a new home, a wedding, a new job)
NONE  only if it is about a store policy (returns, warranty, delivery) or general knowledge unrelated to shopping

Customer: do you have a bunk bed?
PRODUCT: bunk bed

Customer: How much is the MALM bed frame?
PRODUCT: bed frame

Customer: do you sell outdoor heaters?
PRODUCT: outdoor heater

Customer: I need a mirror for my hallway
PRODUCT: mirror

Customer: do you sell televisions?
PRODUCT: television

Customer: I sprained my ankle and can't walk much
NEED: armchair with armrests, footstool

Customer: my neck hurts when I work, what can help?
NEED: office chair with headrest, desk

Customer: I just moved into a tiny flat and have no space
NEED: compact storage, sofa-bed, wall shelf

Customer: I broke my arm last week
NEED: armchair with armrests, side table, footstool

Customer: I'm stressed after work and can't relax
NEED: comfortable armchair, sofa, footstool

Customer: I can't sleep well at night
NEED: comfortable bed, bedside table

Customer: we are expecting a baby soon
HAPPY: crib, changing table, nursery storage

Customer: we just bought our first house
HAPPY: sofa, dining table, bed

Customer: my kids keep leaving toys everywhere
NEED: children's storage, toy boxes

Customer: can I return a chair after assembling it?
NONE

Customer: what does the warranty cover?
NONE

Customer: who is the prime minister?
NONE"""

UNAVAILABLE_TASK = ("Task: the store does NOT sell {item}; the customer has already been told. "
                    "Start your reply with \"Here is something close you might like:\" and suggest 1 or 2 products from the context "
                    "that could do a similar job, as a numbered list (bold name – ₹ price – benefit). Do not claim they do what {item} does. "
                    "If nothing in the context is a sensible substitute, only say which kinds of furniture the store does have.")

NEED_TASK = ("Task: the customer told you about their situation. {opening} "
             "Then suggest up to 3 products from the context that could make them more comfortable, and for each explain "
             "in one sentence how it helps in their situation. Comfort level only: no medical advice or promises.")

_client = None


def llm(messages: list[dict], temperature: float = config.LLM_TEMPERATURE) -> str:
    global _client
    _client = _client or OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    # reasoning_effort="none" stops Qwen3 thinking (about 30x faster); max_tokens stops runaway replies.
    out = _client.chat.completions.create(model=config.LLM_MODEL, messages=messages, temperature=temperature,
                                          max_tokens=400, reasoning_effort=config.LLM_REASONING_EFFORT)
    text = re.sub(r"<think>.*?</think>", "", out.choices[0].message.content or "", flags=re.S)
    return re.sub(r"/(no_)?think", "", text, flags=re.I).strip()  # Ollama sometimes echoes Qwen3's switch


def log_request(item: str, question: str):
    REQUESTS.parent.mkdir(exist_ok=True)
    new = not REQUESTS.exists()
    with REQUESTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "item", "question"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), item, question])


# Words the extractor sometimes returns that are needs or policies, never a missing product.
# ponytail: hand list; grows if the request log shows false "not available" replies.
NOT_PRODUCTS = {"none", "pain", "ache", "hurt", "relief", "comfort", "support", "warranty", "guarantee", "return", "returns",
                "refund", "window", "policy", "delivery", "price", "cost", "order", "assembly", "furniture", "product"}


# Body parts and medical items that are also furniture part names ("Leg", "Arm"); dropped from need searches.
NOT_FURNITURE = {"leg", "legs", "arm", "arms", "foot", "feet", "hand", "hands", "knee", "ankle", "wrist", "neck",
                 "brace", "cast", "crutch", "crutches", "bandage", "splint", "support", "rest"}


# Spare parts ("STOCKSUND - Legs for armchair") are no help to someone describing a need.
PART = re.compile(r"(legs?|supporting leg|armrest|backrest|back rest|cover|slipcover|knob|handle|door|drawer|shelf|"
                  r"hinge|frame|cushion cover|glass door|plinth|rail)\b", re.I)


@lru_cache(maxsize=1)
def catalogue_words() -> frozenset[str]:
    """Every word in product names and categories: what kinds of product the store has."""
    text = " ".join(f"{c['name']} {c['category']}" for c in product_chunks()).lower()
    return frozenset(re.findall(r"[a-z]+", text))


def in_catalogue(word: str) -> bool:
    words = catalogue_words()
    return any(w in words for w in (word, word + "s", word.removesuffix("s"), word.removesuffix("es")))


def route(question: str) -> tuple[str, str]:
    """('product', kind) | ('need' or 'happy', furniture search text) | ('none', '').

    The LLM only labels the message (temperature 0); Python decides what to do with it.
    """
    # Instructions in the system message: in one user message the model echoes "/no_think" instead of answering.
    reply = llm([{"role": "system", "content": ROUTE}, {"role": "user", "content": f"Customer: {question}"}],
                temperature=0)
    line = (reply.splitlines() or ["NONE"])[0].strip(" *")
    label, _, rest = line.partition(":")
    label, rest = label.strip().lower(), rest.strip(" ./*\"'").lower()
    if label in ("product", "need", "happy") and rest and rest != "none":
        return label, rest
    if not rest and label and label != "none":  # the model often drops the "PRODUCT:" label: "desk lamp"
        return "product", label.strip(" ./*\"'")
    return "none", ""


def is_missing(kind: str) -> bool:
    """True when the store has nothing of this kind: its main word is in no product name or category."""
    head = re.findall(r"[a-z]+", re.split(r"\b(?:with|for|in|that|which)\b", kind)[0])  # "bunk bed with a slide" -> bunk, bed
    if not head or NOT_PRODUCTS & set(re.findall(r"[a-z]+", kind)):
        return False
    return not in_catalogue(head[-1])


def answer(question: str, extra_context: str = "") -> dict:
    """Returns {text, sources, missing}. extra_context carries computed facts such as a warranty check."""
    hits = search(question)

    # Policy question (a policy section beats every product): give the model only policies,
    # otherwise it lists loosely matching products ("return window" -> a window table).
    best_rule = max((h["score"] for h in hits if h["kind"] == "rule"), default=0)
    policy = best_rule >= max(h["score"] for h in hits if h["kind"] == "product") and not extra_context
    label, detail = ("none", "") if policy or extra_context else route(question)

    if label in ("need", "happy"):
        # Search for the furniture that helps, not the words of the problem ("my leg is broken" -> table legs).
        query = " ".join(w for w in re.findall(r"[a-z'-]+", detail) if w not in NOT_FURNITURE) or detail
        hits = [h for h in search(query, k=20) if not PART.match(h.get("name", "").partition(" - ")[2])][:6]
    elif hits[0]["score"] < MIN_SCORE and not extra_context:
        return {"text": IDK, "sources": [], "missing": None}
    if policy:
        hits = [h for h in hits if h["kind"] == "rule"]

    missing = detail if label == "product" and is_missing(detail) else None
    task = ""
    if missing:
        task = UNAVAILABLE_TASK.format(item=missing) + "\n"
    elif label == "need":
        task = NEED_TASK.format(opening="Start with one short, warm sentence of sympathy in your own words about exactly what they said.") + "\n"
    elif label == "happy":
        task = NEED_TASK.format(opening="Start by congratulating them warmly in your own words on exactly what they said.") + "\n"

    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    if extra_context:
        context += "\n" + extra_context
    text = llm([
        {"role": "system", "content": SYSTEM + (POLICY_FORMAT if policy or extra_context else PRODUCT_FORMAT)},
        {"role": "user", "content": f"Context:\n{context}\n\n{task}Customer: {question}"},
    ])
    if missing:
        # Small models often repeat the "we don't sell it" line; drop that leading sentence.
        text = re.sub(r"(the store|we) (does|do) not (sell|have|carry)[^.]*\.\s*", "", text, flags=re.I).strip()
        text = NOT_AVAILABLE.format(item=missing) + "\n\n" + text
        log_request(missing, question)
    return {"text": text, "sources": hits, "missing": missing}


if __name__ == "__main__":
    for q in ["my leg is broken, suggest me something", "I have back pain from sitting all day",
              "Do you sell desk lamps?", "Do you have a coffee table?", "What is the capital of France?"]:
        r = answer(q)
        print(f"\n> {q}\n{r['text']}\nmissing={r['missing']}")
