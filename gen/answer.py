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
Format: start with a one-line greeting, then put each product on its own numbered line:
the product name in bold, an en dash, the ₹ price, then its benefit from the context in a few words. Example:
Hello! We have:
1. **NAME - Product, size** – ₹price. Benefit in a few words.
2. **NAME - Product, size** – ₹price. Benefit in a few words.
List at most 3 products, the best matches first."""

POLICY_FORMAT = "\nFormat: answer in 1 to 3 plain sentences. No list, no prices."

# Few-shot: a 1.7b model follows examples far better than instructions.
EXTRACT = """Name the kind of product the customer wants to buy or asks about. Reply with the product kind only.
If the message is about a policy (returns, warranty, delivery), a feeling or need, or is not about products, reply NONE.

Customer: do you have a bunk bed?
Product: bunk bed

Customer: How much is the MALM bed frame?
Product: bed frame

Customer: do you sell outdoor heaters?
Product: outdoor heater

Customer: can I return a chair after assembling it?
Product: NONE

Customer: my neck hurts when I work, what can help?
Product: NONE

Customer: what does the warranty cover?
Product: NONE"""

UNAVAILABLE_TASK = ("Task: the store does NOT sell {item}; the customer has already been told. "
                    "Start your reply with \"Here is something close you might like:\" and suggest 1 or 2 products from the context "
                    "that could do a similar job, as a numbered list (bold name – ₹ price – benefit). Do not claim they do what {item} does. "
                    "If nothing in the context is a sensible substitute, only say which kinds of furniture the store does have.")

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


@lru_cache(maxsize=1)
def catalogue_words() -> frozenset[str]:
    """Every word in product names and categories: what kinds of product the store has."""
    text = " ".join(f"{c['name']} {c['category']}" for c in product_chunks()).lower()
    return frozenset(re.findall(r"[a-z]+", text))


def in_catalogue(word: str) -> bool:
    words = catalogue_words()
    return any(w in words for w in (word, word + "s", word.removesuffix("s"), word.removesuffix("es")))


def missing_item(question: str) -> str | None:
    """The kind of product asked for, if the store has nothing of that kind; else None.

    The LLM only names the product (a 1.7b model can't judge stock reliably).
    Python decides: the product is missing when its main word is in no product name or category.
    """
    # Instructions in the system message: in one user message the model echoes "/no_think" instead of answering.
    reply = llm([{"role": "system", "content": EXTRACT}, {"role": "user", "content": f"Customer: {question}\nProduct:"}],
                temperature=0)
    kind = (reply.splitlines() or ["none"])[0].lower().removeprefix("product:").strip(" ./*\"'")
    head = re.findall(r"[a-z]+", re.split(r"\b(?:with|for|in|that|which)\b", kind)[0])  # "bunk bed with a slide" -> bunk, bed
    if not head or NOT_PRODUCTS & set(re.findall(r"[a-z]+", kind)):
        return None
    return None if in_catalogue(head[-1]) else kind


def answer(question: str, extra_context: str = "") -> dict:
    """Returns {text, sources, missing}. extra_context carries computed facts such as a warranty check."""
    hits = search(question)
    if hits[0]["score"] < MIN_SCORE and not extra_context:
        return {"text": IDK, "sources": [], "missing": None}

    # Policy question (a policy section beats every product): give the model only policies,
    # otherwise it lists loosely matching products ("return window" -> a window table).
    best_rule = max((h["score"] for h in hits if h["kind"] == "rule"), default=0)
    policy = best_rule >= max(h["score"] for h in hits if h["kind"] == "product") and not extra_context
    if policy:
        hits = [h for h in hits if h["kind"] == "rule"]

    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    if extra_context:
        context += "\n" + extra_context
    missing = None if policy else missing_item(question)
    task = UNAVAILABLE_TASK.format(item=missing) + "\n" if missing else ""
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
    for q in ["How much is the NORDVIKEN bar table?", "Can I return an assembled chair?",
              "Do you sell desk lamps?", "Do you have a coffee table?", "Do you sell pianos?",
              "What is the capital of France?"]:
        r = answer(q)
        print(f"\n> {q}  (best score {search(q)[0]['score']:.2f})\n{r['text']}\nmissing={r['missing']}")
