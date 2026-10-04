"""Step 4: build the prompt, call the LLM, return the answer with its sources.

Also handles products the store does not carry: the bot says so, logs the
request in data/requests/requests.csv, and suggests the closest product it has.
"""
import csv
import json
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

INTENTS = {"PRODUCT_SEARCH", "PRODUCT_RECOMMENDATION", "PRODUCT_COMPARISON", "STORE_INFORMATION",
           "ORDER_SUPPORT", "COMPLAINT", "CASUAL_CONVERSATION", "GENERAL_QUESTION"}
POLICY_INTENTS = {"STORE_INFORMATION", "ORDER_SUPPORT", "COMPLAINT"}

# LLM #1: understand the customer before anything is searched. Few-shot: a 1.7b model follows examples
# far better than instructions. The model only describes the message; Python decides what to do with it.
UNDERSTAND = """Read the customer's message (and their earlier messages, if given) and describe it as one JSON object:
{"intent": one of PRODUCT_SEARCH, PRODUCT_RECOMMENDATION, PRODUCT_COMPARISON, STORE_INFORMATION, ORDER_SUPPORT, COMPLAINT, CASUAL_CONVERSATION, GENERAL_QUESTION,
 "item": the kind of item they name, even one a furniture store may not sell, else "",
 "problem": the problem, pain, injury, life event or situation they describe, else "",
 "emotion": one of pain, frustrated, angry, confused, excited, worried, disappointed, neutral, casual,
 "sentiment": positive, negative or neutral,
 "furniture": furniture that would help their situation, else [],
 "constraints": limits such as space, budget or room, else [],
 "meaning": one sentence on what they really need}
Intents:
PRODUCT_SEARCH: they name a kind of item they want. PRODUCT_COMPARISON: they compare items.
PRODUCT_RECOMMENDATION: they describe a problem, pain, feeling, life event or situation instead of naming an item.
STORE_INFORMATION: returns, warranty, delivery, assembly, store policy. ORDER_SUPPORT: an existing order.
COMPLAINT: something went wrong with their purchase. CASUAL_CONVERSATION: greetings, thanks, chit-chat, talk about themselves or this chatbot.
GENERAL_QUESTION: general knowledge unrelated to the store.
For "furniture" name only furniture (chairs, armchairs, recliners, sofas, beds, footstools, tables, desks, storage), never medical items.
If earlier messages explain the current one (an injury, a new baby), use them.

Customer: do you have a bunk bed?
{"intent": "PRODUCT_SEARCH", "item": "bunk bed", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Wants a bunk bed."}

Customer: How much is the MALM bed frame?
{"intent": "PRODUCT_SEARCH", "item": "bed frame", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Wants the price of the MALM bed frame."}

Customer: do you sell televisions?
{"intent": "PRODUCT_SEARCH", "item": "television", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Wants to buy a television."}

Customer: I need a mirror for my hallway
{"intent": "PRODUCT_SEARCH", "item": "mirror", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": ["hallway"], "meaning": "Wants a mirror for the hallway."}

Customer: which is better for a small room, a sofa-bed or a daybed?
{"intent": "PRODUCT_COMPARISON", "item": "sofa-bed", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": ["sofa-bed", "daybed"], "constraints": ["small room"], "meaning": "Wants to choose between a sofa-bed and a daybed for a small room."}

Customer: my leg is broken and I need something comfortable
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "broken leg", "emotion": "pain", "sentiment": "negative", "furniture": ["recliner", "armchair with armrests", "footstool"], "constraints": [], "meaning": "Needs comfortable seating that supports the leg while recovering."}

Customer: my neck hurts when I work, what can help?
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "neck pain while working", "emotion": "pain", "sentiment": "negative", "furniture": ["office chair with headrest", "desk"], "constraints": [], "meaning": "Needs a work setup that is easier on the neck."}

Customer: I just moved into a tiny flat and have no space
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "tiny flat with no space", "emotion": "worried", "sentiment": "neutral", "furniture": ["compact storage", "sofa-bed", "wall shelf"], "constraints": ["small space"], "meaning": "Needs furniture that saves space."}

Customer: I'm stressed after work and can't relax
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "stress after work", "emotion": "worried", "sentiment": "negative", "furniture": ["comfortable armchair", "sofa", "footstool"], "constraints": [], "meaning": "Needs a comfortable place to relax at home."}

Customer: we are expecting a baby soon
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "a baby is coming", "emotion": "excited", "sentiment": "positive", "furniture": ["crib", "changing table", "nursery storage"], "constraints": [], "meaning": "Needs to furnish a nursery."}

Earlier messages: my leg is broken
Customer: I need something for my room
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "broken leg", "emotion": "pain", "sentiment": "negative", "furniture": ["armchair with armrests", "footstool", "bedside table"], "constraints": ["bedroom"], "meaning": "Needs bedroom furniture that is easy to use with a broken leg."}

Customer: can I return a chair after assembling it?
{"intent": "STORE_INFORMATION", "item": "", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Asks about the return policy for assembled items."}

Customer: where is my order? it was supposed to come yesterday
{"intent": "ORDER_SUPPORT", "item": "", "problem": "late order", "emotion": "worried", "sentiment": "negative", "furniture": [], "constraints": [], "meaning": "Wants to know where a late order is."}

Customer: the table I bought arrived scratched, this is so annoying
{"intent": "COMPLAINT", "item": "table", "problem": "table arrived scratched", "emotion": "frustrated", "sentiment": "negative", "furniture": [], "constraints": [], "meaning": "Wants a damaged table fixed or replaced."}

Customer: hi! I want to build this chatbot
{"intent": "CASUAL_CONVERSATION", "item": "", "problem": "", "emotion": "casual", "sentiment": "positive", "furniture": [], "constraints": [], "meaning": "Greets and chats about building the chatbot."}

Customer: who is the prime minister?
{"intent": "GENERAL_QUESTION", "item": "", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Asks a general knowledge question."}"""

CASUAL = """You are the friendly assistant of a furniture store in India. The customer is just chatting.
Reply warmly in 1 or 2 short sentences, then offer to help with furniture, prices, warranty or returns.
Never state product names, prices or store policies here, and never answer general knowledge questions."""

UNAVAILABLE_TASK = ("Task: the store does NOT sell {item}; the customer has already been told. "
                    "Start your reply with \"Here is something close you might like:\" and suggest 1 or 2 products from the context "
                    "that could do a similar job, as a numbered list (bold name – ₹ price – benefit). Do not claim they do what {item} does. "
                    "If nothing in the context is a sensible substitute, only say which kinds of furniture the store does have.")

NEED_TASK = ("Task: the customer told you about their situation. {opening} "
             "Then suggest up to 3 products from the context that could make them more comfortable, and for each explain "
             "in one sentence how it helps in their situation. Comfort level only: no medical advice or promises.")

_client = None


def llm(messages: list[dict], temperature: float = config.LLM_TEMPERATURE, json_mode: bool = False) -> str:
    global _client
    _client = _client or OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    # reasoning_effort="none" stops Qwen3 thinking (about 30x faster); max_tokens stops runaway replies.
    out = _client.chat.completions.create(model=config.LLM_MODEL, messages=messages, temperature=temperature,
                                          max_tokens=400, reasoning_effort=config.LLM_REASONING_EFFORT,
                                          **({"response_format": {"type": "json_object"}} if json_mode else {}))
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


def parse_understanding(reply: str) -> dict:
    """Clean the model's JSON. Anything missing or malformed falls back to empty values (intent "")."""
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        raw = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        raw = {}
    raw = raw if isinstance(raw, dict) else {}

    def text(key):
        v = raw.get(key)
        return v.strip(" ./*\"'").lower() if isinstance(v, str) else ""

    def items(key):
        v = raw.get(key)
        v = v if isinstance(v, list) else [v] if isinstance(v, str) else []
        return [s.strip().lower() for s in v if isinstance(s, str) and s.strip()]

    intent = text("intent").upper()
    return {"intent": intent if intent in INTENTS else "", "item": text("item"), "problem": text("problem"),
            "emotion": text("emotion") or "neutral", "sentiment": text("sentiment") or "neutral",
            "furniture": items("furniture"), "constraints": items("constraints"),
            "meaning": raw["meaning"].strip() if isinstance(raw.get("meaning"), str) else ""}


def understand(question: str, history: list[str] = ()) -> dict:
    """LLM #1 (temperature 0): intent, item, problem, emotion, sentiment, furniture, constraints, meaning."""
    earlier = " | ".join(h[:300] for h in history[-3:])
    msg = (f"Earlier messages: {earlier}\n" if earlier else "") + f"Customer: {question}"
    # Instructions in the system message: in one user message the model echoes "/no_think" instead of answering.
    return parse_understanding(llm([{"role": "system", "content": UNDERSTAND}, {"role": "user", "content": msg}],
                                   temperature=0, json_mode=True))


def is_missing(kind: str) -> bool:
    """True when the store has nothing of this kind: its main word is in no product name or category."""
    head = re.findall(r"[a-z]+", re.split(r"\b(?:with|for|in|that|which)\b", kind)[0])  # "bunk bed with a slide" -> bunk, bed
    if not head or NOT_PRODUCTS & set(re.findall(r"[a-z]+", kind)):
        return False
    return not in_catalogue(head[-1])


SYMPATHY = "Start with one short, warm sentence of sympathy in your own words about exactly what they said."
CONGRATS = "Start by congratulating them warmly in your own words on exactly what they said."
ACKNOWLEDGE = "Start with one short sentence showing you understood their situation, in your own words."


def answer(question: str, extra_context: str = "", history: list[str] = ()) -> dict:
    """Returns {text, sources, missing, understanding}.

    extra_context carries computed facts such as a warranty check; history is the customer's earlier messages.
    """
    u = parse_understanding("") if extra_context else understand(question, history)
    intent = u["intent"]
    if intent == "CASUAL_CONVERSATION":  # chit-chat: no search, no store facts
        text = llm([{"role": "system", "content": CASUAL}, {"role": "user", "content": question}])
        return {"text": text, "sources": [], "missing": None, "understanding": u}
    if intent == "GENERAL_QUESTION":
        return {"text": IDK, "sources": [], "missing": None, "understanding": u}

    hits = search(question)
    # Policy question: give the model only policies, otherwise it lists loosely matching products
    # ("return window" -> a window table). A policy section beating every product also counts.
    best_rule = max((h["score"] for h in hits if h["kind"] == "rule"), default=0)
    rule_wins = best_rule >= max(h["score"] for h in hits if h["kind"] == "product")
    policy = not extra_context and (intent in POLICY_INTENTS or rule_wins and intent != "PRODUCT_RECOMMENDATION")
    need = intent == "PRODUCT_RECOMMENDATION" and bool(u["furniture"]) and not policy and not extra_context

    if need:
        # Search for the furniture that helps, not the words of the problem ("my leg is broken" -> table legs).
        detail = " ".join(u["furniture"])
        query = " ".join(w for w in re.findall(r"[a-z'-]+", detail) if w not in NOT_FURNITURE) or detail
        hits = [h for h in search(query, k=20) if not PART.match(h.get("name", "").partition(" - ")[2])][:6]
    elif hits[0]["score"] < MIN_SCORE and not extra_context:
        return {"text": IDK, "sources": [], "missing": None, "understanding": u}
    if policy:
        hits = [h for h in hits if h["kind"] == "rule"]

    item = u["item"]
    missing = item if intent.startswith("PRODUCT") and not policy and item and is_missing(item) else None
    if need:
        opening = {"positive": CONGRATS, "negative": SYMPATHY}.get(u["sentiment"], ACKNOWLEDGE)
    else:
        opening = SYMPATHY if u["sentiment"] == "negative" else ""
    task = ""
    if missing:
        task = UNAVAILABLE_TASK.format(item=missing) + "\n"
    elif need:
        task = NEED_TASK.format(opening=opening) + "\n"
    elif opening:
        task = f"Task: {opening} Then answer their question.\n"

    # LLM #2 gets what LLM #1 understood, so it answers the reason behind the request, not only the words.
    about = " ".join(filter(None, [
        u["meaning"],
        u["problem"] and f"Situation: {u['problem']}.",
        u["emotion"] not in ("neutral", "casual") and f"Feeling: {u['emotion']}.",
        u["constraints"] and f"Limits: {', '.join(u['constraints'])}.",
    ]))
    about = f"What the customer needs: {about}\n" if about else ""

    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    if extra_context:
        context += "\n" + extra_context
    text = llm([
        {"role": "system", "content": SYSTEM + (POLICY_FORMAT if policy or extra_context else PRODUCT_FORMAT)},
        {"role": "user", "content": f"Context:\n{context}\n\n{about}{task}Customer: {question}"},
    ])
    if missing:
        # Small models often repeat the "we don't sell it" line; drop that leading sentence.
        text = re.sub(r"(the store|we) (does|do) not (sell|have|carry)[^.]*\.\s*", "", text, flags=re.I).strip()
        text = NOT_AVAILABLE.format(item=missing) + "\n\n" + text
        log_request(missing, question)
    return {"text": text, "sources": hits, "missing": missing, "understanding": u}


if __name__ == "__main__":
    u = parse_understanding('```json\n{"intent": "product_search", "item": "Desk Lamp.", "furniture": "desk"}\n```')
    assert (u["intent"], u["item"], u["furniture"], u["sentiment"]) == ("PRODUCT_SEARCH", "desk lamp", ["desk"], "neutral")
    assert parse_understanding("not json")["intent"] == "" and parse_understanding('{"intent": "HACK"}')["intent"] == ""
    for q in ["my leg is broken, suggest me something", "I have back pain from sitting all day",
              "Do you sell desk lamps?", "Do you have a coffee table?", "What is the capital of France?"]:
        r = answer(q)
        print(f"\n> {q}\n{r['text']}\nmissing={r['missing']}")
