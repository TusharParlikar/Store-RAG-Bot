"""Step 4: build the prompt, call the LLM, return the answer with its sources.

Also handles products the store does not carry: the bot says so, logs the
request in data/requests/requests.csv, and suggests the closest product it has.
"""
import csv
import json
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

import config
from nlp.chunks import inr, product_chunks
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
7. If a "Warranty check (computed)" line is given, use its status and date as-is. Never calculate dates yourself.
8. You are also a warm, honest salesperson: show how each product makes their life better so they want to buy it.
   Never pressure them and never invent discounts, offers, stock limits or deadlines."""

# Said after every product list, so the customer always has an easy next step towards buying.
NEXT_STEP = "Would you like one of these? Tell me the number and I'll share the full details."
CASUAL_NEXT_STEP = "Can I help you find something for your home today?"

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
           "PURCHASE", "ORDER_SUPPORT", "COMPLAINT", "CASUAL_CONVERSATION", "GENERAL_QUESTION"}
POLICY_INTENTS = {"STORE_INFORMATION", "ORDER_SUPPORT", "COMPLAINT"}

# LLM #1: understand the customer before anything is searched. Few-shot: a 1.7b model follows examples
# far better than instructions. The model only describes the message; Python decides what to do with it.
UNDERSTAND = """Read the customer's message (and their earlier messages, if given) and describe it as one JSON object:
{"intent": one of PRODUCT_SEARCH, PRODUCT_RECOMMENDATION, PRODUCT_COMPARISON, PURCHASE, STORE_INFORMATION, ORDER_SUPPORT, COMPLAINT, CASUAL_CONVERSATION, GENERAL_QUESTION,
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
PURCHASE: they decide to buy, or ask for more about, one specific product: by its name (MALM, HEMNES) or one suggested earlier ("that one", "the second one", "I'll take it").
STORE_INFORMATION: returns, warranty, delivery, assembly, store policy. ORDER_SUPPORT: an existing order.
COMPLAINT: a product they bought from this store is faulty, damaged or late. CASUAL_CONVERSATION: greetings, thanks, chit-chat, talk about this chatbot.
Any feeling (lonely, sad, stressed, tired), life event or misfortune at home (an accident, a flood, a break-in) is PRODUCT_RECOMMENDATION, never CASUAL_CONVERSATION or COMPLAINT.
GENERAL_QUESTION: general knowledge unrelated to the store.
For "furniture" name only furniture (chairs, armchairs, recliners, sofas, beds, footstools, tables, desks, storage), never medical items.
Use earlier messages only when the current message is vague on its own ("something for my room", "that one").
If the current message names what they want, describe the current message and leave out the earlier problem.

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

Customer: I feel sad and alone since my kids moved out
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "feeling alone after the kids moved out", "emotion": "disappointed", "sentiment": "negative", "furniture": ["comfortable armchair", "reading lamp table", "bookcase"], "constraints": [], "meaning": "Needs a cosy, comforting space at home."}

Customer: my son just got his first job!
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "son got his first job", "emotion": "excited", "sentiment": "positive", "furniture": ["desk", "office chair", "bookcase"], "constraints": [], "meaning": "Wants to set up a good work space to celebrate the new job."}

Customer: a pipe burst and ruined our living room
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "living room ruined by a burst pipe", "emotion": "worried", "sentiment": "negative", "furniture": ["sofa", "coffee table", "tv bench"], "constraints": ["living room"], "meaning": "Needs to refurnish the living room."}

Customer: we are expecting a baby soon
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "a baby is coming", "emotion": "excited", "sentiment": "positive", "furniture": ["crib", "changing table", "nursery storage"], "constraints": [], "meaning": "Needs to furnish a nursery."}

Earlier messages: my leg is broken
Customer: I need something for my room
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "broken leg", "emotion": "pain", "sentiment": "negative", "furniture": ["armchair with armrests", "footstool", "bedside table"], "constraints": ["bedroom"], "meaning": "Needs bedroom furniture that is easy to use with a broken leg."}

Customer: I want to buy a sofa
{"intent": "PRODUCT_SEARCH", "item": "sofa", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Wants to buy a sofa."}

Earlier messages: my neck hurts when I work
Customer: ok I'll get that
{"intent": "PURCHASE", "item": "", "problem": "", "emotion": "excited", "sentiment": "positive", "furniture": [], "constraints": [], "meaning": "Decides to buy a product suggested earlier."}

Customer: I'd like to buy the HEMNES bed frame, tell me more about it
{"intent": "PURCHASE", "item": "bed frame", "problem": "", "emotion": "excited", "sentiment": "positive", "furniture": [], "constraints": [], "meaning": "Wants details on the HEMNES bed frame before buying it."}

Earlier messages: my back hurts after work | I'll take the MARKUS chair
Customer: show me desks that go with my new office chair
{"intent": "PRODUCT_SEARCH", "item": "desk", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": ["desk"], "constraints": [], "meaning": "Wants a desk to go with a new office chair."}

Customer: tell me more about the second one
{"intent": "PURCHASE", "item": "", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Wants details on the second product suggested earlier."}

Customer: can I return a chair after assembling it?
{"intent": "STORE_INFORMATION", "item": "", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Asks about the return policy for assembled items."}

Customer: where is my order? it was supposed to come yesterday
{"intent": "ORDER_SUPPORT", "item": "", "problem": "late order", "emotion": "worried", "sentiment": "negative", "furniture": [], "constraints": [], "meaning": "Wants to know where a late order is."}

Customer: the table I bought arrived scratched, this is so annoying
{"intent": "COMPLAINT", "item": "table", "problem": "table arrived scratched", "emotion": "frustrated", "sentiment": "negative", "furniture": [], "constraints": [], "meaning": "Wants a damaged table fixed or replaced."}

Customer: the door of my cabinet came off after a month
{"intent": "COMPLAINT", "item": "cabinet", "problem": "cabinet door came off after a month", "emotion": "frustrated", "sentiment": "negative", "furniture": [], "constraints": [], "meaning": "Wants a faulty cabinet repaired or replaced."}

Customer: hi! I want to build this chatbot
{"intent": "CASUAL_CONVERSATION", "item": "", "problem": "", "emotion": "casual", "sentiment": "positive", "furniture": [], "constraints": [], "meaning": "Greets and chats about building the chatbot."}

Customer: who is the prime minister?
{"intent": "GENERAL_QUESTION", "item": "", "problem": "", "emotion": "neutral", "sentiment": "neutral", "furniture": [], "constraints": [], "meaning": "Asks a general knowledge question."}"""

CASUAL = """You are the friendly assistant of a furniture store in India. The customer is just chatting.
Reply warmly in 1 or 2 short sentences, then offer to help with furniture, prices, warranty or returns.
Never state product names, prices or store policies here, and never answer general knowledge questions.
If they ask you to write or explain anything (a poem, a story, code, facts), kindly say you can only help with the furniture store."""

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


def head_noun(kind: str) -> str:
    """Main word of a product type: "Office chair with armrests" -> chair, "Laptop table, 100x36 cm" -> table."""
    words = re.findall(r"[a-z]+", re.split(r"\b(?:with|for|in|that|which)\b|,", kind.lower())[0])
    return words[-1] if words else ""


@lru_cache(maxsize=1)
def catalogue_words() -> frozenset[str]:
    """What kinds of product the store has: the main word of every product type, plus category words.

    Main words only, so a "Laptop table" does not make the store look like it sells laptops.
    """
    heads = {head_noun(c["name"].partition(" - ")[2]) for c in product_chunks()}
    return frozenset(heads | set(re.findall(r"[a-z]+", " ".join(c["category"] for c in product_chunks()).lower())))


# Common Indian/US words for product types the catalogue names differently.
# ponytail: hand list; grows if the request log shows false "not available" replies.
SYNONYMS = {"crib": "cot", "couch": "sofa", "closet": "wardrobe", "almirah": "wardrobe", "cupboard": "cabinet"}


def in_catalogue(word: str) -> bool:
    word = SYNONYMS.get(word, word)
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


PURCHASED = re.compile(r"\b(bought|ordered|order|delivered|delivery|arrived|purchased?|refund|replace(ment)?)\b", re.I)
BROKEN = re.compile(r"\b(broke|broken|snapped|cracked|came off|fell off|wobbly|damaged|scratched|faulty|stopped working)\b", re.I)


def is_broken_furniture(text: str) -> bool:
    """A piece of furniture broke ("the leg of my table snapped"), not a body part ("my leg is broken")."""
    words = set(re.findall(r"[a-z]+", text.lower())) - NOT_FURNITURE - NOT_PRODUCTS
    return bool(BROKEN.search(text)) and any(in_catalogue(w) for w in words)


def is_missing(kind: str) -> bool:
    """True when the store has nothing of this kind: its main word is in no product name or category."""
    head = re.findall(r"[a-z]+", re.split(r"\b(?:with|for|in|that|which)\b", kind)[0])  # "bunk bed with a slide" -> bunk, bed
    if not head or NOT_PRODUCTS & set(re.findall(r"[a-z]+", kind)):
        return False
    return not in_catalogue(head[-1])


# Buying words right after the bot listed products. The small model often reads "ok I'll get that" as the old need again.
BUY = re.compile(r"\b(i'?ll (take|get|buy|have)|i ?will (take|get|buy)|(buy|take|get|want) (it|this|that|this one|that one)|"
                 r"(that|this|the (first|second|third)) one|add (it|this|that)|(number|option|#)\s*[1-3])\b"
                 r"|^\s*[1-3]\s*\.?\s*$", re.I)  # or just the number from the list
ORDINAL = re.compile(r"\b(first|1st|second|2nd|third|3rd)\b|(?:number|no\.?|#|option)\s*([1-3])\b|^\s*([1-3])\s*\.?\s*$")

PURCHASE_TASK = ("Task: the customer wants to buy {name}. In 2 or 3 warm sentences say it is a good choice, "
                 "what it is, and how it suits their situation. Do not mention any other product.\n")


def plain(s: str) -> str:
    """Lowercase without accents, so "hattefjall" matches "HATTEFJÄLL"."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def pick_product(question: str, shown: list[dict]) -> dict | None:
    """The one product the customer means: named (suggested earlier first), "the second one", or the only one shown."""
    q = plain(question)
    for h in [*shown, *(h for h in search(question, k=10) if h["kind"] == "product")]:
        brand = plain(h["name"].partition(" - ")[0])
        if brand and re.search(rf"\b{re.escape(brand)}\b", q):
            return h
    m = ORDINAL.search(q)
    if m and shown:
        i = ("first", "1st", "second", "2nd", "third", "3rd").index(m[1]) // 2 if m[1] else int(m[2] or m[3]) - 1
        return shown[i] if i < len(shown) else None
    return shown[0] if len(shown) == 1 else None


def purchase_reply(question: str, product: dict, about: str) -> str:
    """Warm words from the LLM, then the exact facts from the data, then the next step."""
    intro = llm([
        {"role": "system", "content": SYSTEM + POLICY_FORMAT},
        {"role": "user", "content": f"Context:\n[{product['source']}] {product['text']}\n\n{about}"
                                    f"{PURCHASE_TASK.format(name=product['name'])}Customer: {question}"},
    ])
    months = product["warranty_months"]
    years = f" ({months // 12} years)" if months >= 12 and months % 12 == 0 else ""
    facts = [f"- Price: {inr(product['price'])}", f"- Warranty: {months} months{years}",
             f"- Category: {product['category']}"]
    if product.get("good_for"):
        facts.append(f"- Good for: {product['good_for'].replace('-', ' ')}")
    if product.get("goes_with"):
        facts.append(f"- Goes well with: {product['goes_with']}")
    return (f"{intro}\n\n**{product['name']}**\n" + "\n".join(facts)
            + "\n\nHow would you like to proceed? Pick an option below.")


SYMPATHY = "Start with one short, warm sentence of sympathy in your own words about exactly what they said."
CONGRATS = "Start by congratulating them warmly in your own words on exactly what they said."
ACKNOWLEDGE = "Start with one short sentence showing you understood their situation, in your own words."


def answer(question: str, extra_context: str = "", history: list[str] = (), shown: list[dict] = ()) -> dict:
    """Returns {text, sources, missing, understanding}, plus `product` when the customer picked one to buy.

    extra_context carries computed facts such as a warranty check; history is the customer's earlier messages;
    shown is the products in the bot's last reply, so "I'll take the second one" can be resolved.
    """
    u = parse_understanding("") if extra_context else understand(question, history)
    if u["intent"] in ("PRODUCT_RECOMMENDATION", "") and is_broken_furniture(question):
        u["intent"] = "COMPLAINT"  # the small model reads "the leg of my table snapped" as an injury
    elif u["intent"] == "COMPLAINT" and not (is_broken_furniture(question) or PURCHASED.search(question)):
        u["intent"] = "PRODUCT_RECOMMENDATION"  # a flood or accident at home is not about something we sold
    intent = u["intent"]

    # LLM #2 gets what LLM #1 understood, so it answers the reason behind the request, not only the words.
    about = " ".join(filter(None, [
        u["meaning"],
        u["problem"] and f"Situation: {u['problem']}.",
        u["emotion"] not in ("neutral", "casual") and f"Feeling: {u['emotion']}.",
        u["constraints"] and f"Limits: {', '.join(u['constraints'])}.",
    ]))
    about = f"What the customer needs: {about}\n" if about else ""

    if intent == "PURCHASE" or shown and BUY.search(question):
        product = pick_product(question, list(shown))
        if product:
            return {"text": purchase_reply(question, product, about), "sources": [product], "missing": None,
                    "understanding": u, "product": product}
        if shown:  # "I'll get that" after several suggestions: ask instead of guessing
            options = "\n".join(f"{i}. **{h['name']}** – {inr(h['price'])}" for i, h in enumerate(shown, 1))
            return {"text": f"Great choice! Which one would you like?\n{options}", "sources": list(shown),
                    "missing": None, "understanding": u}
        intent = "PRODUCT_SEARCH"  # a kind of product, not a specific one: search as usual

    if intent == "CASUAL_CONVERSATION":  # chit-chat: no search, no store facts
        text = llm([{"role": "system", "content": CASUAL}, {"role": "user", "content": question}])
        if len(text) > 250 or text.count("\n") >= 2:  # chit-chat is 1 or 2 sentences; more is a poem or essay
            text = "I'd love to, but I can only help with our furniture store. " + CASUAL_NEXT_STEP
        if "furniture" not in text.lower() and "home" not in text.lower():  # the small model forgets to steer back
            text += "\n\n" + CASUAL_NEXT_STEP
        return {"text": text, "sources": [], "missing": None, "understanding": u}
    if intent == "GENERAL_QUESTION":
        return {"text": IDK, "sources": [], "missing": None, "understanding": u}

    hits = search(question)
    # Policy question: give the model only policies, otherwise it lists loosely matching products
    # ("return window" -> a window table). A policy section beating every product also counts.
    best_rule = max((h["score"] for h in hits if h["kind"] == "rule"), default=0)
    rule_wins = best_rule >= max((h["score"] for h in hits if h["kind"] == "product"), default=0)
    need = intent == "PRODUCT_RECOMMENDATION" and not extra_context
    # A "need" with no furniture named, beaten by a policy, is a misread policy question ("what does the warranty not cover?").
    policy = not extra_context and (intent in POLICY_INTENTS or rule_wins and not (need and u["furniture"]))
    need = need and not policy

    if need:
        # Search for the furniture that helps, not the words of the problem ("my leg is broken" -> table legs).
        # No furniture named: search what they need ("a sleep-friendly room for daytime rest").
        detail = " ".join(u["furniture"]) or u["meaning"].lower() or question.lower()
        query = " ".join(w for w in re.findall(r"[a-z'-]+", detail) if w not in NOT_FURNITURE) or detail
        hits = [h for h in search(query, k=20) if not PART.match(h.get("name", "").partition(" - ")[2])][:6]
    elif hits[0]["score"] < MIN_SCORE and not extra_context:
        return {"text": IDK, "sources": [], "missing": None, "understanding": u}
    if policy:
        hits = [h for h in hits if h["kind"] == "rule"]
        if intent == "COMPLAINT":  # other policies (expiry) led the small model to invent a "lifetime guarantee"
            hits = [h for h in hits if h["source"].startswith(("warranty.md", "returns.md"))] or hits

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

    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    if extra_context:
        context += "\n" + extra_context
    text = llm([
        {"role": "system", "content": SYSTEM + (POLICY_FORMAT if policy or extra_context else PRODUCT_FORMAT)},
        # Policy answers need only the facts: the understanding line made the small model say "I don't know".
        {"role": "user", "content": f"Context:\n{context}\n\n{'' if policy else about}{task}Customer: {question}"},
    ], temperature=0.3 if policy or extra_context else config.LLM_TEMPERATURE)  # facts: stay close to the policy text
    text = re.sub(r"What it is and why it suits them[.:]\s*", "", text)  # the small model sometimes copies the format example
    if missing:
        # Small models often repeat the "we don't sell it" line; drop that leading sentence.
        text = re.sub(r"(the store|we) (does|do) not (sell|have|carry)[^.]*\.\s*", "", text, flags=re.I).strip()
        text = NOT_AVAILABLE.format(item=missing) + "\n\n" + text
        log_request(missing, question)
    if not policy and not extra_context and re.search(r"^1\. \*\*", text, re.M):
        text += "\n\n" + NEXT_STEP
    return {"text": text, "sources": hits, "missing": missing, "understanding": u}


if __name__ == "__main__":
    u = parse_understanding('```json\n{"intent": "product_search", "item": "Desk Lamp.", "furniture": "desk"}\n```')
    assert (u["intent"], u["item"], u["furniture"], u["sentiment"]) == ("PRODUCT_SEARCH", "desk lamp", ["desk"], "neutral")
    assert parse_understanding("not json")["intent"] == "" and parse_understanding('{"intent": "HACK"}')["intent"] == ""
    shown = [{"name": "HATTEFJÄLL - Office chair"}, {"name": "NILSOVE - Chair"}, {"name": "JÄRVFJÄLLET - Office chair"}]
    assert pick_product("I'll buy the hattefjall", shown) is shown[0]
    assert pick_product("tell me more about the second one", shown) is shown[1]
    assert pick_product("option 3 please", shown) is shown[2]
    assert pick_product("ok I'll get that", shown) is None and pick_product("ok I'll get that", shown[:1]) is shown[0]
    assert all(BUY.search(q) for q in ["ok ill get that", "I'll take the second one", "i want this one", "2", " 3. ", "option 1"])
    assert not any(BUY.search(q) for q in ["do you have a desk?", "my neck hurts", "show me other chairs",
                                           "tell me more about returns", "I have 2 kids"])
    assert pick_product("2", shown) is shown[1]
    assert head_noun("Office chair with armrests") == "chair" and head_noun("Laptop table, 100x36 cm") == "table"
    assert is_broken_furniture("the leg of my new table snapped") and is_broken_furniture("the drawer of my wardrobe broke")
    assert not any(is_broken_furniture(q) for q in ["my leg is broken", "I broke my arm", "my house got flooded"])
    for q in ["my leg is broken, suggest me something", "I have back pain from sitting all day",
              "Do you sell desk lamps?", "Do you have a coffee table?", "What is the capital of France?"]:
        r = answer(q)
        print(f"\n> {q}\n{r['text']}\nmissing={r['missing']}")
