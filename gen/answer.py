"""Answer one customer message: understand it, pick a path, search, call the LLM, clean up the reply.

Order of work in answer(): understand() -> purchase / chit-chat / off-topic shortcuts -> search ->
stock check -> answer call -> "not available" line, product links and next step.
The prompt text is in gen/prompts.py.
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
from gen.prompts import (ACKNOWLEDGE, CASUAL, CASUAL_NEXT_STEP, CONGRATS, FOLLOW_UP, IDK, NEED_TASK,
                         NEXT_STEP, NOT_AVAILABLE, POLICY_FORMAT, PRODUCT_FORMAT, PURCHASE_TASK, SYMPATHY,
                         SYSTEM, UNAVAILABLE_TASK, UNDERSTAND)
from nlp.chunks import inr, product_chunks
from rag.index import search

# Below this best-match score the question is not about the store: answer without the LLM.
# Picked by testing off-topic questions (see `python -m gen.answer`).
MIN_SCORE = 0.30

# Chunks sent to the answer call. It lists at most 3 products; on a laptop CPU every extra chunk
# (~100 tokens) adds about 2 seconds of prompt reading.
MAX_CONTEXT = 4


REQUESTS = Path(__file__).resolve().parent.parent / "data" / "requests" / "requests.csv"


INTENTS = {"PRODUCT_SEARCH", "PRODUCT_RECOMMENDATION", "PRODUCT_COMPARISON", "STORE_INFORMATION",
           "PURCHASE", "ORDER_SUPPORT", "COMPLAINT", "CASUAL_CONVERSATION", "GENERAL_QUESTION"}
POLICY_INTENTS = {"STORE_INFORMATION", "ORDER_SUPPORT", "COMPLAINT"}


_client = None


def llm(messages: list[dict], temperature: float = config.LLM_TEMPERATURE, json_mode: bool = False,
        on_token=None) -> str:
    """One chat call. With on_token, the reply is streamed and on_token gets the text so far after each piece."""
    global _client
    _client = _client or OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    extra = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    if config.LLM_KEEP_ALIVE:
        extra["extra_body"] = {"keep_alive": config.LLM_KEEP_ALIVE}
    # reasoning_effort="none" stops Qwen3 thinking (about 30x faster); max_tokens stops runaway replies.
    out = _client.chat.completions.create(model=config.LLM_MODEL, messages=messages, temperature=temperature,
                                          max_tokens=400, reasoning_effort=config.LLM_REASONING_EFFORT,
                                          stream=on_token is not None, **extra)
    if on_token is None:
        text = out.choices[0].message.content or ""
    else:
        text = ""
        for chunk in out:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
                on_token(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
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
                 r"(that|this|the (first|second|third)) one|add (it|this|that)|(number|option|#)\s*[1-3]|"
                 r"(take|get|buy|want|choose|pick|like)\b[^.?!]{0,20}\b(first|second|third|1st|2nd|3rd))\b"
                 r"|^\s*[1-3]\s*\.?\s*$"  # or just the number from the list
                 r"|^\s*(the\s+)?(first|second|third|1st|2nd|3rd)(\s+one)?\s*[.!]?\s*$"  # or "1st", "the second one"
                 r"|\b(the\s+)?(cheapest|least expensive|most expensive|priciest)(\s+one)?\b|\b(the last|last one)\b", re.I)


def ordinals(text: str) -> str:
    """Join "1 st" into "1st", as customers type it both ways."""
    return re.sub(r"\b([1-3])\s+(st|nd|rd)\b", r"\1\2", text, flags=re.I)
ORDINAL = re.compile(r"\b(first|1st|second|2nd|third|3rd)\b|(?:number|no\.?|#|option)\s*([1-3])\b|^\s*([1-3])\s*\.?\s*$")


def plain(s: str) -> str:
    """Lowercase without accents, so "hattefjall" matches "HATTEFJÄLL"."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def pick_product(question: str, shown: list[dict]) -> dict | None:
    """The one product the customer means: named (suggested earlier first), "the second one", or the only one shown."""
    q = plain(ordinals(question))
    for h in [*shown, *(h for h in search(question, k=10) if h["kind"] == "product")]:
        brand = plain(h["name"].partition(" - ")[0])
        if brand and re.search(rf"\b{re.escape(brand)}\b", q):
            return h
    if shown and re.search(r"\b(cheapest|least expensive|lowest price)\b", q):
        return min(shown, key=lambda h: h["price"])
    if shown and re.search(r"\b(most expensive|priciest|best quality)\b", q):
        return max(shown, key=lambda h: h["price"])
    if shown and re.search(r"\b(the last|last one)\b", q):
        return shown[-1]
    m = ORDINAL.search(q)
    if m and shown:
        i = ("first", "1st", "second", "2nd", "third", "3rd").index(m[1]) // 2 if m[1] else int(m[2] or m[3]) - 1
        return shown[i] if i < len(shown) else None
    return shown[0] if len(shown) == 1 else None


def purchase_reply(question: str, product: dict, about: str, on_token=None) -> str:
    """Warm words from the LLM, then the exact facts from the data, then the next step."""
    intro = llm([
        {"role": "system", "content": SYSTEM + POLICY_FORMAT},
        {"role": "user", "content": f"Context:\n[{product['source']}] {product['text']}\n\n{about}"
                                    f"{PURCHASE_TASK.format(name=product['name'])}Customer: {question}"},
    ], on_token=on_token)
    months = product["warranty_months"]
    years = f" ({months // 12} years)" if months >= 12 and months % 12 == 0 else ""
    facts = [f"- Price: {inr(product['price'])}", f"- Warranty: {months} months{years}",
             f"- Category: {product['category']}"]
    if product.get("good_for"):
        facts.append(f"- Good for: {product['good_for'].replace('-', ' ')}")
    if product.get("goes_with"):
        facts.append(f"- Goes well with: {product['goes_with']}")
    if product.get("link"):
        facts.append(f"- Product page: [{product['name'].partition(' - ')[0]}]({product['link']})")
    return (f"{intro}\n\n**{product['name']}**\n" + "\n".join(facts)
            + "\n\nHow would you like to proceed? Pick an option below.")


def said_now(problem: str, question: str) -> bool:
    """True when the current message itself describes the problem, not only an earlier one."""
    stems = {w[:5] for w in re.findall(r"[a-z]+", problem.lower()) if len(w) > 3}  # "stress" matches "stressed"
    return not stems or bool(stems & {w[:5] for w in re.findall(r"[a-z]+", question.lower())})


# Product names that are also everyday words or first names: they count only when typed in capitals ("LACK table"),
# so "I lack space" or "hallo" do not pull in products.
COMMON_WORD_NAMES = {"lack", "hallo", "urban", "utter", "harry", "erik", "glenn", "len", "hol", "pax", "stig",
                     "jules", "micke", "nisse", "cilla", "terje", "bror", "rast", "olov", "olaus"}


def product_names(h: dict) -> list[str]:
    """Plain product names of a chunk: "STENSELE / RÖNNINGE - Table" -> ["stensele", "ronninge"]."""
    return [plain(n.strip()) for n in h["name"].partition(" - ")[0].split("/") if n.strip()]


@lru_cache(maxsize=1)
def catalogue_by_name() -> dict[str, list[dict]]:
    out = {}
    for c in product_chunks():
        for n in product_names(c):
            out.setdefault(n, []).append(c)
    return out


def named_products(question: str) -> list[dict]:
    """Products the customer names ("MALM bed"): best search matches first, at most 3 per name and 6 in all."""
    q = plain(question)
    caps = unicodedata.normalize("NFKD", question).encode("ascii", "ignore").decode()
    names = [n for n in catalogue_by_name() if re.search(rf"\b{re.escape(n)}\b", q)
             and (n not in COMMON_WORD_NAMES or re.search(rf"\b{re.escape(n.upper())}\b", caps))]
    if not names:
        return []
    ranked = [h for h in search(question, k=20) if h["kind"] == "product"]
    out, seen = [], set()
    for n in names:
        picked = 0
        for h in [h for h in ranked if n in product_names(h)] + catalogue_by_name()[n]:
            if picked < 3 and h["name"] not in seen:  # colour variants share a name: one link each
                seen.add(h["name"])
                out.append(h)
                picked += 1
    return out[:6]


def product_links(products: list[dict]) -> str:
    rows = [f"- [{h['name']}]({h['link']}) – {inr(h['price'])}" for h in products if h.get("link")]
    return "\n\n**Product pages:**\n" + "\n".join(rows) if rows else ""


def distinct(hits) -> list[dict]:
    """Drop repeats: the catalogue lists some products twice (same name and price, different item id)."""
    seen, out = set(), []
    for h in hits:
        key = (h.get("name") or h["source"], h.get("price"))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def answer(question: str, extra_context: str = "", history: list[str] = (), shown: list[dict] = (),
           on_token=None) -> dict:
    """Returns {text, sources, missing, understanding}, plus `product` when the customer picked one to buy.

    extra_context carries computed facts such as a warranty check; history is the customer's earlier messages;
    shown is the products in the bot's last reply, so "I'll take the second one" can be resolved.
    on_token(text_so_far) is called while the reply is written, so the page can show it live.
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

    if intent == "PURCHASE" or shown and BUY.search(ordinals(question)):
        product = pick_product(question, list(shown))
        if product:
            return {"text": purchase_reply(question, product, about, on_token), "sources": [product], "missing": None,
                    "understanding": u, "product": product}
        if shown:  # "I'll get that" after several suggestions: ask instead of guessing
            options = "\n".join(f"{i}. **{h['name']}** – {inr(h['price'])}" for i, h in enumerate(shown, 1))
            return {"text": f"Great choice! Which one would you like?\n{options}", "sources": list(shown),
                    "missing": None, "understanding": u}
        intent = "PRODUCT_SEARCH"  # a kind of product, not a specific one: search as usual

    if intent == "CASUAL_CONVERSATION":  # chit-chat: no search, no store facts
        # Not streamed: the poem check below runs on the whole reply, and chit-chat is short anyway.
        text = llm([{"role": "system", "content": CASUAL}, {"role": "user", "content": question}])
        if len(text) > 250 or text.count("\n") >= 2:  # chit-chat is 1 or 2 sentences; more is a poem or essay
            text = "I'd love to, but I can only help with our furniture store. " + CASUAL_NEXT_STEP
        # The small model forgets to steer back; skip when it already ends by asking how it can help.
        if "furniture" not in text.lower() and "home" not in text.lower() and not text.rstrip().endswith("?"):
            text += "\n\n" + CASUAL_NEXT_STEP
        return {"text": text, "sources": [], "missing": None, "understanding": u}
    if intent == "GENERAL_QUESTION":
        return {"text": IDK, "sources": [], "missing": None, "understanding": u}

    named = named_products(question)  # products the customer mentions by name get direct links
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
        hits = distinct(h for h in search(query, k=20) if not PART.match(h.get("name", "").partition(" - ")[2]))
        hits = hits[:MAX_CONTEXT]
    elif hits[0]["score"] < MIN_SCORE and not extra_context and not named:
        return {"text": IDK, "sources": [], "missing": None, "understanding": u}
    if policy:
        hits = [h for h in hits if h["kind"] == "rule"]
        if intent == "COMPLAINT":  # other policies (expiry) led the small model to invent a "lifetime guarantee"
            hits = [h for h in hits if h["source"].startswith(("warranty.md", "returns.md"))] or hits
    elif not extra_context:
        hits = distinct(hits)[:MAX_CONTEXT]

    item = u["item"]
    missing = item if intent.startswith("PRODUCT") and not policy and item and is_missing(item) else None
    # Sympathy or congratulations once: when the situation only comes from earlier messages, it was already said.
    follow_up = bool(history) and bool(u["problem"]) and not said_now(u["problem"], question)
    if follow_up:
        opening = FOLLOW_UP if need else ""
    elif need:
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
    ], temperature=0.3 if policy or extra_context else config.LLM_TEMPERATURE,  # facts: stay close to the policy text
        on_token=on_token)
    text = re.sub(r"What it is and why it suits them[.:]\s*", "", text)  # the small model sometimes copies the format example
    if missing:
        # Small models often repeat the "we don't sell it" line; drop that leading sentence.
        text = re.sub(r"(the store|we) (does|do) not (sell|have|carry)[^.]*\.\s*", "", text, flags=re.I).strip()
        text = NOT_AVAILABLE.format(item=missing) + "\n\n" + text
        log_request(missing, question)
    if not policy and not extra_context and re.search(r"^1\. \*\*", text, re.M):
        # The model's own closing offer ("Let me know if...") would repeat ours.
        text = re.sub(r"\n+(let me know|feel free|would you like|if you)[^\n]*$", "", text, flags=re.I).rstrip()
        text += product_links(named) + "\n\n" + NEXT_STEP
    else:
        text += product_links(named)
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
    assert BUY.search(ordinals("i want to take 1 st chair and this 1st chair")) and not BUY.search("I want 2 chairs")
    assert all(BUY.search(q) for q in ["1st", "the second one", "third."]) and not BUY.search("first time buying a sofa")
    priced = [{"name": "A - x", "price": 30}, {"name": "B - y", "price": 10}, {"name": "C - z", "price": 20}]
    assert pick_product("the cheapest one please", priced)["price"] == 10
    assert pick_product("most expensive", priced)["price"] == 30 and pick_product("the last one", priced) is priced[2]
    assert BUY.search("the cheapest one please") and BUY.search("actually show me the last one")
    assert not BUY.search("I bought a sofa last year")
    assert {n for h in named_products("How much is the MALM bed?") for n in product_names(h)} >= {"malm"}
    assert named_products("I lack space at home") == [] and named_products("hallo") == []
    assert any("lack" in product_names(h) for h in named_products("What does the LACK coffee table cost?"))
    assert pick_product(ordinals("i want to take 1 st chair"), shown) is shown[0]
    assert said_now("broken arm", "i have broken arm") and not said_now("broken leg", "I need something for my room")
    assert said_now("stress after work", "I'm feeling stressed") and said_now("", "anything")
    assert len(distinct([{"name": "A", "price": 1, "source": "x"}, {"name": "A", "price": 1, "source": "y"}])) == 1
    assert head_noun("Office chair with armrests") == "chair" and head_noun("Laptop table, 100x36 cm") == "table"
    assert is_broken_furniture("the leg of my new table snapped") and is_broken_furniture("the drawer of my wardrobe broke")
    assert not any(is_broken_furniture(q) for q in ["my leg is broken", "I broke my arm", "my house got flooded"])
    for q in ["my leg is broken, suggest me something", "I have back pain from sitting all day",
              "Do you sell desk lamps?", "Do you have a coffee table?", "What is the capital of France?"]:
        r = answer(q)
        print(f"\n> {q}\n{r['text']}\nmissing={r['missing']}")
