"""Answer one customer message. answer() is the only function the chat page calls.

The steps, in order. Each is one function below or one module in gen/:

  1. Cart commands ("check out", "show my cart")         gen/cart.py        no LLM
  2. Read the message: intent, feeling, needs            gen/understand.py  LLM call 1
  3. A pick ("the second one") goes into the cart        gen/picking.py, pick_reply()
  4. Chit-chat and off-topic questions end here          casual_reply()
  5. Search products and policies                        retrieve(), rag/index.py
  6. Stock check, then the task for the model            gen/catalogue.py, task_line()
  7. Write the reply                                     write()            LLM call 2
  8. Add the "not available" line, links, next step      finish()
"""

import csv
import re
from datetime import datetime
from pathlib import Path

import config
from gen import cart as cart_
from gen.catalogue import NOT_FURNITURE, PART, distinct, is_missing, named_products, product_links
from gen.llm import llm
from gen.picking import BUY, buys_by_name, ordinals, pick_products
from gen.prompts import (
    ACKNOWLEDGE,
    CASUAL,
    CASUAL_NEXT_STEP,
    CONGRATS,
    FOLLOW_UP,
    IDK,
    NEED_TASK,
    NEXT_STEP,
    NOT_AVAILABLE,
    POLICY_FORMAT,
    PRODUCT_FORMAT,
    PURCHASE_TASK,
    SYMPATHY,
    SYSTEM,
    UNAVAILABLE_TASK,
)
from gen.understand import parse_understanding, read_message
from nlp.chunks import inr
from rag.index import search

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

# If the best search match scores below this, the question is not about the store.
# The bot then says "I don't know" without a second LLM call.
MIN_SCORE = 0.30

# How many chunks go to the answer call. It lists at most 3 products, and on a laptop
# CPU every extra chunk (about 100 tokens) adds about 2 seconds of reading.
MAX_CONTEXT = 4

# How many results a need search looks through before spare parts are dropped.
NEED_SEARCH_SIZE = 20

# Temperature for policy answers: stay close to the written facts.
POLICY_TEMPERATURE = 0.3

# A chit-chat reply longer than this is a poem or an essay the customer asked for.
CASUAL_MAX_CHARS = 250

# Intents that are answered from the policy files only.
POLICY_INTENTS = {"STORE_INFORMATION", "ORDER_SUPPORT", "COMPLAINT"}

# A complaint sees only these policies. With the expiry policy in view, the small
# model once invented a "lifetime guarantee".
COMPLAINT_POLICIES = ("warranty.md", "returns.md")

# Where requests for products the store does not sell are logged.
REQUESTS = Path(__file__).resolve().parent.parent / "data" / "requests" / "requests.csv"


# --------------------------------------------------------------------------------------
# The main function
# --------------------------------------------------------------------------------------


def answer(
    question: str,
    history: list[str] = (),
    shown: list[dict] = (),
    on_token=None,
    cart: list[dict] = (),
) -> dict:
    """Answer one message.

    question       what the customer typed
    history        the customer's earlier messages, oldest first
    shown          the products in the bot's last reply, so "the second one" can be resolved
    on_token       called with the text so far while the reply is written (for live display)
    cart           what the customer has picked so far

    Returns a dict:
      text           the reply
      sources        the products and policies the reply is based on
      missing        the product asked for that the store does not sell, or None
      understanding  what LLM call 1 understood
      cart           the cart to keep for the next message
      product        only when the customer picked exactly one product
    """
    cart = list(cart)
    u = parse_understanding("")  # empty understanding, until the message is read

    def reply(text, sources=(), missing=None, **more):
        """Build the result. Reads `u` and `cart` as they are at the time of the call."""
        return {
            "text": text,
            "sources": list(sources),
            "missing": missing,
            "understanding": u,
            "cart": cart,
            **more,
        }

    # ---- 1. Cart commands need no LLM -------------------------------------------------
    about_cart = cart_.command(question, cart)
    if about_cart:
        text, cart = about_cart
        return reply(text)

    # ---- 2. Read the message ----------------------------------------------------------
    u = read_message(question, history)
    intent = u["intent"]
    about = about_line(u)

    # ---- 3. A pick goes into the cart -------------------------------------------------
    # The customer keeps shopping until they say "check out".
    # Three ways to see a pick: the model says so, a buying phrase follows a list
    # ("the second one"), or the customer wants to buy a product they name.
    after_a_list = shown and BUY.search(ordinals(question))
    is_pick = intent == "PURCHASE" or after_a_list or buys_by_name(question)
    if is_pick:
        picked = pick_products(question, list(shown))

        if picked:
            cart, new = cart_.add(cart, picked)
            text = pick_reply(question, picked, about, on_token)
            text = f"{text}\n\n{cart_.added_line(cart, new)}"
            if len(picked) == 1:
                return reply(text, picked, product=picked[0])
            return reply(text, picked)

        if shown:
            # "I'll get that" after several suggestions: ask instead of guessing.
            options = "\n".join(
                f"{number}. **{product['name']}** – {inr(product['price'])}"
                for number, product in enumerate(shown, 1)
            )
            return reply(f"Great choice! Which one would you like?\n{options}", shown)

        # A kind of product, not a specific one ("I want to buy a sofa"): search as usual.
        intent = "PRODUCT_SEARCH"

    # ---- 4. Chit-chat and off-topic questions -----------------------------------------
    if intent == "CASUAL_CONVERSATION":
        return reply(casual_reply(question))
    if intent == "GENERAL_QUESTION":
        return reply(IDK)

    # ---- 5. Search --------------------------------------------------------------------
    named = named_products(question)  # products mentioned by name get direct links later
    hits, policy, need = retrieve(question, u, intent)

    # Nothing in the data is close to the question. (A need or a named product never
    # ends here: those have their own search.)
    can_be_off_topic = not need and not named
    if can_be_off_topic and hits[0]["score"] < MIN_SCORE:
        return reply(IDK)

    if policy:
        hits = [hit for hit in hits if hit["kind"] == "rule"]
        if intent == "COMPLAINT":
            relevant = [hit for hit in hits if hit["source"].startswith(COMPLAINT_POLICIES)]
            hits = relevant or hits
    else:
        hits = distinct(hits)[:MAX_CONTEXT]

    # ---- 6. Stock check, then the task for the model ----------------------------------
    item = u["item"]
    asks_for_product = intent.startswith("PRODUCT") and not policy and item
    missing = item if asks_for_product and is_missing(item) else None
    task = task_line(u, need, missing, history, question)

    # ---- 7. Write the reply -----------------------------------------------------------
    # Policy answers do not get the `about` line: it made the small model say "I don't know".
    text = write(question, hits, "" if policy else about, task, policy, on_token)

    # ---- 8. Clean up ------------------------------------------------------------------
    if missing:
        log_request(missing, question)
    text = finish(text, missing, named, lists_products=not policy)
    return reply(text, hits, missing)


# --------------------------------------------------------------------------------------
# The steps
# --------------------------------------------------------------------------------------


def about_line(u: dict) -> str:
    """What call 1 understood, as one line for call 2.

    With it, the answer addresses the reason behind the request, not only its words.
    """
    parts = [u["meaning"]]
    if u["problem"]:
        parts.append(f"Situation: {u['problem']}.")
    if u["emotion"] not in ("neutral", "casual"):
        parts.append(f"Feeling: {u['emotion']}.")
    if u["constraints"]:
        parts.append(f"Limits: {', '.join(u['constraints'])}.")

    about = " ".join(part for part in parts if part)
    return f"What the customer needs: {about}\n" if about else ""


def pick_reply(question: str, picked: list[dict], about: str, on_token=None) -> str:
    """The reply to a pick.

    One product: a few warm words from the LLM, then its exact facts from the data.
    Several products: a plain list, no LLM.
    """
    if len(picked) > 1:
        rows = [f"- **{product['name']}** – {inr(product['price'])}" for product in picked]
        return "Good choices:\n" + "\n".join(rows)

    product = picked[0]

    # The warm words. The model sees only this one product.
    context = f"[{product['source']}] {product['text']}"
    task = PURCHASE_TASK.format(name=product["name"])
    intro = llm(
        [
            {"role": "system", "content": SYSTEM + POLICY_FORMAT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n{about}{task}Customer: {question}",
            },
        ],
        on_token=on_token,
    )

    # The facts, straight from the data so they are always exact.
    months = product["warranty_months"]
    years = f" ({months // 12} years)" if months >= 12 and months % 12 == 0 else ""
    facts = [
        f"- Price: {inr(product['price'])}",
        f"- Warranty: {months} months{years}",
        f"- Category: {product['category']}",
    ]
    if product.get("good_for"):
        facts.append(f"- Good for: {product['good_for'].replace('-', ' ')}")
    if product.get("goes_with"):
        facts.append(f"- Goes well with: {product['goes_with']}")
    if product.get("link"):
        short_name = product["name"].partition(" - ")[0]
        facts.append(f"- Product page: [{short_name}]({product['link']})")

    return f"{intro}\n\n**{product['name']}**\n" + "\n".join(facts)


def casual_reply(question: str) -> str:
    """A short friendly reply to chit-chat, with no search and no store facts.

    Not streamed: the length check below needs the whole reply, and chit-chat is short.
    """
    text = llm([{"role": "system", "content": CASUAL}, {"role": "user", "content": question}])

    # Chit-chat is 1 or 2 sentences. More means it wrote the poem or essay it was asked for.
    if len(text) > CASUAL_MAX_CHARS or text.count("\n") >= 2:
        return "I'd love to, but I can only help with our furniture store. " + CASUAL_NEXT_STEP

    # The small model forgets to steer back to shopping.
    # Skip the extra line when the reply already ends by asking how it can help.
    lowered = text.lower()
    steers_back = "furniture" in lowered or "home" in lowered or text.rstrip().endswith("?")
    if not steers_back:
        text += "\n\n" + CASUAL_NEXT_STEP
    return text


def retrieve(question: str, u: dict, intent: str) -> tuple[list[dict], bool, bool]:
    """Search, and decide what kind of answer this is.

    Returns (hits, policy, need):
      hits    the chunks to answer from, best first
      policy  True when the answer must come from the policy files only
      need    True when the customer described a situation and gets recommendations
    """
    hits = search(question)

    # Does a policy section match better than every product?
    best_rule = max((hit["score"] for hit in hits if hit["kind"] == "rule"), default=0)
    best_product = max((hit["score"] for hit in hits if hit["kind"] == "product"), default=0)
    rule_wins = best_rule >= best_product

    need = intent == "PRODUCT_RECOMMENDATION"

    # A policy question gets only policies. Otherwise the model lists loosely matching
    # products ("return window" finds a window table).
    # A "need" with no furniture named, beaten by a policy, is a misread policy question
    # ("what does the warranty not cover?").
    real_need = need and u["furniture"]
    policy = intent in POLICY_INTENTS or (rule_wins and not real_need)
    need = need and not policy

    if need:
        # Search for the furniture that helps, not the words of the problem:
        # "my leg is broken" would find table legs.
        # With no furniture named, search what they need ("a quiet room for daytime sleep").
        wanted = " ".join(u["furniture"]) or u["meaning"].lower() or question.lower()
        words = [word for word in re.findall(r"[a-z'-]+", wanted) if word not in NOT_FURNITURE]
        query = " ".join(words) or wanted

        found = search(query, k=NEED_SEARCH_SIZE)
        without_parts = [hit for hit in found if not is_spare_part(hit)]
        hits = distinct(without_parts)[:MAX_CONTEXT]

    return hits, policy, need


def is_spare_part(hit: dict) -> bool:
    """ "STOCKSUND - Legs for armchair" is a spare part, no help to someone with a need."""
    product_type = hit.get("name", "").partition(" - ")[2]
    return bool(PART.match(product_type))


def said_now(problem: str, question: str) -> bool:
    """True when the current message itself describes the problem.

    False when the problem only comes from an earlier message. Words are compared
    by their first 5 letters, so "stress" matches "stressed".
    """
    problem_stems = {word[:5] for word in re.findall(r"[a-z]+", problem.lower()) if len(word) > 3}
    if not problem_stems:
        return True
    question_stems = {word[:5] for word in re.findall(r"[a-z]+", question.lower())}
    return bool(problem_stems & question_stems)


def task_line(u: dict, need: bool, missing: str | None, history: list[str], question: str) -> str:
    """The instruction for call 2: how to open, and what to do when the product is not sold."""
    # Sympathy or congratulations is said once. When the situation only comes from
    # earlier messages, it was already said.
    follow_up = bool(history) and bool(u["problem"]) and not said_now(u["problem"], question)

    if follow_up:
        opening = FOLLOW_UP if need else ""
    elif need:
        by_sentiment = {"positive": CONGRATS, "negative": SYMPATHY}
        opening = by_sentiment.get(u["sentiment"], ACKNOWLEDGE)
    else:
        opening = SYMPATHY if u["sentiment"] == "negative" else ""

    if missing:
        return UNAVAILABLE_TASK.format(item=missing) + "\n"
    if need:
        return NEED_TASK.format(opening=opening) + "\n"
    if opening:
        return f"Task: {opening} Then answer their question.\n"
    return ""


def write(
    question: str,
    hits: list[dict],
    about: str,
    task: str,
    policy: bool,
    on_token=None,
) -> str:
    """LLM call 2: write the reply from the retrieved chunks.

    A policy answer is plain sentences at a low temperature; a product answer is a
    numbered list at the normal temperature.
    """
    context = "\n".join(f"[{hit['source']}] {hit['text']}" for hit in hits)

    reply_format = POLICY_FORMAT if policy else PRODUCT_FORMAT
    temperature = POLICY_TEMPERATURE if policy else config.LLM_TEMPERATURE

    messages = [
        {"role": "system", "content": SYSTEM + reply_format},
        {"role": "user", "content": f"Context:\n{context}\n\n{about}{task}Customer: {question}"},
    ]
    return llm(messages, temperature=temperature, on_token=on_token)


def finish(text: str, missing: str | None, named: list[dict], lists_products: bool) -> str:
    """What Python adds after the model has written."""
    # The small model sometimes copies this line from the format example.
    text = re.sub(r"What it is and why it suits them[.:]\s*", "", text)

    if missing:
        # Small models often repeat "we don't sell it". Drop that sentence, then
        # put our own honest line first.
        text = re.sub(
            r"(the store|we) (does|do) not (sell|have|carry)[^.]*\.\s*", "", text, flags=re.I
        ).strip()
        text = NOT_AVAILABLE.format(item=missing) + "\n\n" + text

    links = product_links(named)

    has_numbered_list = bool(re.search(r"^1\. \*\*", text, re.M))
    if lists_products and has_numbered_list:
        # The model's own closing offer ("Let me know if ...") would repeat ours.
        text = re.sub(
            r"\n+(let me know|feel free|would you like|if you)[^\n]*$", "", text, flags=re.I
        ).rstrip()
        return text + links + "\n\n" + NEXT_STEP

    return text + links


def log_request(item: str, question: str):
    """Keep a record of what customers ask for and the store does not sell."""
    REQUESTS.parent.mkdir(exist_ok=True)
    is_new_file = not REQUESTS.exists()

    with REQUESTS.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(["time", "item", "question"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), item, question])
