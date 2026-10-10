"""Step 1 of every answer: read the customer's message.

One LLM call describes the message as JSON (intent, feeling, what would help).
Python then cleans that JSON and corrects two labels the small model often gets wrong.

The rest of the code calls read_message().
"""

import json
import re

from gen.catalogue import NOT_FURNITURE, NOT_PRODUCTS, in_catalogue
from gen.llm import llm
from gen.prompts import UNDERSTAND

# Every intent the model may answer with. Anything else is treated as "unknown".
INTENTS = {
    "PRODUCT_SEARCH",
    "PRODUCT_RECOMMENDATION",
    "PRODUCT_COMPARISON",
    "STORE_INFORMATION",
    "PURCHASE",
    "ORDER_SUPPORT",
    "COMPLAINT",
    "CASUAL_CONVERSATION",
    "GENERAL_QUESTION",
}

# How many earlier messages the model sees, and how much of each.
MEMORY_MESSAGES = 3
MEMORY_CHARS = 300

# Words that show the customer is talking about something they bought from the store.
PURCHASED = re.compile(
    r"\b(bought|ordered|order|delivered|delivery|arrived|purchased?|refund|replace(ment)?)\b",
    re.I,
)

# Words that show something is broken.
BROKEN = re.compile(
    r"\b(broke|broken|snapped|cracked|came off|fell off|wobbly|damaged|scratched|faulty"
    r"|stopped working)\b",
    re.I,
)


# --------------------------------------------------------------------------------------
# The LLM call
# --------------------------------------------------------------------------------------


def understand(question: str, history: list[str] = ()) -> dict:
    """LLM call 1, at temperature 0. Returns the cleaned understanding (see parse_understanding)."""
    # Give the model the last few messages, so "something for my room" after
    # "my leg is broken" keeps its meaning.
    recent = history[-MEMORY_MESSAGES:]
    earlier = " | ".join(message[:MEMORY_CHARS] for message in recent)

    user_message = f"Customer: {question}"
    if earlier:
        user_message = f"Earlier messages: {earlier}\n{user_message}"

    # The instructions go in the system message. Put in the user message, the model
    # echoes "/no_think" instead of answering.
    messages = [
        {"role": "system", "content": UNDERSTAND},
        {"role": "user", "content": user_message},
    ]
    reply = llm(messages, temperature=0, json_mode=True)
    return parse_understanding(reply)


def parse_understanding(reply: str) -> dict:
    """Turn the model's reply into a dict with every key present and clean values.

    Missing or malformed parts fall back to empty values, and an unknown intent becomes "".
    So the caller never has to check the shape.
    """
    # Find the JSON object, even if the model wrapped it in text or code fences.
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        raw = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    def text(key: str) -> str:
        """A lowercase string, with stray quotes and punctuation removed."""
        value = raw.get(key)
        if not isinstance(value, str):
            return ""
        return value.strip(" ./*\"'").lower()

    def items(key: str) -> list[str]:
        """A list of lowercase strings. A single string counts as a list of one."""
        value = raw.get(key)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [item.strip().lower() for item in value if isinstance(item, str) and item.strip()]

    intent = text("intent").upper()
    meaning = raw.get("meaning")

    return {
        "intent": intent if intent in INTENTS else "",
        "item": text("item"),
        "problem": text("problem"),
        "emotion": text("emotion") or "neutral",
        "sentiment": text("sentiment") or "neutral",
        "furniture": items("furniture"),
        "constraints": items("constraints"),
        "meaning": meaning.strip() if isinstance(meaning, str) else "",
    }


# --------------------------------------------------------------------------------------
# Fixes in Python
# --------------------------------------------------------------------------------------


def is_broken_furniture(text: str) -> bool:
    """True when a piece of furniture broke: "the leg of my table snapped".

    False for a body part: "my leg is broken". Body words are removed before looking
    for a furniture word.
    """
    if not BROKEN.search(text):
        return False
    words = set(re.findall(r"[a-z]+", text.lower()))
    words = words - NOT_FURNITURE - NOT_PRODUCTS
    return any(in_catalogue(word) for word in words)


def read_message(question: str, history: list[str] = ()) -> dict:
    """understand(), then correct two labels the small model gets wrong."""
    u = understand(question, history)
    furniture_broke = is_broken_furniture(question)

    # "The leg of my table snapped" is read as an injury. It is a complaint.
    if u["intent"] in ("PRODUCT_RECOMMENDATION", "") and furniture_broke:
        u["intent"] = "COMPLAINT"

    # A flood or an accident at home is read as a complaint. It is not about
    # something the store sold, so it is a need.
    elif u["intent"] == "COMPLAINT" and not (furniture_broke or PURCHASED.search(question)):
        u["intent"] = "PRODUCT_RECOMMENDATION"

    return u
