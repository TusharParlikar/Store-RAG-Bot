"""The chat page. Run from the project root:  streamlit run app/main.py

This file is the chat loop only: take a message, get the answer, show it.
How the page looks (header, sidebar, buttons) is in app/ui.py.

Streamlit runs this whole file again on every message and every button click.
What must survive between runs (the messages and the cart) lives in st.session_state.
"""

import re
import sys
from pathlib import Path

# Make the project root importable when Streamlit starts this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import ui
from gen.answer import MIN_SCORE, answer
from gen.catalogue import named_products
from gen.picking import listed
from gen.understand import understand
from gen.warranty import report
from rag.index import search

# A chat message that asks whether something is still under warranty. The bot cannot know
# the purchase date, so the reply also points to the checker in the sidebar.
ASKS_WARRANTY_STATUS = re.compile(
    r"\b(still|is|are)\b.*\b(under warranty|in warranty|covered)\b"  # "is it still covered?"
    r"|\bwarranty\b.*\b(expired?|status|valid|left|over)\b"  # "has my warranty expired?"
    r"|\bcheck\b.*\bwarranty\b",  # "check warranty"
    re.I,
)
WARRANTY_HINT = (
    "To check a product you bought, use **Warranty check** in the sidebar: "
    "type the product and pick the purchase date."
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the store catalogue and waking the assistant...")
def warm_up():
    """Run once per server start, so the first customer does not wait for the loading."""
    # Loads the embedding model and the FAISS index.
    search("chair", k=1)

    # Loads the LLM and caches the long understanding prompt.
    # On a laptop the first reply then takes about 10 seconds instead of 2 minutes.
    try:
        understand("hello")
    except Exception:
        # The LLM is down. The chat shows the error on the first message instead.
        pass


def warranty_reply(product_text: str, purchase) -> str:
    """The answer of the sidebar warranty checker. Plain Python: no LLM, so it always works.

    The product is found by name ("MARKUS") or, failing that, by search ("my sofa").
    """
    named = named_products(product_text)
    found = [hit for hit in search(product_text) if hit["kind"] == "product"]

    if named:
        product = named[0]
    elif found and found[0]["score"] >= MIN_SCORE:
        product = found[0]
    else:
        return (
            f"I could not find a product called **{product_text}**. "
            'Please check the name, for example "MARKUS office chair".'
        )
    return report(product["name"], purchase, product["warranty_months"])


def get_reply(prompt: str, live) -> dict:
    """Ask the bot for a reply.

    `live` is the slot on the page where the reply appears word by word while it is written.
    """
    messages = st.session_state.messages

    # What the customer said before this message.
    earlier = [message["content"] for message in messages[:-1] if message["role"] == "user"]

    # The products the bot's last reply listed: what "the second one" refers to.
    last_bot = next((m for m in reversed(messages) if m["role"] == "assistant"), {})
    shown = listed(last_bot.get("content", ""), last_bot.get("sources", []))

    try:
        result = answer(
            prompt,
            history=earlier,
            shown=shown,
            cart=st.session_state.cart,
            on_token=lambda text_so_far: live.markdown(text_so_far + " ▌"),
        )
    except Exception as error:
        # The LLM is down or the key is wrong: say so instead of showing a stack trace.
        name = type(error).__name__
        result = {
            "text": f"Sorry, I can't reach the language model right now ({name}).",
            "sources": [],
            "missing": None,
        }

    # The model returned nothing: never leave a blank bubble.
    if not result["text"].strip():
        result["text"] = "Sorry, I lost my words for a moment. Could you ask that again?"

    # A question like "is my chair still under warranty?" also gets the way to check it.
    if ASKS_WARRANTY_STATUS.search(prompt):
        result["text"] += "\n\n" + WARRANTY_HINT
    return result


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------

ui.header()
warm_up()

# A new visitor starts with the welcome message and an empty cart.
if "messages" not in st.session_state:
    ui.new_chat()

warranty_request = ui.sidebar()

# The sidebar warranty checker was used: put the question and its answer in the chat.
if warranty_request:
    product_text, purchase = warranty_request
    question = f"Warranty check: {product_text}, bought on {purchase:%d %b %Y}"
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": warranty_reply(product_text, purchase)}
    )

# Everything said so far.
for number, message in enumerate(st.session_state.messages):
    ui.message(message, key=str(number))

# The next message: typed, or sent by a button ("queued").
prompt = st.chat_input("Ask me anything about our furniture...")
if not prompt:
    prompt = st.session_state.pop("queued", None)

# Before the first message, offer three starter questions.
only_welcome_so_far = len(st.session_state.messages) == 1
if only_welcome_so_far and not prompt:
    prompt = ui.example_buttons()

if prompt:
    # Show the customer's message.
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    ui.message(user_message, key="new-user")

    # Get and show the bot's reply.
    with st.chat_message("assistant", avatar="🪑"):
        live = st.empty()
        # Shown at once, so the bubble is never empty while the model reads the message.
        live.markdown("_Thinking..._")

        result = get_reply(prompt, live)
        reply = {
            "role": "assistant",
            "content": result["text"],
            "sources": result["sources"],
            "missing": result["missing"],
            "product": result.get("product"),
        }

        # The final text replaces the streamed one. It may have more: the
        # "not available" line, product links, the next step.
        live.markdown(reply["content"])
        ui.extras(reply, key=str(len(st.session_state.messages)))

    st.session_state.messages.append(reply)

    # The sidebar cart was drawn before this reply. If the cart changed, draw the page again.
    new_cart = result.get("cart", st.session_state.cart)
    if new_cart != st.session_state.cart:
        st.session_state.cart = new_cart
        st.rerun()
