"""How the chat page looks. The chat loop itself is in app/main.py.

header()           page title and style
sidebar()          the cart and the warranty date box
message()          one chat bubble
extras()           what goes under a reply: buttons, the "request noted" note, sources
example_buttons()  starter questions for a new chat
"""

from datetime import date

import streamlit as st

from gen import cart as cart_
from nlp.chunks import inr

# --------------------------------------------------------------------------------------
# Texts and style
# --------------------------------------------------------------------------------------

# The bot speaks first and asks what the customer wants.
WELCOME = {
    "role": "assistant",
    "content": (
        "Hello, welcome to Nest & Oak! What are you looking for today? "
        "Tell me what you need, or what is going on at home, and I'll find furniture that fits. "
        "Pick as many as you like, then say **check out**."
    ),
}

# Starter questions shown before the customer has typed anything.
EXAMPLES = [
    "How much is the NORDVIKEN bar table?",
    "Can I return an assembled item?",
    "Do you have a coffee table?",
]

AVATARS = {"user": "🙂", "assistant": "🪑"}

STYLE = """
<style>
.block-container {padding-top: 2rem; max-width: 760px;}
.hero {background: linear-gradient(135deg, #B5651D 0%, #8B4A16 100%); color: #fff;
       padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {font-size: 1.6rem; margin: 0; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .9;}
.src {display: inline-block; background: #F1EADF; border-radius: 999px; padding: 2px 10px;
      margin: 2px 4px 2px 0; font-size: .78rem; color: #5A4632;}
.note {background: #FFF4E5; border-left: 3px solid #B5651D; padding: .4rem .8rem;
       border-radius: 6px; font-size: .85rem; margin-top: .4rem;}
</style>
<div class="hero">
  <h1>🪑 Nest &amp; Oak Assistant</h1>
  <p>Ask about prices, warranty and returns, or tell me what you need.</p>
</div>
"""


# --------------------------------------------------------------------------------------
# Page state
# --------------------------------------------------------------------------------------


def new_chat():
    """Start over: only the welcome message, and an empty cart."""
    st.session_state.messages = [WELCOME]
    st.session_state.cart = []


def send(text: str):
    """Send a message for the customer, as if they had typed it. Every button uses this."""
    st.session_state.queued = text
    st.rerun()


# --------------------------------------------------------------------------------------
# Page parts
# --------------------------------------------------------------------------------------


def header():
    st.set_page_config(page_title="Nest & Oak Assistant", page_icon="🪑", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)


def sidebar() -> date | None:
    """Draw the sidebar. Returns the purchase date when the customer gave one, else None."""
    with st.sidebar:
        # The cart, with a Check out button once something is in it.
        cart = st.session_state.cart
        st.subheader(f"🛒 Cart ({len(cart)})")
        if cart:
            st.markdown(cart_.listing(cart))
            st.caption(cart_.size(cart))
            if st.button("Check out", type="primary", use_container_width=True):
                send("check out")
        else:
            st.caption("Empty. Tell me the number of a product to add it.")

        st.divider()

        # The purchase date for warranty questions.
        st.subheader("Warranty check")
        use_date = st.toggle("I know my purchase date")
        purchase = st.date_input(
            "Purchase date",
            value=date.today(),
            max_value=date.today(),
            disabled=not use_date,
        )
        st.caption("Used when you ask about warranty.")

        st.divider()

        if st.button("Clear chat", use_container_width=True):
            new_chat()
            st.rerun()

    return purchase if use_date else None


def example_buttons() -> str | None:
    """Three starter questions. Returns the one clicked, if any."""
    st.caption("Try one of these:")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        if column.button(example, use_container_width=True):
            return example
    return None


def message(msg: dict, key: str):
    """One chat bubble, with its extras underneath."""
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        extras(msg, key)


# --------------------------------------------------------------------------------------
# Under a reply
# --------------------------------------------------------------------------------------


def extras(msg: dict, key: str):
    """What goes under a reply. `key` keeps the buttons of different replies apart."""
    # Next-step buttons, after the customer picked one product.
    if msg.get("product"):
        next_steps(msg["product"], key)

    # A note that a request for something not sold was passed on.
    if msg.get("missing"):
        note = f'<div class="note">📝 Request noted: <b>{msg["missing"]}</b></div>'
        st.markdown(note, unsafe_allow_html=True)

    # The products and policies the reply is based on.
    if msg.get("sources"):
        with st.expander("Sources"):
            for source in msg["sources"]:
                show_source(source)


def next_steps(product: dict, key: str):
    """Buttons after a pick. Each sends a follow-up question; the first opens the product page."""
    # The questions use the product type, not its name: a name would read as
    # "buy this one" again. The size is dropped too: "bookcase, 120x30x237 cm"
    # makes the warranty question look like a product search.
    product_type = product["name"].partition(" - ")[2].split(",")[0].lower()
    kind = product_type or product["category"].lower()
    goes_with = product["goes_with"].split(";")[0].strip().lower()
    months = product["warranty_months"]

    questions = {
        "What goes with it?": f"Show me {goes_with}",
        "Warranty and returns": (
            f"What does the {months}-month warranty on a {kind} cover, and can I return it?"
        ),
        "Similar options": f"Show me other {kind} options",
    }

    columns = st.columns(len(questions) + 1)
    if product.get("link"):
        columns[0].link_button("Product page", product["link"], use_container_width=True)
    for column, (label, question) in zip(columns[1:], questions.items()):
        if column.button(label, key=f"{key}-{label}", use_container_width=True):
            send(question)


def show_source(source: dict):
    """One line of the Sources panel: where it came from, how close a match, and a link."""
    label = f'<span class="src">{source["source"]} · {source["score"]:.2f}</span>'
    if source.get("link"):
        product = f'[{source["name"]}]({source["link"]}) — {inr(source["price"])}'
        st.markdown(f"{label} {product}", unsafe_allow_html=True)
    else:
        st.markdown(label, unsafe_allow_html=True)
