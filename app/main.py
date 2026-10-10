"""Streamlit chat page. Run from the project root: streamlit run app/main.py"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from gen import cart as cart_
from gen.answer import answer, listed, understand
from gen.warranty import describe
from nlp.chunks import inr
from rag.index import search

st.set_page_config(page_title="Nest & Oak Assistant", page_icon="🪑", layout="centered")

st.markdown("""
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
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the store catalogue and waking the assistant...")
def warm_up():
    search("chair", k=1)  # loads the embedding model and FAISS index once per server
    try:  # loads the LLM and caches the long understanding prompt: the first customer then waits ~10s, not ~2 min
        understand("hello")
    except Exception:  # LLM down: the chat shows the error on the first message instead
        pass


warm_up()

# The bot speaks first and asks what the customer wants.
WELCOME = {"role": "assistant", "content": "Hello, welcome to Nest & Oak! What are you looking for today? "
           "Tell me what you need, or what is going on at home, and I'll find furniture that fits. "
           "Pick as many as you like, then say **check out**."}
if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME]
    st.session_state.cart = []

with st.sidebar:
    cart = st.session_state.cart
    st.subheader(f"🛒 Cart ({len(cart)})")
    if cart:
        st.markdown(cart_.listing(cart))
        st.caption(cart_.size(cart))
        if st.button("Check out", type="primary", use_container_width=True):
            st.session_state.queued = "check out"
            st.rerun()
    else:
        st.caption("Empty. Tell me the number of a product to add it.")
    st.divider()
    st.subheader("Warranty check")
    use_date = st.toggle("I know my purchase date")
    purchase = st.date_input("Purchase date", value=date.today(), max_value=date.today(),
                             disabled=not use_date)
    st.caption("Used when you ask about warranty.")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = [WELCOME]
        st.session_state.cart = []
        st.rerun()


def show_next_steps(product: dict, key: str):
    """Buttons after a customer picks a product: each one sends a follow-up message, or opens the product page."""
    # The product type, not its name: a name would read as "buy this one" again.
    # Size dropped too: "bookcase, 120x30x237 cm" makes the warranty question look like a product search.
    kind = product["name"].partition(" - ")[2].split(",")[0].lower() or product["category"].lower()
    steps = {"What goes with it?": f"Show me {product['goes_with'].split(';')[0].strip().lower()}",  # the product's words would find it again
             "Warranty and returns": f"What does the {product['warranty_months']}-month warranty on a {kind} cover, and can I return it?",
             "Similar options": f"Show me other {kind} options"}
    cols = st.columns(len(steps) + 1)
    if product.get("link"):
        cols[0].link_button("Product page", product["link"], use_container_width=True)
    for col, (label, text) in zip(cols[1:], steps.items()):
        if col.button(label, key=f"{key}-{label}", use_container_width=True):
            st.session_state.queued = text
            st.rerun()


def show_extras(msg: dict, key: str):
    if msg.get("product"):
        show_next_steps(msg["product"], key)
    if msg.get("missing"):
        st.markdown(f'<div class="note">📝 Request noted: <b>{msg["missing"]}</b></div>', unsafe_allow_html=True)
    if msg.get("sources"):
        with st.expander("Sources"):
            for s in msg["sources"]:
                label = f'{s["source"]} · {s["score"]:.2f}'
                if s.get("link"):
                    st.markdown(f'<span class="src">{label}</span> [{s["name"]}]({s["link"]}) — {inr(s["price"])}',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="src">{label}</span>', unsafe_allow_html=True)


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "🪑"):
        st.markdown(msg["content"])
        show_extras(msg, str(i))

prompt = st.chat_input("Ask me anything about our furniture...") or st.session_state.pop("queued", None)
if len(st.session_state.messages) == 1 and not prompt:  # only the welcome so far
    st.caption("Try one of these:")
    cols = st.columns(3)
    for col, example in zip(cols, ["How much is the NORDVIKEN bar table?",
                                   "Can I return an assembled item?",
                                   "Do you have a coffee table?"]):
        if col.button(example, use_container_width=True):
            prompt = example

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    extra = ""
    if use_date and "warrant" in prompt.lower():
        product = next((h for h in search(prompt) if h["kind"] == "product"), None)
        if product:
            extra = describe(product["name"], purchase, product["warranty_months"])

    with st.chat_message("assistant", avatar="🪑"):
        live = st.empty()  # the reply appears here word by word while the model writes it
        live.markdown("_Thinking..._")  # at once, so the bubble is never empty while the model reads the message
        try:
            earlier = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
            last_bot = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), {})
            shown = listed(last_bot.get("content", ""), last_bot.get("sources", []))
            r = answer(prompt, extra, history=earlier, shown=shown, cart=st.session_state.cart,
                       on_token=lambda t: live.markdown(t + " ▌"))
        except Exception as e:  # LLM down or bad key: show it instead of a stack trace
            r = {"text": f"Sorry, I can't reach the language model right now ({type(e).__name__}).",
                 "sources": [], "missing": None}
        if not r["text"].strip():  # the model returned nothing: never leave a blank bubble
            r["text"] = "Sorry, I lost my words for a moment. Could you ask that again?"
        reply = {"role": "assistant", "content": r["text"], "sources": r["sources"], "missing": r["missing"],
                 "product": r.get("product")}
        live.markdown(reply["content"])  # final text: adds the "not available" line and the next step
        show_extras(reply, str(len(st.session_state.messages)))
    st.session_state.messages.append(reply)
    if r.get("cart", st.session_state.cart) != st.session_state.cart:
        st.session_state.cart = r["cart"]
        st.rerun()  # redraw the sidebar cart, which was drawn before this reply
