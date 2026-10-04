"""Step 6: Streamlit chat page. Run from the project root: streamlit run app/main.py"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from gen.answer import answer
from gen.warranty import describe
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


@st.cache_resource(show_spinner="Loading the store catalogue...")
def warm_up():
    search("chair", k=1)  # loads the embedding model and FAISS index once per server


warm_up()

with st.sidebar:
    st.subheader("Warranty check")
    use_date = st.toggle("I know my purchase date")
    purchase = st.date_input("Purchase date", value=date.today(), max_value=date.today(),
                             disabled=not use_date)
    st.caption("Used when you ask about warranty.")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_extras(msg: dict):
    if msg.get("missing"):
        st.markdown(f'<div class="note">📝 Request noted: <b>{msg["missing"]}</b></div>', unsafe_allow_html=True)
    if msg.get("sources"):
        with st.expander("Sources"):
            for s in msg["sources"]:
                label = f'{s["source"]} · {s["score"]:.2f}'
                if s.get("link"):
                    st.markdown(f'<span class="src">{label}</span> [{s["name"]}]({s["link"]}) — {s["price"]:g} SAR',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="src">{label}</span>', unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙂" if msg["role"] == "user" else "🪑"):
        st.markdown(msg["content"])
        show_extras(msg)

prompt = st.chat_input("Ask me anything about our furniture...")
if not st.session_state.messages and not prompt:
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
        with st.spinner("Thinking..."):
            try:
                r = answer(prompt, extra)
            except Exception as e:  # LLM down or bad key: show it instead of a stack trace
                r = {"text": f"Sorry, I can't reach the language model right now ({type(e).__name__}).",
                     "sources": [], "missing": None}
        reply = {"role": "assistant", "content": r["text"], "sources": r["sources"], "missing": r["missing"]}
        st.markdown(reply["content"])
        show_extras(reply)
    st.session_state.messages.append(reply)
