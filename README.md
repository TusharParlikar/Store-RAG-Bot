# Store RAG Bot

A chat assistant for a furniture store. It answers only from the store's own data and helps customers find, and buy, furniture that suits their situation.

Customers rarely ask for a product by name. They say "my leg is broken" or "we're expecting a baby", and a plain search on those words finds table legs. This bot first works out what the customer means, then searches for furniture that helps, and answers with real prices and policies from the catalogue. If the data does not hold the answer, it says "I don't know".

## Example

> **Customer:** my leg is broken
>
> **Bot:** Hello! I'm sorry to hear you're feeling pain, especially with your leg. Here are some products that could help:
> 1. **INGATORP - Chair with armrests** – ₹10,458. This chair provides support and comfort for long hours, which could help you stay seated while recovering.
> 2. **REMSTA - Armchair** – ₹18,682. ...
>
> Would you like one of these? Tell me the number and I'll share the full details.
>
> **Customer:** 1
>
> **Bot:** *(a short description, then)* Price: ₹10,458 · Warranty: 60 months (5 years) · Product page link · buttons: What goes with it? / Warranty and returns / Similar options

Replies are from real runs, shortened.

## Features

- **Understands the situation**: reads intent, feeling and needs from each message and remembers the last 3 messages.
- **Empathy, once**: sympathy for a problem or congratulations for good news, then up to 3 products and how each helps.
- **Facts only from the data**: prices in ₹, warranty lengths and policies come from `data/`, never from the model's memory.
- **Honest about stock**: says when a product is not sold, logs the request for the store, and suggests the closest match.
- **Guides the purchase**: "2", "1st", "the cheapest one" or "I'll take the HATTEFJÄLL" shows the product's facts, its page link and next-step buttons.
- **Direct links**: name a product ("How much is the MALM bed?") and the reply ends with links to its product pages.
- **Warranty check**: from a purchase date, Python works out whether the item is still covered.
- **Local or hosted model**: Ollama on your machine, or any OpenAI-compatible API such as Groq. Same code.

## Quick start

Prerequisites: Python 3.11 or 3.12 and [Ollama](https://ollama.com).

```bash
git clone https://github.com/TusharParlikar/Store-RAG-Bot.git
cd Store-RAG-Bot
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
ollama pull qwen3:1.7b
cp .env.example .env            # defaults point at local Ollama
streamlit run app/main.py
```

Open http://localhost:8501. The first start downloads the embedding model, builds the search index and loads the LLM, which takes a few minutes on a laptop.

## How it works

```mermaid
sequenceDiagram
    actor C as Customer
    participant UI as app/main.py (Streamlit)
    participant A as gen/answer.py
    participant L as LLM (Ollama or Groq)
    participant F as rag/index.py (FAISS)
    C->>UI: "my leg is broken"
    UI->>A: message + earlier messages + products shown last
    A->>L: Call 1: understand the message (JSON)
    L-->>A: intent, emotion, helpful furniture, meaning
    Note over A: Python picks the path from the intent
    A->>F: search "armchair with armrests footstool"
    F-->>A: closest products and policies, with scores
    A->>L: Call 2: store rules + products + needs + task
    L-->>A: reply text, streamed
    Note over A: Python adds links, the next step, the "not available" line
    A-->>UI: text, sources, missing item, picked product
    UI-->>C: reply, buttons, sources
```

1. **Understand.** One LLM call describes the message as JSON: intent, feeling, the furniture that would help.
2. **Route.** Python picks the path: recommend, search, purchase, policy, chit-chat or "I don't know".
3. **Retrieve.** The question, or the helpful furniture, is searched in a FAISS index of products and policy sections.
4. **Answer.** A second LLM call writes the reply from the retrieved facts only.
5. **Clean up.** Python adds the stock notice, product links and a next step.

The model never decides on its own: it describes the message and puts facts into words, and Python does the rest. Full detail, including the intent table and the build phase, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Configuration

All settings are environment variables, read by [config.py](config.py). See [.env.example](.env.example).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_BASE_URL` | no | `http://localhost:11434/v1` | OpenAI-compatible endpoint (Ollama or Groq) |
| `LLM_API_KEY` | no | `ollama` | API key; set `<YOUR_GROQ_API_KEY>` for Groq |
| `LLM_MODEL` | no | `qwen3:1.7b` | Model name (`qwen/qwen3-32b` on Groq) |
| `LLM_TEMPERATURE` | no | `0.8` | Warmth of product replies (understanding uses 0, policy answers 0.3) |
| `LLM_REASONING_EFFORT` | no | `none` | `none` turns Qwen3 thinking off |
| `LLM_KEEP_ALIVE` | no | `2h` for a local endpoint, otherwise unset | How long Ollama keeps the model loaded |
| `EMBED_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |

`.env` is git-ignored. Never commit keys.

## Project structure

```
app/main.py              Streamlit chat page, next-step buttons, warranty date box
gen/answer.py            understanding, routing, stock check, purchase flow
gen/prompts.py           all prompt text
gen/warranty.py          warranty date maths in plain Python
rag/index.py             build the FAISS index and search it
nlp/chunks.py            chunking, embeddings, rupee formatting
nlp/prepare_products.py  raw IKEA file to the product table
data/raw/                downloaded dataset
data/products/           product table used by the bot
data/rules/              warranty, returns and expiry policies (demo policies)
tests/scenarios.py       end-to-end chat scenarios
docs/                    architecture and deployment notes
config.py                settings from environment variables
```

## Tests

```bash
python -m gen.warranty          # warranty date edge cases
python -m gen.answer            # Python self-checks, then a few live questions
python -m tests.scenarios       # 77 chat scenarios against the real index and LLM
```

The scenarios cover prices, policies, needs and feelings, products not sold, off-topic questions, chit-chat, complaints, memory, picking a product and named-product links. Results vary a little between runs, because the small model does not always give the same answer.

## Data

Real IKEA product data, not generated: the IKEA Saudi Arabia scrape from [TidyTuesday 2020-11-03](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-11-03), with 2,962 unique products in 17 categories. Prices are converted from SAR to INR at a fixed rate (`SAR_TO_INR` = 23.5). The dataset has no warranty, benefit or pairing columns, so these are set per category in `CATEGORY_INFO` in [nlp/prepare_products.py](nlp/prepare_products.py). The store name and the policies in [data/rules/](data/rules/) are made up for the demo, and product links go to IKEA's site.

To rebuild from the raw file: `python nlp/prepare_products.py`, then `python -m rag.index`.

## Limitations

With the local `qwen3:1.7b` model on a laptop CPU (i5-1335U, no GPU):
- **Slow.** The first words of a reply appear after 20 to 35 seconds, and a full reply takes 25 to 55 seconds.
- **Small-model slips.** It sometimes skips the requested opening, drifts towards health wording ("pain relief") despite the comfort-only rule, or answers a complaint with the return policy instead of the warranty route.
- **One product per pick.** "I'll take the first and the third" picks one.

A larger hosted model (`qwen/qwen3-32b` on Groq) should be faster and more accurate. That is untested.

## Roadmap

Not built yet:
- A cart, and suggestions for what goes with the items in it.
- A planner for bigger setups such as "an office for 30 people".
- Saving the final order with an order number.
- Deployment: the plan is in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## License

No license file yet.
