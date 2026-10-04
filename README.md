# Store RAG Bot

A chat assistant for a furniture store that answers only from the store's own data and helps customers find, and buy, furniture that suits their situation.

Customers rarely ask for a product by name. They say "my leg is broken" or "we're expecting a baby". A plain search on those words finds table legs and baby-shaped nothing. This bot first works out what the customer means, then searches for the furniture that helps, and answers with real prices and policies from the catalogue. If the data does not contain the answer, it says "I don't know" instead of guessing.

## Features

- **Understands the situation**: reads intent, feeling and needs from each message, and remembers the last 3 messages.
- **Empathy first**: sympathy for a problem, congratulations for good news, then up to 3 products and how each one helps.
- **Facts only from the data**: prices in ₹, warranty lengths and policies come from `data/`, never from the model's memory.
- **Honest about stock**: "Sorry, laptop is not available in our store right now", logs the request for the store, and suggests the closest match.
- **Guides the purchase**: "the second one" or "I'll take the HATTEFJÄLL" shows the exact product facts and next-step buttons.
- **Warranty check**: from a purchase date, Python works out whether the item is still covered.
- **Runs locally or free in the cloud**: Ollama on your machine, or Groq's free tier when deployed. Same code.

## Architecture

```mermaid
flowchart LR
    subgraph Build["Build phase (offline, when data changes)"]
        RAW[data/raw/ikea.csv] --> PREP[nlp/prepare_products.py]
        PREP --> PROD[data/products/products.csv]
        RULES[data/rules/*.md] --> CHUNK[nlp/ chunk + embed<br/>all-MiniLM-L6-v2]
        PROD --> CHUNK
        CHUNK --> IDX[(rag/ FAISS index)]
    end

    subgraph Ask["Ask phase (every message)"]
        USER([Customer]) --> APP[app/ Streamlit chat<br/>cart, Done button]
        APP --> PLAN{plan/ router}
        PLAN -->|question / need| SEARCH[rag/ search top 3-5]
        PLAN -->|setup| SETUP[plan/ setup planner<br/>rules/setups + Python maths]
        SETUP --> SEARCH
        PLAN -->|done| SAVE[gen/ closing remark]
        SEARCH --> IDX
        SEARCH --> GEN[gen/ prompt + LLM call]
        GEN --> LLM[[LLM via OpenAI-compatible API<br/>Ollama locally, Groq when deployed]]
        GEN --> WARR[gen/ warranty check<br/>plain Python date maths]
        SAVE --> ORD[(data/orders/)]
        GEN --> APP
    end
```

### Customers describing a problem

Every message (except a warranty check with a purchase date) first goes through an understanding step: one LLM call (temperature 0, few-shot, JSON mode) that describes the message before anything is searched. It also sees the customer's last 3 messages, so "I need something for my room" after "my leg is broken" is understood as part of the same situation.

```json
{"intent": "PRODUCT_RECOMMENDATION", "item": "", "problem": "broken leg", "emotion": "pain",
 "sentiment": "negative", "furniture": ["armchair with armrests", "footstool"], "constraints": ["bedroom"],
 "meaning": "Needs bedroom furniture that is easy to use with a broken leg."}
```

The model only describes the message. Python decides what happens next:

| Intent | What happens |
|---|---|
| `PRODUCT_SEARCH`, `PRODUCT_COMPARISON` | Search the question. The `item` is checked against the catalogue, see below. |
| `PRODUCT_RECOMMENDATION` | Search for the `furniture` that helps, not the words of the problem (which would match table legs), and drop spare parts. |
| `PURCHASE` | The customer picked one product: details and next steps, see below. |
| `STORE_INFORMATION`, `ORDER_SUPPORT`, `COMPLAINT` | Policy sections only. |
| `CASUAL_CONVERSATION` | A short friendly reply with no search and no store facts. |
| `GENERAL_QUESTION` | "I don't know", without a second LLM call. |

The opening of the reply follows the sentiment: sympathy for a problem or complaint, congratulations for good news ("a baby is coming"), and a short acknowledgement for a neutral situation. The answer step also receives the meaning, situation, feeling and limits, so it answers the reason behind the request, not only its words. If the JSON is broken, the message falls back to a plain search with the score cutoff.

Benefits stay at comfort level: no medical advice or promises.

### Customers who want to buy

When a customer picks a product ("I'd like the HATTEFJÄLL chair, tell me more", "the second one", "ok I'll get that"), Python works out which product they mean. It checks names first (the products in the last reply come first), then words like "second" or "option 2". If only one product was listed, that one is picked. A buying phrase right after a list counts too, because the small model often reads "ok I'll get that" as the earlier need again.

- **One product found:** a short, warm paragraph from the LLM, then the exact facts from the data (price, warranty, category, good for, goes well with). Under it are next-step buttons: Product page, What goes with it?, Warranty and returns, Similar options.
- **Several listed, none named:** the bot asks which one, with a numbered list.

The cart and saving the order come later (project.md steps 8 and 10).

### Products the store does not carry

When a customer asks for something not in the catalogue (for example a desk lamp or a carpet), the bot:
1. Says honestly that it is not available and that the request was passed to the team.
2. Logs the request in `data/requests/requests.csv` (time, item, question), so the store can see what people ask for.
3. Suggests the closest products it does have, with price and benefit.

How it decides: the understanding step names the product kind (`item`). Python then checks whether that word appears in any product name or category. A 1.7b model cannot judge stock reliably on its own.

Key rules from the design:
- `data/` is the only place facts live. The LLM only puts retrieved facts into words.
- Low search score means "I don't know" without calling the LLM.
- All maths (warranty dates, setup quantities) is plain Python, never the LLM.

## Stack

| Part | Choice |
|---|---|
| Frontend | Streamlit |
| LLM | `qwen3:1.7b` on Ollama (local), `qwen/qwen3-32b` on Groq free tier (deployed). Same code, switched by `.env`. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (CPU) |
| Vector search | FAISS |
| Config | `.env` read by [config.py](config.py) |

## Dataset

Real IKEA product data, not generated: the IKEA Saudi Arabia scrape from TidyTuesday 2020-11-03 (also on Kaggle as "IKEA SA Furniture Web Scraping"). 3,694 rows, 2,962 unique products, 17 categories. Raw prices are in SAR; [nlp/prepare_products.py](nlp/prepare_products.py) converts them to INR at `SAR_TO_INR` (23.5) and keeps the original in `price_sar`.

- Raw file: [data/raw/ikea.csv](data/raw/ikea.csv)
- Source: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-11-03

The dataset has name, category, price and description. The extra columns project.md needs (`warranty_months`, `benefit`, `good_for`, `goes_with`) are set per category in `CATEGORY_INFO` in [nlp/prepare_products.py](nlp/prepare_products.py). The policy files in [data/rules/](data/rules/) are demo store policies.

## Run locally

Use Python 3.11 or 3.12 (`faiss-cpu` wheels may lag on newer versions).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
ollama pull qwen3:1.7b
cp .env.example .env            # defaults point at local Ollama
python nlp/prepare_products.py  # rebuild data/products/products.csv
```

```bash
python -m rag.index             # build the FAISS index (also built automatically on first run)
streamlit run app/main.py       # chat page at http://localhost:8501
```

## Tests

```bash
python -m gen.warranty          # warranty date edge cases
python -m tests.scenarios       # 22 end-to-end chat scenarios against the real index and LLM
python -m tests.scenarios 3     # same, 3 rounds each to catch flaky LLM output
```

On CPU with `qwen3:1.7b`, each reply takes about 10 to 30 seconds (two LLM calls: product check and answer).

## Configuration

All settings are environment variables. See [.env.example](.env.example).

| Variable | Meaning |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint (Ollama or Groq) |
| `LLM_API_KEY` | API key (`ollama` for local) |
| `LLM_MODEL` | Model name |
| `LLM_TEMPERATURE` | 0.8 for warmer replies (the understanding step always uses 0) |
| `LLM_REASONING_EFFORT` | `none` turns Qwen3 thinking off (about 30x faster) |
| `EMBED_MODEL` | Sentence-transformers model |

`.env` is git-ignored. Never commit keys.
