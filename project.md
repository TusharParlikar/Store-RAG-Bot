# Store RAG Bot

A chatbot for a store (example: a furniture store). It answers questions from the store's own documents, recommends products and explains why they help, suggests what goes with each pick, plans bigger setups, and saves the final purchase. If the documents don't say it, the bot says "I don't know".

## What it does

1. **Answers questions** from the store's documents: price, warranty, returns, expiry.
2. **Explains benefits:** "I have back pain" gets a chair and a lumbar cushion, with reasons.
3. **Completes the set:** pick a table and it suggests chairs and a tablecloth.
4. **Plans bigger setups:** "I want an office for 30 people" becomes a shopping plan.
5. **Closes and saves:** a good closing remark, then the purchase is saved with an order number and a tag.

## Stack and setup

| Part | Choice | Why |
|---|---|---|
| Frontend | Streamlit | Chat page in pure Python, no web code |
| LLM | Ollama, `qwen3:1.7b` | Small, runs locally, free |
| LLM temperature | 0.4 to 0.5 | Replies sound natural but stay close to the data |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Small, fast, runs on CPU |
| Vector search | FAISS | Simple local index, no server |
| Language | Python | |

Before you start:
- Install Python and Ollama, and download the `qwen3:1.7b` model in Ollama.
- Install these Python packages: streamlit, sentence-transformers, faiss-cpu, pandas, ollama.

Notes:
- "3:1.7b" was read as `qwen3:1.7b`, since that is the Ollama model with that size. Change the name here if you meant a different model.
- Qwen3 models can print their own reasoning before the answer. Turn that off (Ollama has a setting for it) or strip it out before showing the answer.

## Folder layout

```
store-rag-bot/
├── project.md
├── data/
│   ├── products/     # product table, one row per product
│   ├── rules/        # warranty, returns, expiry, setups, upsell_style
│   └── orders/       # saved purchases (order number + tag)
├── nlp/              # clean the data, split into chunks, make vectors
├── rag/              # build the index and search it
├── plan/             # read the message, build the setup plan
├── gen/              # prompt, Ollama call, warranty check
└── app/              # Streamlit chat page, cart, Done button
```

`data/` is the only place the store's information lives. Nothing else holds facts.

## How it works

Two phases:
1. **Build (offline, once, and again when the data changes):** `data/` -> `nlp/` -> `rag/` builds the index.
2. **Ask (live, every message):** `app/` -> `plan/` -> `rag/` -> `gen/`.

```
Customer types a message
        |
        v
plan/  decides what it is: question | need | setup | done
   |-- question -> rag/ search -> gen/ answers with sources
   |-- need     -> rag/ search (matches good_for and benefit) -> gen/ explains the benefits
   |-- setup    -> plan/ builds the plan -> rag/ finds products -> gen/ presents it
   `-- done     -> gen/ closing remark -> save the order in data/orders/

Customer presses Add on a product
        -> cart -> gen/ suggests what goes with it

Streamlit shows: answer + sources + cart
```

This is the finished design. The message routing in `plan/` is added in step 9; until then every typed message goes through the normal search and answer.

## Build order

Work through the steps in order. Each step ends with a check; move on only when it passes. Steps 1 to 6 give a working Version 1 (plain document Q&A). Steps 7 to 10 add the sales features one at a time, so Version 1 stays safe if something breaks.

### Step 1: Data (`data/`)

Write the store's information. Start small.

- `products/`: one table with about 12 furniture products (include a table, a chair, an ergonomic chair, a tablecloth, coasters, a lumbar cushion, a desk lamp and a storage cabinet). Columns:
  - name, category, price, warranty_months
  - shelf_life: only for items that expire, such as wood polish; otherwise empty
  - benefit: one line on why it helps (for example "curved back supports the lower back")
  - good_for: tags like back-pain, long-hours, small-room, decoration
  - goes_with: 2 or 3 partner products or categories (a table goes with chairs and a tablecloth)
- `rules/`: one plain-language file each for warranty, returns and expiry. The `upsell_style` and `setups` files are written later, in steps 8 and 9.

Fill every column now so you don't have to edit every row later.

- [ ] **Done when:** every product has all columns filled, and each policy file reads clearly on its own.

### Step 2: Chunk and embed (`nlp/`)

- Read the files in `data/` and clean the text.
- Split into chunks: **one product row = one chunk** (including its benefit and good_for text), **one policy section = one chunk**.
- Turn each chunk into a vector with the embedding model.

- [ ] **Done when:** you can print the chunks and each one makes sense alone: no product row cut in half, no policy rule split across two chunks.

### Step 3: Index and search (`rag/`)

- Build the FAISS index from the chunk vectors. Rebuild it when the data changes.
- Search: embed the question and return the top 3 to 5 chunks with their source names and similarity scores.

- [ ] **Done when:** "warranty on the chair" returns the warranty rule and the chair row, and "back pain" returns the ergonomic chair and the lumbar cushion. An off-topic question gets a clearly low score.

### Step 4: Answer with Ollama (`gen/`)

- Build the prompt from the rules, the retrieved chunks and the question. Send it to Ollama with temperature 0.4 to 0.5 and return the answer with its sources.
- If the best search score is low, skip the LLM and answer "I don't know" directly. A 1.7b model often answers anyway. Pick the cutoff by testing a few off-topic questions.
- Prompt rules, first set:
  1. Answer only from the given context.
  2. If the context doesn't contain the answer, say "I don't know" and suggest contacting the store.
  3. Quote prices and warranty lengths exactly as written.
  4. Say which source the answer came from.
  5. Keep answers short.

- [ ] **Done when:** "How much is the oak table?" gives the right price with its source, "Do you sell sofas?" (when you don't) gives "I don't know", and a question that isn't about the store is refused politely.

### Step 5: Warranty check (`gen/`)

- A small plain-Python function: purchase date + the product's warranty months gives covered or expired, plus the expiry date.
- The LLM never does date math. It only receives the result and puts it in words.
- The purchase date comes from a date box in the app (step 6), and later from a saved order (step 10).

- [ ] **Done when:** the function gives the right answer for a normal date, the last day of the warranty, and the day after it.

### Step 6: Streamlit chat page (`app/`)

- A chat box, with the answer and its sources shown under it.
- An optional purchase-date box for warranty questions.

- [ ] **Done when:** you ask questions in the browser and get answers with sources, and "Is my chair still under warranty?" works with a date set in the box.

**Version 1 works at this point.** Everything below is added on top of it, one feature at a time.

### Step 7: Explain benefits (`gen/`)

- When the customer describes a need, the search matches the `good_for` and `benefit` text. The bot explains why each product helps, using the stored `benefit` line.
- Temperature stays at 0.4 to 0.5.
- Add prompt rule 6: describe benefits at comfort level only ("can help support your lower back"). No medical promises, no cures.

- [ ] **Done when:** "I have back pain" gives an ergonomic chair and a lumbar support cushion, each with a reason in comfort-level wording.

### Step 8: Cart and complete the set (`app/`, `gen/`, `data/rules/`)

- Each recommended product shows an **Add** button. The cart lives in the Streamlit session.
- When a product is added, the bot suggests what goes with it, using `goes_with`, and skips anything already in the cart. Example: a table brings up chairs (a reliable partner for the table) and a tablecloth (for decoration).
- Tone: warm and feeling-led. Help the customer picture the result, such as a table ready for family dinners and guests, instead of listing features.
- Write `rules/upsell_style`: 3 or 4 example lines in this tone. The small model copies the style from examples.
- Add prompt rule 7: suggest at most 3 extra items at a time, only products from the data, only true claims. No fake "only 1 left" and no countdowns. If the customer says no, drop it.

- [ ] **Done when:** adding a table suggests chairs and a tablecloth in a warm tone, never more than 3 suggestions, nothing invented, and nothing already in the cart is suggested again.

### Step 9: Think before answering (`plan/`)

**Routing.** `plan/` reads each typed message and picks one label: question, need, setup or done. Question and need use the normal search and answer (steps 4 and 7). Done is handled in step 10. This reading step may use temperature 0, since it should not vary.

**Setup planner.** For a message like "I want an office for 30 people":
1. **Understand:** what is being set up, for how many people, and is there a budget? If something key is missing, ask one question.
2. **Look at our side:** this is a furniture store, so what can we offer for that setup?
3. **Look it up** in `rules/setups`: each setup type (office, classroom, cafe, bedroom) lists must-haves and nice-to-haves with a quantity per person.
4. **Scale it:** Python multiplies by the number of people and rounds up. The LLM does no maths.
5. **Find real products** for each line using `rag/`.
6. **Present:** must-haves first, then add-ons with a one-line reason each, every product with an Add button.

Write `rules/setups`. Example for an office:

| Item | Type | Quantity |
|---|---|---|
| Chair | Must-have | 1 per person |
| Table or desk | Must-have | 1 per person |
| Coasters | Nice-to-have | 1 per person |
| Desk lamp | Nice-to-have | 1 per person |
| Storage cabinet | Nice-to-have | 1 per 5 people |

Why it works this way: a 1.7b model can't reliably plan on its own. The plan comes from the file, and the model only reads the message and writes the friendly text.

- [ ] **Done when:** "I want to set up an office for 30 people" shows 30 chairs, 30 tables or desks, then coasters, lamps and 6 cabinets as add-ons, using real products. A setup that isn't in the file (a gym, say) gets an honest "I don't have a plan for that yet".

### Step 10: Done and save the order (`gen/`, `app/`, `data/orders/`)

- The customer types "done" or presses the **Done** button.
- The bot shows a summary (items, quantities, total) and a good closing remark.
- Save one record per order in `data/orders/`: order number (ORD-0001, ORD-0002 and so on), tag, date, items, quantities, total.
- The tag is the setup and size (`office-30`), the need (`back-pain`), or `general` when there is neither.
- A customer who gives an order number can ask about warranty without typing a date; the bot reads the date from the saved order.
- No payments. This is only a record.

- [ ] **Done when:** after picking items and saying "done", you get the summary and the closing line, a new record with a number and a tag appears in `data/orders/`, and a warranty question with that order number works.

### Step 11: Full run-through

Run one full conversation: ask a price, describe back pain, add a chair, accept a suggestion, ask for an office for 30, add items, say "done", check the saved order, then ask about its warranty. Fix whatever breaks.

- [ ] **Done when:** the whole conversation runs without errors and the saved order looks right.

## Kept out on purpose

To keep this simple, the first version has no LangChain, no database, no login, no payments, no Docker and no fine-tuning. Add these later only if needed:
- Fine-tune the embedding model on store question and answer pairs (the deep learning step).
- Move prices, stock and orders into SQL, keeping policies in the vector index.