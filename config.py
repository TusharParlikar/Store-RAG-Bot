"""All settings, read from environment variables.

Locally they come from the .env file; on Streamlit Cloud from the app's Secrets.
Every setting has a default that works with Ollama on this machine.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- The language model: any OpenAI-compatible endpoint ---------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")

# How warm and varied product replies are. Understanding always uses 0, policy answers 0.3.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

# "none" turns Qwen3's thinking off. Some hosted models reject "none": use "low" there.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none")

# How long Ollama keeps the model loaded after a reply. Reloading costs over a minute
# on a laptop CPU. Sent only to a local endpoint: hosted APIs do not know this field.
_LOCAL = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "2h" if _LOCAL else "")

# --- The embedding model for search ------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
