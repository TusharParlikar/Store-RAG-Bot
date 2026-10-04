"""All settings come from environment variables (.env locally, Secrets on Streamlit Cloud)."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none")  # "none" turns Qwen3 thinking off
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
# How long Ollama keeps the model loaded after a reply. Reloading costs ~80s on a laptop CPU.
# Sent only to a local endpoint: hosted APIs such as Groq do not know this field.
_LOCAL = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "2h" if _LOCAL else "")
EMBED_MODEL =os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
