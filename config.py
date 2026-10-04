"""All settings come from environment variables (.env locally, Secrets on Streamlit Cloud)."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.45"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
