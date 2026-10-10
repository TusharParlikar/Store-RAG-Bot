"""The one place that talks to the language model.

Any OpenAI-compatible API works: Ollama on this machine, or a hosted one such as Groq.
Which one is used comes from the settings in config.py.
"""

import re

from openai import OpenAI

import config

# Longest reply the model may write. Stops runaway answers.
MAX_TOKENS = 400

# The client is created on the first call and then reused.
_client = None


def llm(
    messages: list[dict],
    temperature: float = config.LLM_TEMPERATURE,
    json_mode: bool = False,
    on_token=None,
) -> str:
    """Send one chat request and return the reply text.

    messages     the usual list of {"role": ..., "content": ...}
    temperature  0 = always the same answer, higher = warmer and more varied
    json_mode    ask the model for one JSON object
    on_token     if given, the reply is streamed: on_token(text_so_far) is called after each piece
    """
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    # Optional settings, added only when they apply.
    extra = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    if config.LLM_KEEP_ALIVE:
        # Ollama only: keep the model in memory between replies.
        extra["extra_body"] = {"keep_alive": config.LLM_KEEP_ALIVE}

    streaming = on_token is not None
    out = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        # "none" stops Qwen3 from thinking out loud, which is about 30x faster.
        reasoning_effort=config.LLM_REASONING_EFFORT,
        stream=streaming,
        **extra,
    )

    # Collect the reply: all at once, or piece by piece.
    if not streaming:
        text = out.choices[0].message.content or ""
    else:
        text = ""
        for chunk in out:
            piece = chunk.choices[0].delta.content if chunk.choices else None
            if piece:
                text += piece
                on_token(text)

    return clean(text)


def clean(text: str) -> str:
    """Remove Qwen3's thinking text and its "/no_think" switch, which Ollama sometimes echoes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"/(no_)?think", "", text, flags=re.I)
    return text.strip()
