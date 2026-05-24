"""
Minimal LLM client for hemorrhage case inference.

Uses existing ``requests`` dependency and environment variables only.
No ``src.models`` imports.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import requests

from src.tasks.hemorrhage.inference.parse import extract_first_json_object

LOGGER = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("usz_api", "ollama")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "usz_api").strip().lower()
USZ_LLM_URL = os.getenv("USZ_LLM_URL", "http://localhost:8100/generate")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11500")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
TIMEOUT = int(os.getenv("LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "120")))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _extract_system_user(messages: list) -> Tuple[str, str]:
    sys_parts: List[str] = []
    user_parts: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            sys_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    return "\n\n".join(p for p in sys_parts if p), "\n\n".join(p for p in user_parts if p)


def _build_chat_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/api/chat"):
        return clean
    if clean.endswith("/api/generate"):
        return f"{clean[:-len('/api/generate')]}/api/chat"
    if clean.endswith("/api"):
        return f"{clean}/chat"
    return f"{clean}/api/chat"


def call_usz_api(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "prompt": (user_prompt or "").strip(),
        "system_prompt": system_prompt or "",
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
        "disable_think": os.getenv("LLM_DISABLE_THINK", "").strip().lower() in ("1", "true", "yes"),
    }
    response = requests.post(USZ_LLM_URL, json=payload, timeout=TIMEOUT)
    if response.status_code != 200:
        snippet = (response.text or "")[:500]
        raise RuntimeError(f"USZ LLM API HTTP {response.status_code}: {snippet}")
    body = response.json()
    result = body.get("response", "")
    if isinstance(result, list):
        final_text = "\n".join(str(x) for x in result)
    else:
        final_text = str(result)
    return extract_first_json_object(final_text.strip())


def _call_ollama_messages(messages: list) -> str:
    chat_url = _build_chat_url(OLLAMA_URL)
    response = requests.post(
        chat_url,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "num_predict": LLM_MAX_TOKENS,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response missing message.content")
    return content.strip()


def call_llm(messages: list) -> str:
    """Provider-agnostic entry point for hemorrhage inference."""
    system_prompt, user_prompt = _extract_system_user(messages)

    if LLM_PROVIDER == "ollama":
        return _call_ollama_messages(messages)

    if LLM_PROVIDER == "usz_api":
        return call_usz_api(system_prompt, user_prompt)

    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}; allowed: {SUPPORTED_PROVIDERS}")
