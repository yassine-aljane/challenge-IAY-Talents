"""
Thin wrapper around the Groq chat-completions API.

Centralizing this here means every agent goes through the same client
construction (fails fast with a clear error if GROQ_API_KEY is missing,
rather than a confusing 401 deep inside an agent) and the same low
temperature default for the structured-extraction/scoring calls.
"""

from __future__ import annotations

import json
import os

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your free Groq API key "
                "(https://console.groq.com/keys)."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
    """Call the LLM and parse its reply as a JSON object.

    Uses Groq's JSON mode (response_format=json_object) so the model is
    constrained to emit a single JSON object; we still defensively re-parse
    and let callers validate the result against a Pydantic schema.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}") from e


def chat_text(system_prompt: str, user_prompt: str, max_tokens: int = 900, temperature: float = 0.4) -> str:
    """Call the LLM and return its free-text reply (used for cover letters)."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
