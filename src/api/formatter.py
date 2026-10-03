"""Optional repair of a broken tool-call reply by a small OpenRouter model.

Off unless OPENROUTER_API_KEY is set. Used only when ChatGPT's reply contains
"tool_calls" but does not parse (stray prose, bad escapes), so a cheap model can fix
the JSON without another slow, rate-limited ChatGPT round trip. Only the broken reply
and the tool schemas are sent, never the conversation: free-tier models may log prompts.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from src.log import setup_logging

log = setup_logging("formatter")

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_SYSTEM = (
    "You repair broken JSON. The user message holds a model reply that was meant to be "
    'exactly one JSON object of the form {"tool_calls":[{"name":"<function>","arguments":{...}}]}, '
    "plus the allowed function schemas. Output ONLY the repaired JSON object: no Markdown, no "
    "commentary. Keep every name and argument value as written; do not invent calls or "
    'arguments. If it cannot be repaired into that form, output {"tool_calls":[]}.'
)


def enabled() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def _request(body: dict) -> str:
    request = urllib.request.Request(
        _ENDPOINT,
        json.dumps(body).encode(),
        {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY'].strip()}",
            "Content-Type": "application/json",
            "X-Title": "CatGPT Gateway formatter",
        },
    )
    timeout = float(os.getenv("CATGPT_FORMATTER_TIMEOUT", "30"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"] or ""


async def repair_tool_call(reply: str, tools) -> str | None:
    """Return the model's repaired JSON text, or None when disabled or unavailable."""
    if not enabled() or not (reply or "").strip():
        return None
    schemas = json.dumps(
        [{"name": t.function.name, "parameters": t.function.parameters} for t in tools]
    )
    body = {
        "model": os.getenv("CATGPT_FORMATTER_MODEL", "openrouter/free"),
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Allowed functions:\n{schemas}\n\nReply to repair:\n{reply[:8000]}"},
        ],
    }
    try:
        return await asyncio.to_thread(_request, body)
    except Exception as e:  # network, free-tier 429, unexpected payload
        log.warning(f"Formatter unavailable ({type(e).__name__}): {e}")
        return None
