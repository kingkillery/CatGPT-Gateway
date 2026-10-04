"""Bridge a request to another model through OpenRouter instead of ChatGPT.

An agent loop costs one ChatGPT message per step, and ChatGPT rate-limits the session
(HTTP 429, which shows up here as an empty reply). With a bridge model configured, the
turns that follow a tool result are answered by that model, so ChatGPT only sees the
opening turn of a loop; and any request ChatGPT fails to answer falls back to it.

Off unless OPENROUTER_API_KEY and CATGPT_BRIDGE_MODEL are both set. The model is
whatever OpenRouter id you choose: a specific model, or ``openrouter/free`` for the free
router (the request asks OpenRouter to pick only providers that support tools).

PRIVACY: unlike the formatter, the bridge sends the real conversation, tool results (file
contents, command output) included, to OpenRouter and the model behind it. Free-tier
models may log prompts and train on them. Only enable it for work you are happy to send.
Nothing is sent when it is off, and request content is never logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
import uuid

from src.api import cleanup
from src.api.openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    FunctionCallInfo,
    ToolCall,
    UsageInfo,
)
from src.log import setup_logging

log = setup_logging("bridge")

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def model() -> str:
    return os.getenv("CATGPT_BRIDGE_MODEL", "").strip()


def enabled() -> bool:
    return bool(model() and os.getenv("OPENROUTER_API_KEY", "").strip())


def route() -> str:
    """"continuation" (default): serve turns after a tool result here. "fallback": only
    serve requests ChatGPT failed to answer."""
    value = os.getenv("CATGPT_BRIDGE_ROUTE", "continuation").strip().lower()
    return value if value in ("continuation", "fallback") else "continuation"


def routes_to_bridge(request: ChatCompletionRequest) -> bool:
    """True for a mid-loop turn: the model must now react to a tool result."""
    return (
        enabled()
        and route() == "continuation"
        and bool(request.tools)
        and request.tool_choice != "none"
        and bool(request.messages)
        and request.messages[-1].role == "tool"
    )


def _payload(request: ChatCompletionRequest) -> dict:
    messages = []
    for m in request.messages:
        d = m.model_dump(exclude_none=True)
        if m.role == "assistant" and "content" not in d:
            d["content"] = None  # an assistant turn that only called tools
        messages.append(d)
    body: dict = {"model": model(), "messages": messages, "stream": False}
    if request.tools:
        body["tools"] = [t.model_dump() for t in request.tools]
        body["tool_choice"] = request.tool_choice or "auto"
        body["provider"] = {"require_parameters": True}  # only providers that honor `tools`
    for field in ("temperature", "max_tokens", "top_p"):
        if getattr(request, field) is not None:
            body[field] = getattr(request, field)
    return body


def _post(body: dict) -> dict:
    http = urllib.request.Request(
        _ENDPOINT,
        json.dumps(body).encode(),
        {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY'].strip()}",
            "Content-Type": "application/json",
            "X-Title": "CatGPT Gateway bridge",
        },
    )
    with urllib.request.urlopen(http, timeout=float(os.getenv("CATGPT_BRIDGE_TIMEOUT", "120"))) as response:
        return json.loads(response.read())


def _arguments(raw) -> str:
    """Tool arguments as a JSON string. Free models sometimes send slightly broken JSON."""
    if not isinstance(raw, str):
        return json.dumps(raw if raw is not None else {})
    try:
        json.loads(raw)
        return raw
    except ValueError:
        return cleanup.repair_json(raw) or raw


async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse | None:
    """Answer the request with the bridge model, or None if it cannot (caller uses ChatGPT)."""
    if not enabled():
        return None
    try:
        data = await asyncio.to_thread(_post, _payload(request))
        message = data["choices"][0]["message"]
    except Exception as e:  # network, 429 on the free tier, unexpected payload
        log.warning(f"Bridge unavailable ({type(e).__name__}): {e}")
        return None

    offered = {t.function.name for t in request.tools or []}
    calls = [
        ToolCall(
            id=c.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            function=FunctionCallInfo(name=c["function"]["name"], arguments=_arguments(c["function"].get("arguments"))),
        )
        for c in message.get("tool_calls") or []
        if isinstance(c, dict) and isinstance(c.get("function"), dict) and c["function"].get("name") in offered
    ]
    content = (message.get("content") or "").strip()
    if not calls and not content:
        log.warning("Bridge model returned nothing usable")
        return None

    usage = data.get("usage") or {}
    log.info(f"Bridge served this request ({model()}); ChatGPT was not used")
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            Choice(
                message=ChoiceMessage(content=None if calls else content, tool_calls=calls or None),
                finish_reason="tool_calls" if calls else "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        ),
    )
