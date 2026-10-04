"""Bridge a request to another model instead of ChatGPT.

An agent loop costs one ChatGPT message per step, and ChatGPT rate-limits the session
(HTTP 429, which shows up here as an empty reply). With a bridge model configured, the
turns that follow a tool result are answered by that model, so ChatGPT only sees the
opening turn of a loop; and a request ChatGPT fails to answer (an empty reply, or a
made-up "that path does not exist" denial) is handed to it too.

CATGPT_BRIDGE_MODEL is one model or a comma-separated list tried in order; the first
usable answer wins and, if none works, the request goes to ChatGPT as usual. Each entry
picks its backend by prefix:

    stealth/space-bunny-alpha            OpenRouter (OPENROUTER_API_KEY)
    huggingface/deepseek-ai/DeepSeek-V4.1-Flash
                                         Hugging Face inference router (HF_TOKEN)

An entry whose key is not set is skipped. The OpenRouter free router is
``openrouter/free``; a specific id is steadier.

PRIVACY: unlike the formatter, the bridge sends the real conversation, tool results (file
contents, command output) included, to the backend and the provider behind it. Free and
stealth models may log prompts and train on them. Only enable it for work you are happy
to send. Nothing is sent when it is off, and request content is never logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
import uuid
from typing import NamedTuple

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

_HF_PREFIX = "huggingface/"


class Target(NamedTuple):
    spec: str  # as configured
    url: str
    key_env: str
    model: str  # the id the backend expects
    openrouter: bool


def _target(spec: str) -> Target:
    if spec.startswith(_HF_PREFIX):
        return Target(spec, "https://router.huggingface.co/v1/chat/completions", "HF_TOKEN", spec[len(_HF_PREFIX):], False)
    return Target(spec, "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", spec, True)


def models() -> list[str]:
    """The configured models, in order (blank entries ignored)."""
    return [m.strip() for m in os.getenv("CATGPT_BRIDGE_MODEL", "").split(",") if m.strip()]


def targets() -> list[Target]:
    """Configured models whose backend key is present."""
    return [t for t in map(_target, models()) if os.getenv(t.key_env, "").strip()]


def model() -> str:
    """The primary configured model ("" when none)."""
    return (models() or [""])[0]


def enabled() -> bool:
    return bool(targets())


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


def _payload(request: ChatCompletionRequest, target: Target) -> dict:
    messages = []
    for m in request.messages:
        d = m.model_dump(exclude_none=True)
        if m.role == "assistant" and "content" not in d:
            d["content"] = None  # an assistant turn that only called tools
        messages.append(d)
    body: dict = {"model": target.model, "messages": messages, "stream": False}
    if request.tools:
        body["tools"] = [t.model_dump() for t in request.tools]
        body["tool_choice"] = request.tool_choice or "auto"
        if target.openrouter:
            body["provider"] = {"require_parameters": True}  # only providers that honor `tools`
    for field in ("temperature", "max_tokens", "top_p"):
        if getattr(request, field) is not None:
            body[field] = getattr(request, field)
    return body


def _post(target: Target, body: dict) -> dict:
    http = urllib.request.Request(
        target.url,
        json.dumps(body).encode(),
        {
            "Authorization": f"Bearer {os.environ[target.key_env].strip()}",
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


def _answer(request: ChatCompletionRequest, data: dict) -> ChatCompletionResponse | None:
    """Map one backend reply onto our response, or None if it holds nothing usable."""
    message = data["choices"][0]["message"]
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
        return None
    usage = data.get("usage") or {}
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


async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse | None:
    """Answer with the first configured model that works, or None (caller uses ChatGPT)."""
    for target in targets():
        try:
            data = await asyncio.to_thread(_post, target, _payload(request, target))
            response = _answer(request, data)
        except Exception as e:  # network, 402/429, unexpected payload: try the next model
            log.warning(f"Bridge model {target.spec} unavailable ({type(e).__name__}): {e}")
            continue
        if response is None:
            log.warning(f"Bridge model {target.spec} returned nothing usable")
            continue
        served_by = data.get("model") if isinstance(data.get("model"), str) else target.spec
        log.info(f"Bridge served this request ({served_by}); ChatGPT was not used")
        return response
    return None
