"""
OpenAI-compatible API routes.

Provides:
  POST /v1/chat/completions   — chat completions (with tool/function calling)
  GET  /v1/models             — list available models

All requests are serialized through an asyncio.Lock because the underlying
Playwright browser page is single-threaded.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    FunctionCallInfo,
    FunctionDefinition,
    ImageData,
    ImageGenerationRequest,
    ImagesResponse,
    ModelListResponse,
    ModelObject,
    ResponseFunctionCall,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponseUsage,
    ToolCall,
    ToolDefinition,
    UsageInfo,
)
from src.chatgpt.client import ChatGPTClient
from src.claude.client import ClaudeClient
from src.config import Config
from src.log import setup_logging

log = setup_logging("openai_routes")

openai_router = APIRouter()

# Global reference — set by server.py at startup
_client: ChatGPTClient | ClaudeClient | None = None

# Serialize all requests — single browser page, not thread-safe.
# Created lazily to avoid Python 3.9 event-loop binding issues.
_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Get or create the global request lock (lazy init)."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# Track messages in the current thread to prevent thread exhaustion
_thread_message_count = 0
_MAX_THREAD_MESSAGES = 8  # Start a new chat after this many requests
_last_response_time: float = 0.0
_MIN_MESSAGE_GAP = 3.0  # Minimum seconds between messages (ChatGPT needs cooldown)


async def _ensure_fresh_chat() -> None:
    """Enforce cooldown and ensure the next message is sent in a fresh chat.

    For ChatGPT we keep every conversation in a TEMPORARY chat
    (?temporary-chat=true) so it is never saved to the sidebar history: a new
    temporary chat is started whenever the page is not already on one, or when
    the current thread has accumulated enough messages that ChatGPT's UI starts
    to degrade (~6-8). Conversation context is flattened into each prompt, so a
    fresh chat never loses memory. Claude has no temporary-chat mode, so it
    keeps the count-based rotation only.

    Also enforces a minimum gap between consecutive messages, since the UI may
    not accept rapid-fire messages properly.
    """
    global _thread_message_count, _last_response_time

    # Enforce minimum gap between messages
    if _last_response_time > 0:
        elapsed = time.time() - _last_response_time
        if elapsed < _MIN_MESSAGE_GAP:
            wait = _MIN_MESSAGE_GAP - elapsed
            log.debug(f"Cooldown: waiting {wait:.1f}s before next message")
            await asyncio.sleep(wait)

    client = _get_client()

    # Decide whether to (re)start a chat.
    needs_new = _thread_message_count >= _MAX_THREAD_MESSAGES
    if Config.PROVIDER == "chatgpt":
        # ChatGPT: require a temporary chat so nothing is saved to history.
        try:
            on_temp_chat = "temporary-chat=true" in (client.page.url or "")
        except Exception:
            on_temp_chat = False
        if not on_temp_chat:
            needs_new = True

    if not needs_new:
        return  # Already on a usable (temporary) chat — no navigation needed

    try:
        await client.new_chat()
        _thread_message_count = 0
    except Exception as e:
        log.warning(f"new_chat() failed, retrying once: {e}")
        try:
            await asyncio.sleep(2)
            await client.new_chat()
            _thread_message_count = 0
        except Exception as e2:
            log.error(f"new_chat() retry also failed: {e2}")
            # Don't raise — continue with current thread rather than failing
            log.warning("Continuing with current thread despite new_chat failure")


def _increment_thread_count() -> None:
    """Increment the thread message counter after a successful response."""
    global _thread_message_count, _last_response_time
    _thread_message_count += 1
    _last_response_time = time.time()
    log.debug(f"Thread message count: {_thread_message_count}/{_MAX_THREAD_MESSAGES}")


def _get_model_id() -> str:
    """Return the configured OpenAI-compatible model ID."""
    if Config.DEFAULT_MODEL:
        return Config.DEFAULT_MODEL
    if Config.PROVIDER == "claude":
        return "claude-browser"
    return "catgpt-browser"


def set_openai_client(client: ChatGPTClient | ClaudeClient) -> None:
    """Called by server.py to inject the client."""
    global _client
    _client = client


def _get_client() -> ChatGPTClient | ClaudeClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    return _client


# ── Helpers ─────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def _extract_content_text(content) -> str:
    """Extract text from message content (handles both string and list format)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else ""
    return str(content)


def _extract_image_urls(content) -> list[str]:
    """Extract image URLs from message content (OpenAI vision format)."""
    if not isinstance(content, list):
        return []
    urls = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = str(image_url)
            if url:
                urls.append(url)
    return urls


def _extract_file_attachments(content) -> list[dict]:
    """
    Extract file attachments from message content.

    Supported content part format:
      {"type": "file", "file": {"filename": "test.pdf", "data": "base64...", "mime_type": "application/pdf"}}

    Also supports a shorthand data-URL style:
      {"type": "file", "file": {"filename": "test.pdf", "url": "data:application/pdf;base64,..."}}

    Returns list of dicts: [{"filename": str, "data_b64": str, "mime_type": str}, ...]
    """
    if not isinstance(content, list):
        return []
    files = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        file_info = item.get("file", {})
        if not isinstance(file_info, dict):
            continue
        # Reference to a previously uploaded file (Files API): resolve id -> path.
        file_id = file_info.get("file_id")
        if file_id:
            from src.api.files_routes import file_store
            local_path = file_store.path_for(file_id)
            if local_path:
                files.append({"local_path": local_path})
            else:
                log.warning(f"Referenced file_id not found: {file_id}")
            continue
        filename = file_info.get("filename", "attachment")
        # Two ways to supply file data:
        # 1. data + mime_type  2. url (data-URL)
        data_b64 = file_info.get("data")
        mime_type = file_info.get("mime_type", "application/octet-stream")
        url = file_info.get("url", "")
        if not data_b64 and url.startswith("data:"):
            # Parse data URL
            try:
                header, data_b64 = url.split(",", 1)
                # header = "data:application/pdf;base64"
                if ":" in header and ";" in header:
                    mime_type = header.split(":")[1].split(";")[0]
            except ValueError:
                continue
        if data_b64:
            files.append({"filename": filename, "data_b64": data_b64, "mime_type": mime_type})
    return files


async def _download_file(url_or_data: str | dict, download_dir: str = "/tmp/catgpt_files") -> str | None:
    """
    Download / decode a file (image, PDF, etc.) from URL, base64 data URL,
    or a file attachment dict. Returns the local file path.
    """
    import base64
    import hashlib
    import os

    os.makedirs(download_dir, exist_ok=True)

    # ── Dict form (from _extract_file_attachments) ──
    if isinstance(url_or_data, dict):
        # Pre-resolved local path (e.g. a Files API file_id reference).
        local_path = url_or_data.get("local_path")
        if local_path:
            return local_path if os.path.isfile(local_path) else None
        try:
            filename = url_or_data.get("filename", "file")
            data_b64 = url_or_data["data_b64"]
            # Sanitize filename
            safe_name = re.sub(r"[^\w.\-]", "_", filename)
            hash_suffix = hashlib.md5(data_b64[:60].encode()).hexdigest()[:8]
            filepath = os.path.join(download_dir, f"{hash_suffix}_{safe_name}")
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(data_b64))
            log.info(f"Decoded file attachment: {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Failed to decode file attachment: {e}")
            return None

    # ── String forms ──
    url = str(url_or_data)

    if url.startswith("data:"):
        # Base64 data URL: data:image/png;base64,iVBOR... or data:application/pdf;base64,...
        try:
            header, b64data = url.split(",", 1)
            # Detect extension from MIME type
            ext = "bin"
            mime = ""
            if ":" in header and ";" in header:
                mime = header.split(":")[1].split(";")[0]
            ext_map = {
                "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp",
                "image/gif": "gif", "application/pdf": "pdf",
                "text/plain": "txt", "text/csv": "csv",
                "application/json": "json",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            }
            ext = ext_map.get(mime, mime.split("/")[-1] if "/" in mime else "bin")
            filename = f"file_{hashlib.md5(b64data[:100].encode()).hexdigest()[:12]}.{ext}"
            filepath = os.path.join(download_dir, filename)
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(b64data))
            log.info(f"Decoded base64 file: {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Failed to decode base64 data URL: {e}")
            return None
    elif url.startswith(("http://", "https://")):
        # HTTP URL — download it
        try:
            import urllib.request
            ext = "bin"
            for e in ["jpg", "jpeg", "webp", "gif", "png", "pdf", "txt", "csv", "docx", "xlsx"]:
                if e in url.lower():
                    ext = e
                    break
            filename = f"file_{hashlib.md5(url.encode()).hexdigest()[:12]}.{ext}"
            filepath = os.path.join(download_dir, filename)
            urllib.request.urlretrieve(url, filepath)
            log.info(f"Downloaded file: {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Failed to download file from {url}: {e}")
            return None
    elif os.path.isfile(url):
        # Local file path
        return url
    else:
        log.warning(f"Unknown file URL format: {url[:80]}")
        return None


def _build_prompt(messages: list[ChatMessage]) -> str:
    """
    Flatten an OpenAI-style message array into a single prompt string
    that we can paste into ChatGPT's input box.

    The browser already maintains conversation context within a thread,
    so for simple single-turn calls we just send the last user message.
    For multi-turn with system prompts or tool results, we build a
    formatted transcript.
    """
    # Simple case: only one user message (and optionally one system message)
    non_system = [m for m in messages if m.role != "system"]
    system_msgs = [m for m in messages if m.role == "system"]

    # If it's just one user message, send it directly
    if len(non_system) == 1 and non_system[0].role == "user":
        prefix = ""
        if system_msgs:
            sys_text = _extract_content_text(system_msgs[0].content)
            if Config.PROVIDER == "claude":
                # Claude rejects "[System instruction: ...]" as prompt injection.
                # Present it as context instead.
                prefix = f"{sys_text}\n\n"
            else:
                prefix = f"[System instruction: {sys_text}]\n\n"
        user_text = _extract_content_text(non_system[0].content)
        return prefix + (user_text or "")

    # Multi-turn: build a transcript
    parts: list[str] = []
    for msg in messages:
        role = msg.role.capitalize()
        if msg.role == "system":
            if Config.PROVIDER == "claude":
                # For Claude, present system messages as context without the label
                text = _extract_content_text(msg.content)
                if text:
                    parts.append(text)
            else:
                text = _extract_content_text(msg.content)
                if text:
                    parts.append(f"System: {text}")
        elif msg.role == "tool":
            # Tool result — include both the call context and the result
            tool_content = _extract_content_text(msg.content)
            if Config.PROVIDER == "claude":
                parts.append(
                    f"The tool was executed and returned this result:\n{tool_content}\n\n"
                    f"Now use the result above to answer the user's original question in plain text."
                )
            else:
                parts.append(
                    f"[Tool result for {msg.tool_call_id or 'unknown'}]: {tool_content}\n\n"
                    f"Use the tool result to answer the user. Do NOT call tools again."
                )
        elif msg.role == "assistant" and msg.tool_calls:
            # Assistant requested tool calls — show what was called
            calls_desc = []
            for tc in msg.tool_calls:
                calls_desc.append(
                    f'{tc.function.name}({tc.function.arguments})'
                )
            parts.append(f"Assistant called tools: {', '.join(calls_desc)}")
        elif msg.content:
            text = _extract_content_text(msg.content)
            if text:
                parts.append(f"{role}: {text}")

    return "\n\n".join(parts)


def _build_tool_system_prompt(
    tools: list[ToolDefinition],
    tool_choice: str | dict | None = None,
) -> str:
    """
    Build a system-level instruction that tells the model about available tools.

    *tool_choice* controls how insistent the instructions are:
      - "auto" / None  — model decides whether to call a tool or answer directly
      - "required"     — model MUST call at least one tool
      - "none"         — caller should not call this function at all
      - {"type":"function","function":{"name":"X"}} — model MUST call that tool
    """
    tool_descriptions = []
    for tool in tools:
        fn = tool.function
        desc = {
            "name": fn.name,
            "description": fn.description,
            "parameters": fn.parameters,
        }
        tool_descriptions.append(json.dumps(desc, indent=2))

    tools_json = "\n---\n".join(tool_descriptions)

    # ── Determine the decision instruction based on tool_choice ──
    forced_tool_name = None
    if isinstance(tool_choice, dict):
        # {"type": "function", "function": {"name": "X"}}
        forced_tool_name = (
            tool_choice.get("function", {}).get("name")
            if isinstance(tool_choice.get("function"), dict)
            else None
        )

    if forced_tool_name:
        decision = (
            f"You MUST call the function `{forced_tool_name}`. "
            f"Do NOT answer the question yourself — output only the JSON tool call."
        )
    elif tool_choice == "required":
        decision = (
            "You MUST call at least one of the available functions. "
            "Do NOT answer the question yourself — always output tool calls."
        )
    else:
        # "auto" or None — model decides, but prefer the caller's external
        # functions over the provider UI's built-in browsing/tools/memory.
        decision = (
            "Treat the available functions as your only external tools. "
            "If the user's request can be fulfilled or materially assisted by "
            "one or more available functions — including current/exact data, "
            "file or code inspection, code execution, browser/web actions, "
            "or side effects — call the appropriate tool(s). "
            "Do not answer from memory when a relevant function can do the work. "
            "Answer normally in plain text only when no function is relevant."
        )

    # ── Provider-specific prompt framing ──
    if Config.PROVIDER == "claude":
        return f"""You have access to external tools through a structured interface. {decision}

When calling tools, respond with ONLY a JSON code block — no text before or after it:

```json
{{"tool_calls": [{{"name": "<function_name>", "arguments": {{...}}}}]}}
```

Rules:
1. Output ONLY the JSON code block when calling tools. Do not add any commentary, explanation, or text outside the code block.
2. You may call multiple functions in one response by adding them to the array.
3. Use the exact parameter names and types shown in each function's schema.
4. When you receive tool results in a follow-up message, use them to give the user a natural, helpful answer. Do NOT output another JSON tool call for the same request.

Available functions:
{tools_json}

Example — single tool:
```json
{{"tool_calls": [{{"name": "get_current_time", "arguments": {{}}}}]}}
```

Example — multiple tools:
```json
{{"tool_calls": [{{"name": "weather_forecast", "arguments": {{"city": "Tokyo", "date": "today"}}}}, {{"name": "calculate_expression", "arguments": {{"expression": "2+2"}}}}]}}
```
"""
    else:
        return f"""You are in tool-calling mode. {decision}

When calling tools, output ONLY a JSON code block — no other text:

```json
{{"tool_calls": [{{"name": "<function_name>", "arguments": {{...}}}}]}}
```

Rules:
1. Output ONLY the JSON code block when calling tools. No explanation, no text before or after.
2. You may call multiple functions in one response by adding them to the array.
3. Use the exact parameter names and types from each function's schema.
4. When a follow-up message contains tool results, summarize them naturally for the user. Do NOT call tools again for the same request.
5. Do not use native ChatGPT browsing, search, connectors, uploads, or code interpreter for toolable work — use this JSON tool-call protocol.
6. Do not refuse or say tools are unavailable — they are available through this interface.

Available functions:
{tools_json}

Example — single tool:
```json
{{"tool_calls": [{{"name": "get_current_time", "arguments": {{}}}}]}}
```

Example — multiple tools:
```json
{{"tool_calls": [{{"name": "weather_forecast", "arguments": {{"city": "Tokyo", "date": "today"}}}}, {{"name": "calculate_expression", "arguments": {{"expression": "2+2"}}}}]}}
```
"""


def _forced_tool_name_from_choice(tool_choice: str | dict | None) -> str | None:
    """Return the forced function name from OpenAI tool_choice, if any."""
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


_TOOL_REFUSAL_MARKERS = (
    "fake tool call",
    "fabricate a tool call",
    "fabricate a json tool call",
    "fabricated tool call",
    "can't produce a fake",
    "cannot produce a fake",
    "can't call a bash",
    "isn't actually available",
    "is not actually available",
    "isn't available in this chat",
    "doesn't exist in this chat",
    "does not actually exist",
    "no such tool",
)


def _looks_like_tool_refusal(text: str | None) -> bool:
    """True if the model meta-refused to emit a tool call (a "bad call")."""
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in _TOOL_REFUSAL_MARKERS)


def _build_bad_call_correction(forced_name: str | None, forced: bool) -> str:
    """Correction nudge sent after a bad tool call (refusal or malformed JSON)."""
    if forced:
        skeleton = (
            f'{{"tool_calls":[{{"name":"{forced_name}","arguments":{{}}}}]}}'
            if forced_name
            else '{"tool_calls":[{"name":"<an available name>","arguments":{}}]}'
        )
        return (
            "Your previous response was not a valid JSON object. Ignore it and "
            f"reply with ONLY the corrected raw JSON object, exactly this shape: {skeleton}. "
            "Escape every double-quote inside string values, prefer single quotes "
            "inside shell commands, and output no prose and no code fence."
        )
    return (
        "The functions listed above are real tools provided by the caller, who "
        "executes them and returns results — never say they are unavailable. If "
        "one helps with the request, respond with ONLY a raw JSON object of this "
        'shape: {"tool_calls":[{"name":"<function name>","arguments":{}}]} '
        "(escape double-quotes inside strings, single quotes in shell commands, "
        "no code fence). If none are needed, answer the request directly in plain "
        "prose with no remarks about tools or availability."
    )


def _build_tool_protocol_suffix(tool_choice: str | dict | None = None) -> str:
    """Build a final reminder appended after the user's request on tool turns."""
    forced_tool_name = _forced_tool_name_from_choice(tool_choice)
    if forced_tool_name:
        return f"""FINAL RESPONSE FORMAT:
Respond with ONLY a raw JSON object — no Markdown, no prose, no language label — using exactly this schema:

{{"tool_calls":[{{"name":"{forced_tool_name}","arguments":{{}}}}]}}

- `arguments` is a JSON object populated with real values from the request and the schema for `{forced_tool_name}`.
- No placeholders, ellipsis, comments, or trailing commas.
- Derive the answer solely from the request above; do not use built-in web search or other assistants."""

    if tool_choice == "required":
        return """FINAL RESPONSE FORMAT:
Respond with ONLY a raw JSON object — no Markdown, no prose, no language label — containing a single `tool_calls` array.
Each array item is an object with a `name` field (one of the available names listed above, never empty) and an `arguments` field (a JSON object of real values matching that name's schema).
Escape every double-quote that appears inside a string value, prefer single quotes inside shell commands, and include no placeholders, ellipsis, comments, or trailing commas.
Derive the answer solely from the request above; do not use built-in web search or other assistants."""

    return """FINAL TOOL-CALL PROTOCOL REMINDER:
The available functions above are the caller's external CLI tools.
If any function is relevant to the user's last request, output ONLY a JSON tool-call code block.
Use plain text only when no function is relevant.
Do not use native ChatGPT browsing/search/tools instead of these functions."""


_VALID_JSON_ESCAPE = frozenset('"\\/' + "bfnrtu")


def _repair_json_escapes(text: str) -> str:
    """Drop backslashes that introduce invalid JSON escapes (e.g. shell `\\'`).

    Runs only as a fallback when strict `json.loads` fails, on already-extracted
    tool-call candidates. The walk tracks JSON string context and consumes each
    backslash as part of its escape pair, so legitimate `\\\\` and `\\"` survive
    while stray `\\'`, `\\d`, etc. (common from shell commands) normalize to the
    following character. Valid JSON passes through byte-for-byte.
    """
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if not in_string:
            if c == '"':
                in_string = True
            out.append(c)
            i += 1
            continue
        if c != "\\":
            if c == '"':
                in_string = False
            out.append(c)
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt in _VALID_JSON_ESCAPE:
            out.append(c)
            out.append(nxt)
        else:
            out.append(nxt)
        i += 2
    return "".join(out)


def _json_loads_tolerant(candidate: str) -> tuple[dict, str] | None:
    """Strict JSON load with a one-shot invalid-escape repair fallback.

    Returns (parsed_object, string_that_parsed) or None. The returned string is
    the repaired candidate when repair was needed, so downstream re-serialization
    stays consistent.
    """
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _repair_json_escapes(candidate)
        if repaired == candidate:
            return None
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None
        return (parsed, repaired) if isinstance(parsed, dict) else None
    return (parsed, candidate) if isinstance(parsed, dict) else None


def _extract_json_object(text: str, anchor: str = "tool_calls") -> str | None:
    """
    Extract a JSON object containing *anchor* key from *text*.

    Uses two strategies:
      1. Look inside markdown code blocks (```json ... ```)
      2. Find the anchor key and walk outward using brace-depth tracking
         to handle arbitrarily nested JSON (arrays, nested objects, etc.)
    """
    # Strategy 1: code blocks — most reliable when the model obeys the prompt
    for m in re.finditer(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", text):
        candidate = m.group(1).strip()
        if anchor in candidate:
            loaded = _json_loads_tolerant(candidate)
            if loaded and anchor in loaded[0]:
                return loaded[1]

    # Strategy 2: locate anchor, walk to balanced braces
    search_key = f'"{anchor}"'
    idx = text.find(search_key)
    if idx == -1:
        return None

    # Walk backward to the nearest '{'
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None

    # Walk forward tracking brace depth, respecting JSON string literals
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2          # skip escaped char
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    loaded = _json_loads_tolerant(candidate)
                    return loaded[1] if loaded else None
        i += 1

    return None


def _parse_tool_calls(
    response_text: str, tools: list[ToolDefinition]
) -> list[ToolCall] | None:
    """
    Try to parse tool calls from the model's response text.

    Uses robust brace-matching extraction (handles nested JSON, arrays, etc.)
    then validates tool names against the provided tool definitions.
    Returns None if no valid tool calls are found.
    """
    json_str = _extract_json_object(response_text, "tool_calls")
    if not json_str:
        return None

    loaded = _json_loads_tolerant(json_str)
    if not loaded:
        log.debug(f"Failed to parse tool call JSON: {json_str[:200]}")
        return None
    parsed = loaded[0]

    if "tool_calls" not in parsed or not isinstance(parsed["tool_calls"], list):
        return None

    # Validate that the called functions are in the provided tools
    valid_names = {t.function.name for t in tools}
    result: list[ToolCall] = []

    for call in parsed["tool_calls"]:
        if not isinstance(call, dict):
            log.warning(f"Skipping malformed tool call: {call!r}")
            continue

        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name", "")
            arguments = function.get("arguments", {})
        else:
            name = call.get("name", "")
            arguments = call.get("arguments", {})

        if name not in valid_names:
            log.warning(f"Model called unknown tool: {name}")
            continue

        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments)

        result.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=FunctionCallInfo(name=name, arguments=arguments_str),
            )
        )

    return result if result else None


# ── Routes ──────────────────────────────────────────────────────


@openai_router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available models — returns our single browser-backed model."""
    model_id = _get_model_id()
    owned_by = "anthropic" if Config.PROVIDER == "claude" else "catgpt"
    return ModelListResponse(
        data=[
            ModelObject(id=model_id, owned_by=owned_by),
        ]
    )


@openai_router.post("/v1/images/generations", response_model=ImagesResponse)
async def create_image(
    request: ImageGenerationRequest,
) -> ImagesResponse:
    """
    OpenAI-compatible image generation endpoint.

    Sends the prompt to ChatGPT which uses DALL-E to generate images.
    Downloads the generated images and returns them in OpenAI format.
    Supports response_format='b64_json' (default) or 'url' (local file path).
    """
    import base64

    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    # Claude does not support image generation
    if Config.PROVIDER == "claude":
        raise HTTPException(
            status_code=501,
            detail="Image generation is not supported by Claude. This feature is only available with the ChatGPT provider.",
        )

    client = _get_client()

    async with _get_lock():
        start_time = time.time()

        # Build an image-generation prompt.
        # n > 1: we ask ChatGPT to generate multiple images
        # size/quality/style hints are included but ChatGPT web may ignore them.
        prompt_parts = [f"Generate an image: {request.prompt}"]
        if request.n and request.n > 1:
            prompt_parts.append(f"Please generate {request.n} different images.")
        if request.size and request.size != "1024x1024":
            prompt_parts.append(f"Image size: {request.size}.")
        if request.quality == "hd":
            prompt_parts.append("Make it high-definition / highly detailed.")
        if request.style == "natural":
            prompt_parts.append("Use a natural, realistic style.")

        full_prompt = " ".join(prompt_parts)

        log.info(
            f"POST /v1/images/generations — prompt='{request.prompt[:80]}', "
            f"n={request.n}, size={request.size}, response_format={request.response_format}"
        )

        # Start a fresh conversation to avoid thread exhaustion
        await _ensure_fresh_chat()

        # Send to ChatGPT
        try:
            result = await client.send_message(full_prompt)
        except Exception as e:
            log.error(f"Provider error during image generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Provider error: {str(e)}")

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Check if ChatGPT generated images
        if not result.images:
            # ChatGPT may have responded with text instead of generating an image.
            # This can happen when the model declines or gives a text description.
            log.warning(
                f"No images detected in response ({elapsed_ms}ms). "
                f"ChatGPT replied: {result.message[:200]}"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ChatGPT did not generate an image. "
                    f"Model response: {result.message[:500]}"
                ),
            )

        # Build image data objects
        image_data_list: list[ImageData] = []
        for img_info in result.images:
            revised_prompt = img_info.prompt_title or img_info.alt or request.prompt

            if request.response_format == "b64_json":
                # Read the downloaded file and base64-encode it
                if img_info.local_path:
                    try:
                        with open(img_info.local_path, "rb") as f:
                            img_bytes = f.read()
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        image_data_list.append(
                            ImageData(
                                b64_json=b64,
                                revised_prompt=revised_prompt,
                            )
                        )
                    except Exception as e:
                        log.error(f"Failed to read image file {img_info.local_path}: {e}")
                else:
                    log.warning(f"Image has no local_path: {img_info.url[:80]}")
            else:
                # response_format == "url" → return local file path as URL
                image_data_list.append(
                    ImageData(
                        url=img_info.local_path or img_info.url,
                        revised_prompt=revised_prompt,
                    )
                )

        if not image_data_list:
            raise HTTPException(
                status_code=500,
                detail="Images were detected but could not be processed.",
            )

        log.info(
            f"Image generation complete: {len(image_data_list)} image(s), "
            f"{elapsed_ms}ms, format={request.response_format}"
        )

        _increment_thread_count()
        return ImagesResponse(data=image_data_list)


def _sse_chunk(
    completion_id: str,
    model: str,
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    choices: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize one OpenAI ``chat.completion.chunk`` event as an SSE line."""
    if choices is None:
        choices = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


def _stream_chat_completion(
    response: ChatCompletionResponse,
    request: ChatCompletionRequest,
) -> StreamingResponse:
    """Wrap a fully-computed response as an OpenAI SSE stream.

    The browser round-trip is atomic (there is no incremental token emission),
    so the completed content/tool_calls are emitted as delta chunks followed by
    the finish and ``[DONE]`` markers. This makes ``stream=true`` clients work
    even though the reply only becomes available once generation finishes.
    """
    completion_id = response.id
    model = response.model
    choice = response.choices[0]
    content = choice.message.content
    tool_calls = choice.message.tool_calls
    include_usage = bool(
        request.stream_options and request.stream_options.include_usage
    )

    async def event_stream():
        # 1. role chunk
        yield _sse_chunk(
            completion_id, model, delta={"role": "assistant", "content": ""}
        )

        # 2. content / tool_calls delta
        if tool_calls:
            tc_serialized = [
                {
                    "index": i,
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            yield _sse_chunk(completion_id, model, delta={"tool_calls": tc_serialized})
        elif content:
            yield _sse_chunk(completion_id, model, delta={"content": content})

        # 3. finish_reason chunk
        yield _sse_chunk(completion_id, model, finish_reason=choice.finish_reason)

        # 4. usage chunk (only when requested) — empty choices per OpenAI spec
        if include_usage:
            yield _sse_chunk(
                completion_id,
                model,
                choices=[],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

        # 5. terminator
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Drive the provider browser for one chat-completion round-trip.

    Shared by the streaming and non-streaming paths. Builds the prompt (injecting
    tool definitions when present), sends it through the provider client,
    detects/repairs prompt echo, parses any tool calls, and returns an
    OpenAI-formatted response. The global browser lock is held for the round-trip.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty")

    client = _get_client()

    async with _get_lock():
        start_time = time.time()

        # ── Build the prompt ────────────────────────────────
        messages = list(request.messages)

        # If tools are provided, inject tool definitions as a system prompt
        # (unless tool_choice="none", which means ignore tools)
        has_tool_prompt = False
        if request.tools and request.tool_choice != "none":
            tool_system = _build_tool_system_prompt(
                request.tools, tool_choice=request.tool_choice
            )
            # Prepend as the first system message
            messages.insert(0, ChatMessage(role="system", content=tool_system))
            has_tool_prompt = True

        prompt = _build_prompt(messages)
        force_tool_protocol = (
            _forced_tool_name_from_choice(request.tool_choice) is not None
            or request.tool_choice == "required"
        )
        if (
            has_tool_prompt
            and force_tool_protocol
            and not any(msg.role == "tool" for msg in messages)
        ):
            prompt = f"{prompt}\n\n{_build_tool_protocol_suffix(request.tool_choice)}"
        log.info(
            f"POST /v1/chat/completions — model={request.model}, "
            f"{len(request.messages)} messages, prompt={len(prompt)} chars"
        )

        # ── Extract attachments from messages ──────────────
        image_paths: list[str] = []
        file_paths: list[str] = []
        for msg in request.messages:
            if msg.role == "user" and isinstance(msg.content, list):
                # Images (OpenAI vision format)
                image_urls = _extract_image_urls(msg.content)
                for url in image_urls:
                    local_path = await _download_file(url)
                    if local_path:
                        image_paths.append(local_path)
                # Generic file attachments
                file_attachments = _extract_file_attachments(msg.content)
                for fa in file_attachments:
                    local_path = await _download_file(fa)
                    if local_path:
                        file_paths.append(local_path)

        all_attachment_paths = image_paths + file_paths
        if all_attachment_paths:
            log.info(f"Extracted {len(image_paths)} image(s) and {len(file_paths)} file(s) from request")

        # Start a fresh conversation to avoid thread exhaustion
        await _ensure_fresh_chat()

        # ── Send to provider ────────────────────────────────
        try:
            result = await client.send_message(
                prompt,
                image_paths=image_paths or None,
                file_paths=file_paths or None,
            )
        except Exception as e:
            log.error(f"Provider error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Provider error: {str(e)}")

        response_text = result.message
        elapsed_ms = int((time.time() - start_time) * 1000)

        # ── Detect echo (extraction grabbed sent prompt instead of reply) ──
        _echo_markers = ["[System instruction:", "tool-calling mode", "Available functions:"]
        if response_text and has_tool_prompt and any(m in response_text for m in _echo_markers):
            log.warning("Response appears to echo the sent prompt — retrying extraction")
            try:
                await asyncio.sleep(1.5)
                if Config.PROVIDER == "claude":
                    from src.claude.detector import extract_last_response_via_copy
                else:
                    from src.chatgpt.detector import extract_last_response_via_copy
                retry_text = await extract_last_response_via_copy(client.page)
                if retry_text and not any(m in retry_text for m in _echo_markers):
                    response_text = retry_text
                    log.info(f"Retry extraction succeeded: {len(response_text)} chars")
                else:
                    log.warning("Retry extraction still echoed — stripping system prefix")
                    # Last resort: try to find assistant content after the prompt
                    idx = response_text.rfind("\n\n")
                    if idx > 0:
                        tail = response_text[idx:].strip()
                        if tail and not tail.startswith("["):
                            response_text = tail
            except Exception as e:
                log.warning(f"Retry extraction failed: {e}")

        # ── Check for tool calls ────────────────────────────
        tool_calls = None
        finish_reason = "stop"

        if has_tool_prompt and request.tools:
            tool_calls = _parse_tool_calls(response_text, request.tools)
            if tool_calls:
                finish_reason = "tool_calls"
                # When the model calls tools, content should be null
                response_text = None

        # ── Bad-call retry (bounded to one correction turn) ──────────────
        # Default handling when a tool call is needed but the model made a
        # "bad call": a forced/required parse miss, or an auto-mode meta-refusal
        # ("I can't fabricate a tool call ..."). One correction turn in the same
        # thread; parse only the retry text; never after a tool-result turn.
        bad_call = force_tool_protocol or _looks_like_tool_refusal(response_text)
        if (
            not tool_calls
            and has_tool_prompt
            and request.tools
            and bad_call
            and not any(msg.role == "tool" for msg in request.messages)
        ):
            forced_name = _forced_tool_name_from_choice(request.tool_choice)
            correction = _build_bad_call_correction(forced_name, force_tool_protocol)
            try:
                retry_result = await client.send_message(correction)
                retry_text = retry_result.message if retry_result else None
                if retry_text:
                    retry_calls = _parse_tool_calls(retry_text, request.tools)
                    if retry_calls:
                        tool_calls = retry_calls
                        finish_reason = "tool_calls"
                        response_text = None
                        log.info("Bad-call retry produced a tool call")
                    elif not force_tool_protocol and not _looks_like_tool_refusal(retry_text):
                        # Auto mode: keep the clean answer, drop the refusal.
                        response_text = retry_text
                        log.info("Bad-call retry produced a direct answer")
                    elapsed_ms = int((time.time() - start_time) * 1000)
            except Exception as e:
                log.warning(f"Bad-call retry send failed: {e}")

        # ── Build response ──────────────────────────────────
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(response_text or "")

        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=response_text,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        log.info(
            f"Response: {elapsed_ms}ms, finish_reason={finish_reason}, "
            f"tokens≈{response.usage.total_tokens}"
        )

        _increment_thread_count()
        return response


@openai_router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Converts the message array into a single prompt, sends it to the configured
    provider (ChatGPT or Claude) via browser automation, and returns an
    OpenAI-formatted response. Supports tool/function calling via prompt
    injection.

    When ``stream=true`` the response is emitted as OpenAI SSE chunks. The
    browser round-trip is atomic, so the full reply is delivered once generation
    completes rather than token-by-token.
    """
    response = await _run_completion(request)
    if request.stream:
        return _stream_chat_completion(response, request)
    return response


# ── Responses API (/v1/responses) ───────────────────────────────


def _responses_input_to_messages(
    input_data: str | list,
    instructions: str | None = None,
) -> list[ChatMessage]:
    """
    Convert Responses API `input` (string or item array) into a list of
    ChatMessage objects compatible with our existing _build_prompt().

    Handles:
      - Plain string → single user message
      - Array of message objects (role + content)
      - function_call items (assistant requested a tool)
      - function_call_output items (tool results)
    """
    messages: list[ChatMessage] = []

    # System prompt from `instructions`
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))

    # Simple string input
    if isinstance(input_data, str):
        messages.append(ChatMessage(role="user", content=input_data))
        return messages

    # Array of items
    for item in input_data:
        if isinstance(item, str):
            messages.append(ChatMessage(role="user", content=item))
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")

        if item_type == "function_call":
            # Assistant called a tool — record as assistant message with tool_calls
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")
            call_id = item.get("call_id", f"call_{uuid.uuid4().hex[:24]}")
            messages.append(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            type="function",
                            function=FunctionCallInfo(
                                name=name, arguments=arguments
                            ),
                        )
                    ],
                )
            )
        elif item_type == "function_call_output":
            # Tool result — map to role=tool
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            messages.append(
                ChatMessage(
                    role="tool",
                    content=output,
                    tool_call_id=call_id,
                )
            )
        elif item_type == "message" or role:
            # Regular message item
            r = role or item.get("role", "user")
            # Map "developer" role to "system"
            if r == "developer":
                r = "system"
            content = item.get("content", "")
            # Content can be a list of content parts or a string
            if isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts) if text_parts else ""
            messages.append(ChatMessage(role=r, content=content))

    return messages


def _responses_tools_to_chat_tools(
    tools: list[dict],
) -> list[ToolDefinition]:
    """
    Convert flat Responses API tool definitions to nested Chat Completions
    ToolDefinition format so we can reuse _build_tool_system_prompt().

    Responses:  {"type": "function", "name": "X", "parameters": {...}}
    Chat:       {"type": "function", "function": {"name": "X", "parameters": {...}}}
    """
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            tool = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        if tool.get("type") != "function":
            continue
        result.append(
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                ),
            )
        )
    return result


def _build_response_object(
    response_text: str | None,
    tool_calls: list[ToolCall] | None,
    request: "ResponsesRequest",
    prompt_tokens: int,
    completion_tokens: int,
) -> ResponseObject:
    """Build a full ResponseObject from the model output."""
    now = int(time.time())
    output: list = []
    output_text_val: str | None = None

    if tool_calls:
        for tc in tool_calls:
            output.append(
                ResponseFunctionCall(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    call_id=tc.id,
                ).model_dump()
            )
    else:
        text = response_text or ""
        msg = ResponseOutputMessage(
            content=[ResponseOutputText(text=text)]
        )
        output.append(msg.model_dump())
        output_text_val = text

    usage = ResponseUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    # Reconstruct tools for the response envelope
    tools_echo = []
    if request.tools:
        for t in request.tools:
            tools_echo.append(
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
            )

    return ResponseObject(
        created_at=now,
        completed_at=now,
        status="completed",
        model=request.model,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        output=output,
        output_text=output_text_val,
        temperature=request.temperature,
        top_p=request.top_p,
        tool_choice=request.tool_choice or "auto",
        tools=tools_echo,
        previous_response_id=request.previous_response_id,
        usage=usage,
        metadata=request.metadata or {},
    )


async def _stream_response_events(
    resp: ResponseObject,
    response_text: str | None,
    tool_calls: list[ToolCall] | None,
):
    """
    Yield SSE events for a streaming Responses API call.

    Since the browser backend doesn't truly stream, we emit the full
    response as a burst of events matching the OpenAI SSE contract:
      response.created → response.in_progress →
      output_item.added → content_part.added →
      output_text.delta (full text as one chunk) →
      output_text.done → content_part.done →
      output_item.done → response.completed
    """
    seq = 0
    resp_dict = resp.model_dump()

    def _event(event_type: str, data: dict) -> str:
        data["type"] = event_type
        data["sequence_number"] = seq
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # 1) response.created
    created_resp = dict(resp_dict)
    created_resp["status"] = "in_progress"
    created_resp["completed_at"] = None
    created_resp["output"] = []
    created_resp["output_text"] = None
    created_resp["usage"] = None
    yield _event("response.created", {"response": created_resp})
    seq += 1

    # 2) response.in_progress
    yield _event("response.in_progress", {"response": created_resp})
    seq += 1

    if tool_calls:
        # Emit function call output items
        for idx, tc in enumerate(tool_calls):
            fc_item = ResponseFunctionCall(
                name=tc.function.name,
                arguments=tc.function.arguments,
                call_id=tc.id,
            ).model_dump()

            # output_item.added
            fc_added = dict(fc_item)
            fc_added["status"] = "in_progress"
            yield _event("response.output_item.added", {
                "output_index": idx,
                "item": fc_added,
            })
            seq += 1

            # function_call_arguments.delta (one burst)
            yield _event("response.function_call_arguments.delta", {
                "item_id": fc_item["id"],
                "output_index": idx,
                "delta": tc.function.arguments,
            })
            seq += 1

            # function_call_arguments.done
            yield _event("response.function_call_arguments.done", {
                "item_id": fc_item["id"],
                "output_index": idx,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
            seq += 1

            # output_item.done
            yield _event("response.output_item.done", {
                "output_index": idx,
                "item": fc_item,
            })
            seq += 1
    else:
        # Emit text message output
        text = response_text or ""
        msg = ResponseOutputMessage(
            content=[ResponseOutputText(text=text)]
        )
        msg_dict = msg.model_dump()

        # output_item.added (empty content)
        msg_added = dict(msg_dict)
        msg_added["status"] = "in_progress"
        msg_added["content"] = []
        yield _event("response.output_item.added", {
            "output_index": 0,
            "item": msg_added,
        })
        seq += 1

        # content_part.added
        yield _event("response.content_part.added", {
            "item_id": msg_dict["id"],
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })
        seq += 1

        # output_text.delta — full text as one chunk
        if text:
            yield _event("response.output_text.delta", {
                "item_id": msg_dict["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            })
            seq += 1

        # output_text.done
        yield _event("response.output_text.done", {
            "item_id": msg_dict["id"],
            "output_index": 0,
            "content_index": 0,
            "text": text,
        })
        seq += 1

        # content_part.done
        yield _event("response.content_part.done", {
            "item_id": msg_dict["id"],
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        })
        seq += 1

        # output_item.done
        yield _event("response.output_item.done", {
            "output_index": 0,
            "item": msg_dict,
        })
        seq += 1

    # response.completed
    yield _event("response.completed", {"response": resp_dict})


@openai_router.post("/v1/responses")
async def create_response(request: ResponsesRequest):
    """
    OpenAI Responses API endpoint — compatible with Codex CLI.

    Accepts the Responses API format (flat tools, `input` field, `instructions`),
    translates to our internal format, sends to the browser, and returns a
    Responses-API-shaped response (or SSE stream).
    """
    # ── Validate ────────────────────────────────────────────
    if not request.input:
        raise HTTPException(status_code=400, detail="input cannot be empty")

    client = _get_client()

    async with _get_lock():
        start_time = time.time()

        # ── Convert input to ChatMessage list ───────────────
        messages = _responses_input_to_messages(
            request.input, instructions=request.instructions
        )

        # ── Convert flat tools to nested format ─────────────
        chat_tools: list[ToolDefinition] | None = None
        has_tool_prompt = False
        if request.tools:
            raw_tools = [
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
                for t in request.tools
            ]
            chat_tools = _responses_tools_to_chat_tools(raw_tools)
            if chat_tools and request.tool_choice != "none":
                tool_system = _build_tool_system_prompt(
                    chat_tools, tool_choice=request.tool_choice
                )
                messages.insert(
                    0, ChatMessage(role="system", content=tool_system)
                )
                has_tool_prompt = True

        prompt = _build_prompt(messages)
        force_tool_protocol = (
            _forced_tool_name_from_choice(request.tool_choice) is not None
            or request.tool_choice == "required"
        )
        if (
            has_tool_prompt
            and force_tool_protocol
            and not any(msg.role == "tool" for msg in messages)
        ):
            prompt = f"{prompt}\n\n{_build_tool_protocol_suffix(request.tool_choice)}"
        log.info(
            f"POST /v1/responses — model={request.model}, "
            f"input_type={'string' if isinstance(request.input, str) else 'array'}, "
            f"prompt={len(prompt)} chars, stream={request.stream}"
        )

        # Start a fresh conversation to avoid thread exhaustion
        await _ensure_fresh_chat()

        # ── Send to browser ────────────────────────────────
        try:
            result = await client.send_message(prompt)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "error state" in err_msg or "could not find chat input" in err_msg:
                # Page has a DNS/navigation error or UI is broken — attempt recovery
                log.warning(f"Page error detected, attempting recovery: {e}")
                from src.api.server import _browser
                if _browser and await _browser.recover_page():
                    # Retry after recovery
                    try:
                        result = await client.send_message(prompt)
                    except Exception as e2:
                        log.error(f"Provider error after recovery: {e2}", exc_info=True)
                        raise HTTPException(
                            status_code=500, detail=f"Provider error: {str(e2)}"
                        )
                else:
                    raise HTTPException(
                        status_code=503, detail="Browser page is in error state and recovery failed"
                    )
            else:
                log.error(f"Provider error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Provider error: {str(e)}"
                )
        except Exception as e:
            err_name = type(e).__name__
            # TargetClosedError means browser/page crashed — try recovery
            if "TargetClosed" in err_name or "closed" in str(e).lower():
                log.warning(f"Browser/page crashed ({err_name}), attempting recovery...")
                from src.api.server import _browser
                if _browser and await _browser.recover_page():
                    try:
                        result = await client.send_message(prompt)
                    except Exception as e2:
                        log.error(f"Provider error after crash recovery: {e2}", exc_info=True)
                        raise HTTPException(
                            status_code=500, detail=f"Provider error: {str(e2)}"
                        )
                else:
                    raise HTTPException(
                        status_code=503, detail=f"Browser crashed and recovery failed: {err_name}"
                    )
            else:
                log.error(f"Provider error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Provider error: {str(e)}"
                )

        response_text = result.message
        elapsed_ms = int((time.time() - start_time) * 1000)

        # ── Detect echo ────────────────────────────────────
        _echo_markers = [
            "[System instruction:",
            "tool-calling mode",
            "Available functions:",
        ]
        if (
            response_text
            and has_tool_prompt
            and any(m in response_text for m in _echo_markers)
        ):
            log.warning(
                "Response appears to echo the sent prompt — retrying extraction"
            )
            try:
                await asyncio.sleep(1.5)
                if Config.PROVIDER == "claude":
                    from src.claude.detector import extract_last_response_via_copy
                else:
                    from src.chatgpt.detector import extract_last_response_via_copy

                retry_text = await extract_last_response_via_copy(client.page)
                if retry_text and not any(
                    m in retry_text for m in _echo_markers
                ):
                    response_text = retry_text
                    log.info(
                        f"Retry extraction succeeded: {len(response_text)} chars"
                    )
                else:
                    log.warning(
                        "Retry extraction still echoed — stripping system prefix"
                    )
                    idx = response_text.rfind("\n\n")
                    if idx > 0:
                        tail = response_text[idx:].strip()
                        if tail and not tail.startswith("["):
                            response_text = tail
            except Exception as e:
                log.warning(f"Retry extraction failed: {e}")

        # ── Check for tool calls ────────────────────────────
        tool_calls = None
        if has_tool_prompt and chat_tools:
            tool_calls = _parse_tool_calls(response_text, chat_tools)
            if tool_calls:
                response_text = None

        # ── Forced-tool retry (bounded to one correction turn) ──────────
        # Mirror of the chat-completions retry: a forced/required tool call
        # that failed to parse gets a single short correction in the same
        # thread. Parse only the retry text; never after a tool result.
        if (
            not tool_calls
            and force_tool_protocol
            and not any(msg.role == "tool" for msg in messages)
        ):
            forced_name = _forced_tool_name_from_choice(request.tool_choice)
            skeleton = (
                f'{{"tool_calls":[{{"name":"{forced_name}","arguments":{{}}}}]}}'
                if forced_name
                else '{"tool_calls":[{"name":"<an available name>","arguments":{}}]}'
            )
            correction = (
                "Your previous response was not a valid JSON object. Ignore it "
                "and reply with ONLY the corrected raw JSON object, exactly this "
                f"shape: {skeleton}. Escape every double-quote inside string "
                "values, prefer single quotes inside shell commands, and output "
                "no prose and no code fence."
            )
            try:
                retry_result = await client.send_message(correction)
                retry_text = retry_result.message if retry_result else None
                if retry_text:
                    retry_calls = _parse_tool_calls(retry_text, chat_tools)
                    if retry_calls:
                        tool_calls = retry_calls
                        response_text = None
                        log.info("Forced tool-call retry succeeded after parse miss")
            except Exception as e:
                log.warning(f"Forced tool-call retry send failed: {e}")

        # ── Build response ──────────────────────────────────
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(response_text or "")

        resp = _build_response_object(
            response_text, tool_calls, request,
            prompt_tokens, completion_tokens,
        )

        log.info(
            f"Response: {elapsed_ms}ms, "
            f"tool_calls={len(tool_calls) if tool_calls else 0}, "
            f"tokens≈{resp.usage.total_tokens if resp.usage else 0}"
        )

        _increment_thread_count()

        # ── Stream or return ────────────────────────────────
        if request.stream:
            return StreamingResponse(
                _stream_response_events(resp, response_text, tool_calls),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return resp.model_dump()
