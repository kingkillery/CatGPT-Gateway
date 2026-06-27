"""
API routes — FastAPI router for ChatGPT interaction.

Endpoints:
  POST /chat              Send a message in the current/new thread
  POST /thread/{id}/chat  Send a message in a specific thread
  POST /thread/new        Start a new conversation
  GET  /threads           List recent threads
  GET  /status            Health check + login status
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BrowserModelOptionsResponse,
    ChatRequest,
    ChatResponse,
    ImageInfoResponse,
    ModelOptionResponse,
    ModelSelectionRequest,
    ModelSelectionResponse,
    NavigationRequest,
    NavigationResponse,
    StatusResponse,
    ThreadInfo,
    ThreadListResponse,
)
from src.browser.manager import BrowserManager
from src.chatgpt.client import ChatGPTClient
from src.chatgpt.model_selector import inspect_model_picker, select_model_option
from src.claude.client import ClaudeClient
from src.log import setup_logging

log = setup_logging("api_routes")

router = APIRouter()

# Serialize browser access — single page, not thread-safe
_lock = asyncio.Lock()

# Global reference — set by the server on startup
_client: ChatGPTClient | ClaudeClient | None = None
_browser: BrowserManager | None = None


def set_client(client: ChatGPTClient | ClaudeClient, browser: BrowserManager) -> None:
    """Called by server.py to inject the client instance."""
    global _client, _browser
    _client = client
    _browser = browser


def _get_client() -> ChatGPTClient | ClaudeClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    return _client


def _get_browser() -> BrowserManager:
    if _browser is None:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    return _browser


def _is_allowed_chat_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and hostname in {"chatgpt.com", "chat.openai.com", "claude.ai"}
    )


def _build_response(result) -> ChatResponse:
    """Convert internal ChatResponse to API ChatResponse with image data."""
    images = [
        ImageInfoResponse(
            url=img.url,
            alt=img.alt,
            local_path=img.local_path,
            prompt_title=img.prompt_title,
        )
        for img in (result.images or [])
    ]
    return ChatResponse(
        message=result.message,
        thread_id=result.thread_id,
        response_time_ms=result.response_time_ms,
        images=images,
        has_images=result.has_images,
    )


def _model_options_response(options) -> list[ModelOptionResponse]:
    return [
        ModelOptionResponse(
            label=option.label,
            selected=option.selected,
            disabled=option.disabled,
            source=option.source,
        )
        for option in options
    ]


def _message_with_intensity_hint(message: str, intensity: str | None) -> str:
    if not intensity:
        return message
    return (
        f"[Requested reasoning intensity: {intensity}. Use the closest available "
        f"mode/tool behavior for this answer.]\\n\\n{message}"
    )


async def _select_chatgpt_model(
    client: ChatGPTClient | ClaudeClient,
    model: str | None,
    intensity: str | None,
) -> ModelSelectionResponse:
    if not model and not intensity:
        return ModelSelectionResponse(matched=True, reason="no selection requested")
    if not isinstance(client, ChatGPTClient):
        raise HTTPException(status_code=400, detail="Browser model selection is only implemented for ChatGPT")
    selection = await select_model_option(client.page, model=model, intensity=intensity)
    response = ModelSelectionResponse(
        matched=selection.matched,
        selected=selection.selected,
        reason=selection.reason,
        options=_model_options_response(selection.options),
    )
    if model and not selection.matched:
        raise HTTPException(status_code=400, detail=response.model_dump())
    if selection.matched and selection.selected:
        log.info(f"Selected ChatGPT model option: {selection.selected}")
    elif intensity:
        log.warning(f"No ChatGPT model option matched intensity '{intensity}'; using prompt hint")
    return response


# ── Chat ────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message in the current conversation."""
    client = _get_client()
    log.info(f"POST /chat — {len(req.message)} chars")

    async with _lock:
        try:
            if req.target_url:
                if not _is_allowed_chat_url(req.target_url):
                    detail = (
                        "URL must be https://chatgpt.com, "
                        "https://chat.openai.com, or https://claude.ai"
                    )
                    raise HTTPException(status_code=400, detail=detail)
                await _get_browser().navigate(req.target_url)
            await _select_chatgpt_model(client, req.model, req.intensity)
            message = _message_with_intensity_hint(req.message, req.intensity)
            result = await client.send_message(message)
            return _build_response(result)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Chat error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/thread/{thread_id}/chat", response_model=ChatResponse)
async def chat_in_thread(thread_id: str, req: ChatRequest) -> ChatResponse:
    """Send a message in a specific thread. Navigates to it first."""
    client = _get_client()
    log.info(f"POST /thread/{thread_id}/chat — {len(req.message)} chars")

    async with _lock:
        try:
            # Navigate to the thread if not already there
            current_tid = client._extract_thread_id()
            if current_tid != thread_id:
                await client.navigate_to_thread(thread_id)

            await _select_chatgpt_model(client, req.model, req.intensity)
            message = _message_with_intensity_hint(req.message, req.intensity)
            result = await client.send_message(message)
            return _build_response(result)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Thread chat error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/thread/new", response_model=ChatResponse)
async def new_thread(req: ChatRequest) -> ChatResponse:
    """Start a new conversation and send the first message."""
    client = _get_client()
    log.info(f"POST /thread/new — {len(req.message)} chars")

    async with _lock:
        try:
            await client.new_chat()
            await _select_chatgpt_model(client, req.model, req.intensity)
            message = _message_with_intensity_hint(req.message, req.intensity)
            result = await client.send_message(message)
            return _build_response(result)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"New thread error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# ── Threads ─────────────────────────────────────────────────────


@router.get("/threads", response_model=ThreadListResponse)
async def list_threads() -> ThreadListResponse:
    """List recent conversation threads from the sidebar."""
    client = _get_client()
    log.info("GET /threads")

    async with _lock:
        try:
            raw_threads = await client.list_threads()
            threads = [
                ThreadInfo(id=t["id"], title=t["title"], url=t["url"])
                for t in raw_threads
            ]
            return ThreadListResponse(threads=threads)
        except Exception as e:
            log.error(f"Threads list error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))



# ── Browser session ─────────────────────────────────────────────


@router.post("/navigate", response_model=NavigationResponse)
async def navigate(req: NavigationRequest) -> NavigationResponse:
    """Navigate the current browser session to a ChatGPT/Claude URL."""
    if not _is_allowed_chat_url(req.url):
        detail = (
            "URL must be https://chatgpt.com, "
            "https://chat.openai.com, or https://claude.ai"
        )
        raise HTTPException(status_code=400, detail=detail)
    browser = _get_browser()
    log.info(f"POST /navigate — {req.url}")

    async with _lock:
        try:
            await browser.navigate(req.url)
            return NavigationResponse(url=browser.page.url)
        except Exception as e:
            log.error(f"Navigate error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/browser", response_model=BrowserModelOptionsResponse)
async def browser_model_options() -> BrowserModelOptionsResponse:
    """Inspect visible model options in the current ChatGPT browser session."""
    client = _get_client()
    if not isinstance(client, ChatGPTClient):
        raise HTTPException(status_code=400, detail="Browser model inspection is only implemented for ChatGPT")
    log.info("GET /models/browser")

    async with _lock:
        try:
            state = await inspect_model_picker(client.page)
            return BrowserModelOptionsResponse(
                opener_label=state.opener_label,
                options=_model_options_response(state.options),
            )
        except Exception as e:
            log.error(f"Model inspection error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/select", response_model=ModelSelectionResponse)
async def select_browser_model(req: ModelSelectionRequest) -> ModelSelectionResponse:
    """Select a model option in the current ChatGPT browser session."""
    client = _get_client()
    log.info(f"POST /model/select — model={req.model}, intensity={req.intensity}")

    async with _lock:
        try:
            return await _select_chatgpt_model(client, req.model, req.intensity)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Model selection error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# ── Status ──────────────────────────────────────────────────────


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Health check — returns login status and current thread."""
    try:
        client = _get_client()
        logged_in = await _browser.is_logged_in()
        tid = client._extract_thread_id()
        return StatusResponse(status="ok", logged_in=logged_in, current_thread=tid)
    except Exception:
        return StatusResponse(status="ok", logged_in=False, current_thread="")
