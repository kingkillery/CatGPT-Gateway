"""
Pydantic request/response schemas for the API.
"""

from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for sending a message."""
    message: str = Field(..., min_length=1, description="The message to send")
    model: Optional[str] = Field(None, description="Optional browser model label to select first")
    intensity: Optional[str] = Field(None, description="Optional reasoning intensity label/hint")
    target_url: Optional[str] = Field(None, description="Optional allowed UI chat-agent URL to open first")


class NavigationRequest(BaseModel):
    """Request body for navigating the browser session."""
    url: str = Field(..., min_length=1, description="Allowed UI chat-agent URL to open")


class NavigationResponse(BaseModel):
    """Response body after browser navigation."""
    url: str = Field("", description="Current browser URL after navigation")


class ModelOptionResponse(BaseModel):
    """One visible model-picker option."""
    label: str
    selected: bool = False
    disabled: bool = False
    source: str = ""


class BrowserModelOptionsResponse(BaseModel):
    """Visible model-picker options from the browser session."""
    opener_label: str = ""
    options: list[ModelOptionResponse] = Field(default_factory=list)


class ModelSelectionRequest(BaseModel):
    """Request body for selecting a browser model option."""
    model: Optional[str] = Field(None, description="Browser model label to select")
    intensity: Optional[str] = Field(None, description="Reasoning intensity label/hint to select")

class ModelSelectionResponse(BaseModel):
    """Result after selecting a browser model option."""
    matched: bool = False
    selected: str = ""
    reason: str = ""
    options: list[ModelOptionResponse] = Field(default_factory=list)


class ImageInfoResponse(BaseModel):
    """Image metadata in API response."""
    url: str = Field("", description="Original image URL from ChatGPT/DALL-E")
    alt: str = Field("", description="Alt text / image description")
    local_path: str = Field("", description="Local file path after download")
    prompt_title: str = Field("", description="Image generation title shown by ChatGPT")


class ChatResponse(BaseModel):
    """Response body with ChatGPT's reply."""
    message: str = Field(..., description="ChatGPT's response text (markdown)")
    thread_id: str = Field("", description="Conversation thread ID")
    response_time_ms: int = Field(0, description="Time to generate the response in ms")
    images: list[ImageInfoResponse] = Field(default_factory=list, description="Generated images")
    has_images: bool = Field(False, description="Whether the response contains images")


class ThreadInfo(BaseModel):
    """Thread metadata."""
    id: str
    title: str
    url: str


class ThreadListResponse(BaseModel):
    """List of recent threads."""
    threads: list[ThreadInfo]


class StatusResponse(BaseModel):
    """Health check / status."""
    status: str = "ok"
    logged_in: bool = False
    current_thread: str = ""
