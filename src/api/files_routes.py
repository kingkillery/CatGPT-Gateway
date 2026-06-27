"""
OpenAI-compatible Files API (`/v1/files`).

Lets clients upload a file once and reference it by `file_id` in subsequent
chat messages (content part `{"type":"file","file":{"file_id":"file-..."}}`).
When a referenced file is sent, the chat route resolves the id to the stored
local path and uploads it into the provider through the real browser
(`set_input_files`).

Uploaded bytes live under `downloads/uploads/` and a small JSON index maps
`file_id -> metadata`, so ids survive a container restart.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import Config
from src.log import setup_logging

log = setup_logging("files")

_UPLOADS_DIR = Config.PROJECT_ROOT / "downloads" / "uploads"
_INDEX_PATH = _UPLOADS_DIR / "_index.json"


# ── Schemas ──────────────────────────────────────────────────────


class FileObject(BaseModel):
    """OpenAI-compatible file object."""
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str = "assistants"
    status: str = "processed"


class FileListResponse(BaseModel):
    """Response for GET /v1/files."""
    object: str = "list"
    data: list[FileObject] = Field(default_factory=list)


class FileDeleteResponse(BaseModel):
    """Response for DELETE /v1/files/{id}."""
    id: str
    object: str = "file"
    deleted: bool


# ── Store ────────────────────────────────────────────────────────


class FileStore:
    """Disk-backed store mapping file_id to uploaded bytes + metadata.

    Single-process server, so an in-memory index with a JSON sidecar is
    sufficient and survives restarts.
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, object]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        if _INDEX_PATH.exists():
            try:
                self._index = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Could not load file index, starting empty: {e}")
                self._index = {}
        self._loaded = True

    def _persist(self) -> None:
        try:
            _INDEX_PATH.write_text(json.dumps(self._index), encoding="utf-8")
        except OSError as e:
            log.error(f"Could not persist file index: {e}")

    def create(self, filename: str, data: bytes, purpose: str) -> dict[str, object]:
        self._ensure_loaded()
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        safe = re.sub(r"[^\w.\-]", "_", filename) or "upload"
        path = _UPLOADS_DIR / f"{file_id}__{safe}"
        path.write_bytes(data)
        meta: dict[str, object] = {
            "id": file_id,
            "object": "file",
            "bytes": len(data),
            "created_at": int(time.time()),
            "filename": filename,
            "purpose": purpose,
            "status": "processed",
            "path": str(path),
        }
        self._index[file_id] = meta
        self._persist()
        log.info(f"Stored upload {file_id} ({len(data)} bytes) at {path}")
        return meta

    def get(self, file_id: str) -> dict[str, object] | None:
        self._ensure_loaded()
        return self._index.get(file_id)

    def list_all(self) -> list[dict[str, object]]:
        self._ensure_loaded()
        return list(self._index.values())

    def delete(self, file_id: str) -> bool:
        self._ensure_loaded()
        meta = self._index.pop(file_id, None)
        if meta is None:
            return False
        path_value = meta.get("path")
        if isinstance(path_value, str):
            try:
                Path(path_value).unlink(missing_ok=True)
            except OSError as e:
                log.warning(f"Could not remove file bytes for {file_id}: {e}")
        self._persist()
        return True

    def path_for(self, file_id: str) -> str | None:
        meta = self.get(file_id)
        if meta is None:
            return None
        path_value = meta.get("path")
        return path_value if isinstance(path_value, str) else None


file_store = FileStore()


# ── Routes ───────────────────────────────────────────────────────

files_router = APIRouter()


def _to_file_object(meta: dict[str, object]) -> FileObject:
    return FileObject(
        id=str(meta["id"]),
        bytes=int(meta["bytes"]),  # type: ignore[arg-type]
        created_at=int(meta["created_at"]),  # type: ignore[arg-type]
        filename=str(meta["filename"]),
        purpose=str(meta.get("purpose", "assistants")),
        status=str(meta.get("status", "processed")),
    )


@files_router.post("/v1/files", response_model=FileObject)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
) -> FileObject:
    """Store an uploaded file and return an OpenAI-compatible file object."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    meta = file_store.create(file.filename or "upload", data, purpose)
    return _to_file_object(meta)


@files_router.get("/v1/files", response_model=FileListResponse)
async def list_files() -> FileListResponse:
    """List uploaded files."""
    return FileListResponse(data=[_to_file_object(m) for m in file_store.list_all()])


@files_router.get("/v1/files/{file_id}", response_model=FileObject)
async def get_file(file_id: str) -> FileObject:
    """Retrieve metadata for one uploaded file."""
    meta = file_store.get(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    return _to_file_object(meta)


@files_router.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str) -> FileDeleteResponse:
    """Delete one uploaded file."""
    if not file_store.delete(file_id):
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    return FileDeleteResponse(id=file_id, deleted=True)
