#!/usr/bin/env python3
# ─── How to run ───
#   python scripts/catgpt_ask.py "your question"
#   python scripts/catgpt_ask.py --ongoing "ask in the current browser chat"
#   python scripts/catgpt_ask.py --ongoing --model "GPT-5 Thinking" --target-url https://chatgpt.com/g/... "ask a custom GPT"
#   echo "your question" | python scripts/catgpt_ask.py
#   python scripts/catgpt_ask.py --system "You are a senior Rust reviewer" "Review this: ..."
#   python scripts/catgpt_ask.py --file ./notes.txt "Summarize the attached file"
#
# Run a side command through the local CatGPT gateway. Default mode uses the
# OpenAI-compatible endpoint and is best for self-contained reasoning. The
# --ongoing mode types into the current browser session, so it can use whatever
# ChatGPT/GPT/app/tools are active in that session. Responses can be slow
# (especially deep-reasoning models), hence the long default timeout.
#
# Dependency-free (stdlib only): talks to http://localhost:8000/v1 or /chat.
"""CLI to consult ChatGPT Pro (gpt-5.5-pro) via the local CatGPT gateway."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid


def _post_json(url: str, token: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))



def _native_base_from_openai_base(base: str) -> str:
    """Return the gateway root URL for native session endpoints."""
    normalized = base.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized

def _upload_file(base: str, token: str, path: str, timeout: float) -> str:
    """Upload a file via the gateway Files API; return its file_id."""
    with open(path, "rb") as fh:
        content = fh.read()
    filename = os.path.basename(path) or "upload"
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    boundary = f"----catgpt{uuid.uuid4().hex}"
    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(content)
    parts.append(f"\r\n--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="purpose"\r\n\r\n')
    parts.append(b"assistants")
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    payload = b"".join(parts)
    req = urllib.request.Request(
        f"{base}/files",
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["id"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ask the local CatGPT gateway via /v1 or the ongoing browser session.",
    )
    ap.add_argument("prompt", nargs="*", help="The question/prompt (or pipe via stdin).")
    ap.add_argument("--system", default=None, help="Optional system instruction.")
    ap.add_argument("--file", default=None, help="Optional file to attach (uploaded via Files API).")
    ap.add_argument("--model", default=None, help="API model id, or browser model label with --ongoing.")
    ap.add_argument("--base", default=os.getenv("CATGPT_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--token", default=os.getenv("CATGPT_TOKEN", "dummy123"))
    ap.add_argument("--timeout", type=float, default=float(os.getenv("CATGPT_TIMEOUT", "2400")))
    ap.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    ap.add_argument("--intensity", default=None, help="With --ongoing, request fast/normal/deep/pro intensity.")
    ap.add_argument(
        "--ongoing",
        action="store_true",
        help="Use the current browser chat via /chat instead of OpenAI /v1 chat completions.",
    )
    ap.add_argument(
        "--target-url",
        default=None,
        help="With --ongoing, navigate to a ChatGPT/Claude URL before sending.",
    )
    args = ap.parse_args()

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        ap.error("no prompt given (pass as args or via stdin)")

    base = args.base.rstrip("/")
    if (args.target_url or args.intensity) and not args.ongoing:
        ap.error("--target-url and --intensity require --ongoing")
    if args.ongoing:
        if args.system:
            ap.error("--ongoing cannot be combined with --system; include the instruction in the prompt")
        if args.file:
            ap.error("--ongoing cannot attach files; use default /v1 mode for file uploads")
        native_base = _native_base_from_openai_base(base)
        try:
            payload = {"message": prompt}
            if args.target_url:
                payload["target_url"] = args.target_url
            if args.model:
                payload["model"] = args.model
            if args.intensity:
                payload["intensity"] = args.intensity
            result = _post_json(
                f"{native_base}/chat",
                args.token,
                payload,
                args.timeout,
            )
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"connection failed (is the gateway running on {base}?): {e}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(result, indent=2))
            return 0
        try:
            print(result["message"] or "")
        except (KeyError, TypeError):
            print(json.dumps(result, indent=2))
        return 0

    # Build the user content. CatGPT cannot call tools or read your machine,
    # so everything it needs must be in this prompt (+ optional attached file).
    user_content: object = prompt
    if args.file:
        try:
            file_id = _upload_file(base, args.token, args.file, args.timeout)
        except (urllib.error.URLError, OSError, KeyError) as e:
            print(f"file upload failed: {e}", file=sys.stderr)
            return 2
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "file", "file": {"file_id": file_id}},
        ]

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": user_content})

    try:
        result = _post_json(
            f"{base}/chat/completions",
            args.token,
            {"model": args.model or os.getenv("CATGPT_MODEL", "gpt-5.5-pro"), "messages": messages},
            args.timeout,
        )
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"connection failed (is the gateway running on {base}?): {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    try:
        print(result["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
