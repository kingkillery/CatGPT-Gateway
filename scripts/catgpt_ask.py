#!/usr/bin/env python3
# ─── How to run ───
#   python scripts/catgpt_ask.py "your question"
#   echo "your question" | python scripts/catgpt_ask.py
#   python scripts/catgpt_ask.py --system "You are a senior Rust reviewer" "Review this: ..."
#   python scripts/catgpt_ask.py --file ./notes.txt "Summarize the attached file"
#
# Run a one-shot "side command" through the local CatGPT gateway, which exposes
# ChatGPT Pro (gpt-5.5-pro) as an OpenAI-compatible endpoint. This is a pure
# REASONING call: the model CANNOT call tools and cannot see your environment,
# so put ALL needed context in the prompt. Responses can be slow (a pro model
# may take many minutes), hence the long default timeout.
#
# Dependency-free (stdlib only): talks to http://localhost:8000/v1.
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
        description="Run a one-shot reasoning query through the CatGPT gateway (gpt-5.5-pro).",
    )
    ap.add_argument("prompt", nargs="*", help="The question/prompt (or pipe via stdin).")
    ap.add_argument("--system", default=None, help="Optional system instruction.")
    ap.add_argument("--file", default=None, help="Optional file to attach (uploaded via Files API).")
    ap.add_argument("--model", default=os.getenv("CATGPT_MODEL", "gpt-5.5-pro"))
    ap.add_argument("--base", default=os.getenv("CATGPT_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--token", default=os.getenv("CATGPT_TOKEN", "dummy123"))
    ap.add_argument("--timeout", type=float, default=float(os.getenv("CATGPT_TIMEOUT", "2400")))
    ap.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    args = ap.parse_args()

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        ap.error("no prompt given (pass as args or via stdin)")

    base = args.base.rstrip("/")

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
            {"model": args.model, "messages": messages},
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
