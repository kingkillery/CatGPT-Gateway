#!/usr/bin/env python3
# ─── How to run ───
#   python scripts/e2e_watch.py                 # loop every 5 min, log + stdout
#   python scripts/e2e_watch.py --once          # single pass, exit 0/1
#   python scripts/e2e_watch.py --interval 600  # custom interval (seconds)
#   nohup python scripts/e2e_watch.py >/dev/null 2>&1 &   # detached background
#
# Periodically verifies the FULL CatGPT gateway mechanism end-to-end:
#   1. /healthz reachable
#   2. a reasoning side-command returns the exact expected token
#   3. file upload (Files API -> file_id) is read back by the model
# Each pass appends a PASS/FAIL line to the log (default docker-logs/e2e_watch.log).
"""Standing end-to-end health watcher for the CatGPT gateway."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = os.getenv("CATGPT_BASE", "http://localhost:8000/v1")
TOKEN = os.getenv("CATGPT_TOKEN", "dummy123")
MODEL = os.getenv("CATGPT_MODEL", "gpt-5.5-pro")
HEALTH = BASE.rsplit("/v1", 1)[0] + "/healthz"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chat(prompt: str, timeout: float) -> str:
    body = json.dumps(
        {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""


def _upload_and_read(secret: str, timeout: float) -> str:
    # upload via Files API (multipart, stdlib)
    content = f"SECRET PHRASE: {secret}".encode()
    boundary = f"----e2e{uuid.uuid4().hex}"
    payload = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="probe.txt"\r\n',
        b"Content-Type: text/plain\r\n\r\n",
        content,
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="purpose"\r\n\r\nassistants',
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{BASE}/files",
        data=payload,
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        fid = json.loads(r.read().decode())["id"]
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read the attached file and reply with ONLY the secret phrase."},
            {"type": "file", "file": {"file_id": fid}},
        ]}],
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""


def run_pass(timeout: float) -> tuple[bool, str]:
    # 1. health
    try:
        with urllib.request.urlopen(HEALTH, timeout=15) as r:
            if json.loads(r.read().decode()).get("status") != "ok":
                return False, "health!=ok"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"health unreachable: {type(e).__name__}"
    # 2. reasoning
    token = f"watch-{uuid.uuid4().hex[:6]}"
    try:
        ans = _chat(f"Reply with exactly: {token}", timeout)
    except (urllib.error.URLError, OSError, KeyError) as e:
        return False, f"reasoning err: {type(e).__name__}: {str(e)[:80]}"
    if token not in ans:
        return False, f"reasoning mismatch (got {ans[:60]!r})"
    # 3. file upload read-back
    secret = f"{uuid.uuid4().hex[:8]}"
    try:
        ans = _upload_and_read(secret, timeout)
    except (urllib.error.URLError, OSError, KeyError) as e:
        return False, f"upload err: {type(e).__name__}: {str(e)[:80]}"
    if secret not in ans:
        return False, f"upload mismatch (got {ans[:60]!r})"
    return True, "health+reasoning+upload ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="Periodic CatGPT gateway E2E watcher.")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between passes.")
    ap.add_argument("--once", action="store_true", help="Run a single pass and exit.")
    ap.add_argument("--timeout", type=float, default=600.0, help="Per-call timeout (s).")
    ap.add_argument("--log", default=str(Path("docker-logs") / "e2e_watch.log"))
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    def emit(line: str) -> None:
        print(line, flush=True)
        try:
            with open(args.log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    while True:
        t0 = time.time()
        ok, detail = run_pass(args.timeout)
        emit(f"{_now()} {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s) {detail}")
        if args.once:
            return 0 if ok else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
