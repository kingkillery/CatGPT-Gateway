from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import unittest
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import catgpt_ask


@dataclass(frozen=True, slots=True)
class CliResult:
    code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RecordedPost:
    path: str
    auth: str
    body: dict


class RecordingServer(ThreadingHTTPServer):
    """HTTP server that records JSON POST requests."""

    requests: list[RecordedPost]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), RecordingHandler)
        self.requests = []


class RecordingHandler(BaseHTTPRequestHandler):
    """Handler for CatGPT helper CLI tests."""

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(
            RecordedPost(
                path=self.path,
                auth=self.headers.get("Authorization", ""),
                body=body,
            )
        )

        if self.path == "/chat":
            payload = {"message": "ongoing answer", "thread_id": "thread-1"}
        elif self.path == "/navigate":
            payload = {"url": body["url"]}
        else:
            payload = {"choices": [{"message": {"content": "v1 answer"}}]}

        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: str) -> None:
        return


@contextlib.contextmanager
def running_server():
    server = RecordingServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def run_cli(args: list[str]) -> CliResult:
    original_argv = sys.argv[:]
    stdout = io.StringIO()
    stderr = io.StringIO()
    sys.argv = ["catgpt_ask.py", *args]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = catgpt_ask.main()
    finally:
        sys.argv = original_argv
    return CliResult(code=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


class CatGptAskTest(unittest.TestCase):
    def test_ongoing_posts_to_native_chat_when_base_points_at_openai_v1(self) -> None:
        with running_server() as server:
            base = f"http://127.0.0.1:{server.server_port}/v1"

            result = run_cli(["--ongoing", "--base", base, "hello current chat"])

        self.assertEqual(result.code, 0)
        self.assertEqual(result.stdout, "ongoing answer\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request.path, "/chat")
        self.assertEqual(request.auth, "Bearer dummy123")
        self.assertEqual(request.body, {"message": "hello current chat"})

    def test_ongoing_target_model_and_intensity_are_sent_to_chat(self) -> None:
        with running_server() as server:
            base = f"http://127.0.0.1:{server.server_port}/v1"

            result = run_cli(
                [
                    "--ongoing",
                    "--target-url",
                    "https://chatgpt.com/g/example",
                    "--model",
                    "GPT-5 Thinking",
                    "--intensity",
                    "deep",
                    "--base",
                    base,
                    "optimize this prompt",
                ]
            )

        self.assertEqual(result.code, 0)
        self.assertEqual(result.stdout, "ongoing answer\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request.path, "/chat")
        self.assertEqual(
            request.body,
            {
                "message": "optimize this prompt",
                "target_url": "https://chatgpt.com/g/example",
                "model": "GPT-5 Thinking",
                "intensity": "deep",
            },
        )

    def test_default_mode_still_uses_openai_chat_completions(self) -> None:
        with running_server() as server:
            base = f"http://127.0.0.1:{server.server_port}/v1"

            result = run_cli(["--base", base, "hello temporary chat"])

        self.assertEqual(result.code, 0)
        self.assertEqual(result.stdout, "v1 answer\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request.path, "/v1/chat/completions")
        self.assertEqual(request.body["model"], "gpt-5.5-pro")
        self.assertEqual(
            request.body["messages"],
            [{"role": "user", "content": "hello temporary chat"}],
        )


if __name__ == "__main__":
    unittest.main()
