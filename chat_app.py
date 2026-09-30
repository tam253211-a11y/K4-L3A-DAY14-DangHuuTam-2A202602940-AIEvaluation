"""Browser chat for the OrbitTech RAG assistant (standard library server).

Usage:
    python chat_app.py          # http://127.0.0.1:8000
    python chat_app.py 8080     # another port
    python chat_app.py --no-browser

API keys:
    Each chat user can enter their own Gemini API key in the page; it is sent
    with each question in the ``X-Api-Key`` header, used for that request only,
    and never stored or logged by this server.
    When the page sends no key, the server falls back to OPENAI_API_KEY from
    .env (the host's key). Set REQUIRE_USER_KEY=1, or leave OPENAI_API_KEY
    unset, to make every user bring their own key.

Deployment: HOST and PORT environment variables override the defaults.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chat_engine import ChatEngine, ChatError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 20_000
MAX_QUESTION_CHARS = 600
MAX_ANSWER_CHARS = 8_000
MAX_HISTORY_ITEMS = 50
API_KEY_RE = re.compile(r"[A-Za-z0-9_\-.]{20,200}")
PAGE_PATH = Path(__file__).resolve().with_name("chat.html")
FLOW_PATH = Path(__file__).resolve().with_name("flow.html")


def _host_key() -> str:
    """The host's own key, or '' when users must bring theirs."""
    if os.getenv("REQUIRE_USER_KEY", "").strip().lower() in ("1", "true", "yes"):
        return ""
    return os.getenv("OPENAI_API_KEY", "").strip()


def _clean_history(raw: Any) -> list[dict[str, float]]:
    """Keep only well-formed earlier judge scores sent by the page."""
    if not isinstance(raw, list):
        return []
    history: list[dict[str, float]] = []
    for item in raw[-MAX_HISTORY_ITEMS:]:
        if not isinstance(item, dict):
            continue
        scores = {
            str(name): float(value)
            for name, value in item.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and 0.0 <= value <= 1.0
        }
        if scores:
            history.append(scores)
    return history


class ChatHandler(BaseHTTPRequestHandler):
    engine: ChatEngine

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif self.path in ("/flow", "/flow.html"):
            self._send(200, FLOW_PATH.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self._send_json(200, {"host_key_available": bool(_host_key())})
        else:
            self._send_json(404, {"error": "Not found"})

    def _read_payload(self) -> dict[str, Any] | None:
        """Parse the JSON body, or send a 400 and return None."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"error": "Request body is empty or too large"})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Expected a JSON object"})
            return None
        return payload

    def _resolve_key(self) -> tuple[str, str] | None:
        """Return (api_key, source), or send an error and return None."""
        user_key = (self.headers.get("X-Api-Key") or "").strip()
        if user_key and not API_KEY_RE.fullmatch(user_key):
            self._send_json(400, {"error": "That does not look like an API key."})
            return None
        api_key = user_key or _host_key()
        if not api_key:
            self._send_json(
                401,
                {"error": "Add your Gemini API key to start chatting.", "needs_key": True},
            )
            return None
        return api_key, "user" if user_key else "host"

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        if self.path not in ("/api/ask", "/api/evaluate"):
            self._send_json(404, {"error": "Not found"})
            return
        payload = self._read_payload()
        if payload is None:
            return

        question = payload.get("question")
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > MAX_QUESTION_CHARS
        ):
            self._send_json(
                400, {"error": f"Question must be 1 to {MAX_QUESTION_CHARS} characters"}
            )
            return
        question = question.strip()

        answer = payload.get("answer")
        chunk_ids = payload.get("chunk_ids")
        if self.path == "/api/evaluate" and (
            not isinstance(answer, str)
            or not answer.strip()
            or len(answer) > MAX_ANSWER_CHARS
            or not isinstance(chunk_ids, list)
            or not all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
        ):
            self._send_json(400, {"error": "Expected answer text and a list of chunk_ids"})
            return

        resolved = self._resolve_key()
        if resolved is None:
            return
        api_key, key_source = resolved

        try:
            if self.path == "/api/ask":
                result = self.engine.answer(question, api_key)
                result["key_source"] = key_source
            else:
                result = self.engine.evaluate(
                    question,
                    answer.strip(),
                    chunk_ids,
                    api_key,
                    history=_clean_history(payload.get("history")),
                )
        except ChatError as exc:
            self._send_json(exc.status, {"error": exc.message, "needs_key": exc.status == 401})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Request line and status only; headers (and so API keys) are never logged.
        print(f"[chat] {format % args}")


def main() -> int:
    open_browser = "--no-browser" not in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--no-browser"]
    host = os.getenv("HOST", "").strip() or DEFAULT_HOST
    port = int(args[0]) if args else int(os.getenv("PORT") or DEFAULT_PORT)
    try:
        ChatHandler.engine = ChatEngine()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    server = ThreadingHTTPServer((host, port), ChatHandler)
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    key_mode = "host key available as fallback" if _host_key() else "users must bring their own key"
    print(f"OrbitTech chat running at {url} ({key_mode}; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
