from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_state_lock = threading.Lock()
_latest_signal: dict[str, Any] | None = None
_server_started = False


class _TradingViewHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self._send_json(200, {"ok": True, "service": "tradingview-webhook"})
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        global _latest_signal

        if self.path != "/tradingview-webhook":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))

            signal = str(payload.get("signal", "")).upper()
            if signal not in {"YES", "NO", "NONE"}:
                self._send_json(400, {"ok": False, "error": "invalid_signal"})
                return

            normalized = {
                "signal": signal,
                "price": payload.get("price"),
                "vwap": payload.get("vwap"),
                "ticker": str(payload.get("ticker", "")),
                "received_at": time.time(),
            }

            with _state_lock:
                _latest_signal = normalized

            self._send_json(200, {"ok": True})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"ok": False, "error": "invalid_json"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_webhook_server() -> None:
    global _server_started

    with _state_lock:
        if _server_started:
            return
        _server_started = True

    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), _TradingViewHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="tradingview-webhook",
        daemon=True,
    )
    thread.start()


def get_latest_signal(max_age_seconds: float = 45.0) -> dict[str, Any] | None:
    with _state_lock:
        if _latest_signal is None:
            return None
        snapshot = dict(_latest_signal)

    age = time.time() - float(snapshot["received_at"])
    if age > max_age_seconds:
        return None

    return snapshot
