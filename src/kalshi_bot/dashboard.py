"""Simple web dashboard for CobyStrategy."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DB_PATH = Path(os.getenv("COBY_DASHBOARD_DB", "/tmp/coby_dashboard.db"))
_db_lock = threading.Lock()
_server_started = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_lock:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    side TEXT,
                    entry_price INTEGER,
                    exit_price INTEGER,
                    reason TEXT,
                    count INTEGER DEFAULT 1,
                    pnl_cents INTEGER,
                    total_pnl_cents INTEGER,
                    entry_time TEXT,
                    exit_time TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS open_positions (
                    ticker TEXT PRIMARY KEY,
                    side TEXT,
                    entry_price INTEGER,
                    count INTEGER DEFAULT 1,
                    seconds_left REAL,
                    entry_time TEXT
                )
                """
            )

            conn.commit()


def record_entry(
    ticker: str,
    side: str,
    entry_price: int,
    count: int = 1,
    seconds_left: float | None = None,
) -> None:
    """Record a paper/live strategy entry for the dashboard."""
    _init_db()

    now = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO open_positions (
                    ticker,
                    side,
                    entry_price,
                    count,
                    seconds_left,
                    entry_time
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    side,
                    entry_price,
                    count,
                    seconds_left,
                    now,
                ),
            )
            conn.commit()


def record_exit(
    ticker: str,
    side: str,
    entry_price: int,
    exit_price: int,
    reason: str,
    count: int,
    pnl_cents: int,
    total_pnl_cents: int,
) -> None:
    """Record a completed strategy trade for the dashboard."""
    _init_db()

    now = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT entry_time
                FROM open_positions
                WHERE ticker = ?
                """,
                (ticker,),
            ).fetchone()

            entry_time = row["entry_time"] if row else None

            conn.execute(
                """
                INSERT INTO trades (
                    ticker,
                    side,
                    entry_price,
                    exit_price,
                    reason,
                    count,
                    pnl_cents,
                    total_pnl_cents,
                    entry_time,
                    exit_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    side,
                    entry_price,
                    exit_price,
                    reason,
                    count,
                    pnl_cents,
                    total_pnl_cents,
                    entry_time,
                    now,
                ),
            )

            conn.execute(
                """
                DELETE FROM open_positions
                WHERE ticker = ?
                """,
                (ticker,),
            )

            conn.commit()


def _dashboard_data() -> dict[str, Any]:
    _init_db()

    with _db_lock:
        with _connect() as conn:
            trades = conn.execute(
                """
                SELECT *
                FROM trades
                ORDER BY id DESC
                LIMIT 200
                """
            ).fetchall()

            open_positions = conn.execute(
                """
                SELECT *
                FROM open_positions
                ORDER BY entry_time DESC
                """
            ).fetchall()

    trade_rows = [dict(row) for row in trades]
    open_rows = [dict(row) for row in open_positions]

    wins = sum(1 for row in trade_rows if (row["pnl_cents"] or 0) > 0)
    losses = sum(1 for row in trade_rows if (row["pnl_cents"] or 0) < 0)
    breakeven = sum(1 for row in trade_rows if (row["pnl_cents"] or 0) == 0)

    total_trades = len(trade_rows)

    total_pnl = sum((row["pnl_cents"] or 0) for row in trade_rows)

    win_rate = (
        round((wins / total_trades) * 100, 1)
        if total_trades
        else 0.0
    )

    return {
        "trades": trade_rows,
        "open_positions": open_rows,
        "stats": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": win_rate,
            "total_pnl_cents": total_pnl,
        },
    }


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Coby Trading Bot</title>

<style>
body {
    margin: 0;
    padding: 18px;
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1 {
    margin-top: 4px;
    margin-bottom: 4px;
}

.subtitle {
    color: #8b949e;
    margin-bottom: 20px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}

.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px;
}

.big {
    font-size: 26px;
    font-weight: 700;
}

.label {
    color: #8b949e;
    font-size: 13px;
}

.green {
    color: #3fb950;
}

.red {
    color: #f85149;
}

.yellow {
    color: #d29922;
}

.section {
    margin-top: 22px;
}

.trade {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}

.trade-head {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-weight: 700;
}

.small {
    color: #8b949e;
    font-size: 13px;
    line-height: 1.5;
}

.status {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 9px;
    font-size: 12px;
    font-weight: 700;
}

.take-profit {
    background: rgba(46,160,67,.18);
    color: #3fb950;
}

.stop {
    background: rgba(248,81,73,.18);
    color: #f85149;
}

.open {
    background: rgba(210,153,34,.18);
    color: #d29922;
}
</style>
</head>

<body>

<h1>Coby Trading Bot</h1>
<div class="subtitle">Live strategy dashboard</div>

<div class="grid">
    <div class="card">
        <div class="label">Total P&L</div>
        <div id="pnl" class="big">$0.00</div>
    </div>

    <div class="card">
        <div class="label">Win Rate</div>
        <div id="winrate" class="big">0%</div>
    </div>

    <div class="card">
        <div class="label">Trades</div>
        <div id="trades" class="big">0</div>
    </div>

    <div class="card">
        <div class="label">Record</div>
        <div id="record" class="big">0-0</div>
    </div>
</div>

<div class="section">
    <h2>Open Position</h2>
    <div id="openPositions"></div>
</div>

<div class="section">
    <h2>Trade History</h2>
    <div id="tradeHistory"></div>
</div>

<script>

function money(cents) {
    const value = cents / 100;
    const sign = value > 0 ? "+" : "";
    return sign + "$" + value.toFixed(2);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

async function refresh() {
    try {
        const response = await fetch("/api/data");
        const data = await response.json();

        const stats = data.stats;

        const pnl = document.getElementById("pnl");

        pnl.textContent = money(stats.total_pnl_cents);

        pnl.className =
            "big " +
            (
                stats.total_pnl_cents > 0
                    ? "green"
                    : stats.total_pnl_cents < 0
                    ? "red"
                    : ""
            );

        document.getElementById("winrate").textContent =
            stats.win_rate.toFixed(1) + "%";

        document.getElementById("trades").textContent =
            stats.total_trades;

        document.getElementById("record").textContent =
            stats.wins + "-" + stats.losses;

        const open = document.getElementById("openPositions");

        if (!data.open_positions.length) {
            open.innerHTML =
                '<div class="card small">No open position</div>';
        } else {
            open.innerHTML = data.open_positions.map(position => `
                <div class="trade">
                    <div class="trade-head">
                        <span>${escapeHtml(position.side)}</span>
                        <span class="status open">OPEN</span>
                    </div>

                    <div class="small">
                        Entry: ${position.entry_price}¢<br>
                        Contracts: ${position.count}<br>
                        Seconds left at entry:
                        ${
                            position.seconds_left === null
                                ? "-"
                                : Number(position.seconds_left).toFixed(1)
                        }
                        <br>
                        ${escapeHtml(position.ticker)}
                    </div>
                </div>
            `).join("");
        }

        const history = document.getElementById("tradeHistory");

        if (!data.trades.length) {
            history.innerHTML =
                '<div class="card small">No completed trades yet</div>';
        } else {
            history.innerHTML = data.trades.map(trade => {

                const resultClass =
                    trade.pnl_cents > 0
                        ? "green"
                        : trade.pnl_cents < 0
                        ? "red"
                        : "";

                const reasonClass =
                    trade.reason === "STOP"
                        ? "stop"
                        : "take-profit";

                return `
                    <div class="trade">

                        <div class="trade-head">
                            <span>
                                ${escapeHtml(trade.side)}
                                ${trade.entry_price}¢ → ${trade.exit_price}¢
                            </span>

                            <span class="${resultClass}">
                                ${money(trade.pnl_cents)}
                            </span>
                        </div>

                        <div style="margin-top:8px;">
                            <span class="status ${reasonClass}">
                                ${escapeHtml(trade.reason)}
                            </span>
                        </div>

                        <div class="small" style="margin-top:8px;">
                            Contracts: ${trade.count}<br>
                            ${escapeHtml(trade.ticker)}
                        </div>

                    </div>
                `;
            }).join("");
        }

    } catch (error) {
        console.error(error);
    }
}

refresh();

setInterval(refresh, 3000);

</script>

</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:

        if self.path == "/api/data":

            payload = json.dumps(_dashboard_data()).encode("utf-8")

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/json",
            )

            self.send_header(
                "Content-Length",
                str(len(payload)),
            )

            self.end_headers()

            self.wfile.write(payload)

            return

        if self.path == "/" or self.path.startswith("/?"):

            payload = HTML.encode("utf-8")

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )

            self.send_header(
                "Content-Length",
                str(len(payload)),
            )

            self.end_headers()

            self.wfile.write(payload)

            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_dashboard() -> None:
    """Start the dashboard HTTP server in a background thread."""

    global _server_started

    if _server_started:
        return

    _server_started = True

    _init_db()

    port = int(os.getenv("PORT", "8080"))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        DashboardHandler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="coby-dashboard",
    )

    thread.start()
