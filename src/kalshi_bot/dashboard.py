"""Simple web dashboard for confirmed CobyStrategy live fills."""

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
            conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL, side TEXT, entry_price INTEGER,
                exit_price INTEGER, reason TEXT, count INTEGER DEFAULT 1,
                pnl_cents INTEGER, total_pnl_cents INTEGER,
                entry_time TEXT, exit_time TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS open_positions (
                ticker TEXT PRIMARY KEY, side TEXT, entry_price INTEGER,
                count INTEGER DEFAULT 1, seconds_left REAL, entry_time TEXT)""")
            conn.commit()


def record_entry(
    ticker: str,
    side: str,
    entry_price: int,
    count: int = 1,
    seconds_left: float | None = None,
) -> None:
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        with _connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO open_positions
                (ticker, side, entry_price, count, seconds_left, entry_time)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, side, entry_price, count, seconds_left, now),
            )
            conn.commit()


def update_open_count(ticker: str, count: int) -> None:
    _init_db()
    with _db_lock:
        with _connect() as conn:
            conn.execute(
                "UPDATE open_positions SET count = ? WHERE ticker = ?",
                (count, ticker),
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
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        with _connect() as conn:
            row = conn.execute(
                "SELECT entry_time FROM open_positions WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            entry_time = row["entry_time"] if row else None
            conn.execute(
                """INSERT INTO trades
                (ticker, side, entry_price, exit_price, reason, count,
                 pnl_cents, total_pnl_cents, entry_time, exit_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker, side, entry_price, exit_price, reason, count,
                    pnl_cents, total_pnl_cents, entry_time, now,
                ),
            )
            conn.execute("DELETE FROM open_positions WHERE ticker = ?", (ticker,))
            conn.commit()


def _dashboard_data() -> dict[str, Any]:
    _init_db()
    with _db_lock:
        with _connect() as conn:
            trades = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT 200"
            ).fetchall()
            opens = conn.execute(
                "SELECT * FROM open_positions ORDER BY entry_time DESC"
            ).fetchall()

    trade_rows = [dict(r) for r in trades]
    open_rows = [dict(r) for r in opens]
    wins = sum(1 for r in trade_rows if (r["pnl_cents"] or 0) > 0)
    losses = sum(1 for r in trade_rows if (r["pnl_cents"] or 0) < 0)
    total = len(trade_rows)
    pnl = sum((r["pnl_cents"] or 0) for r in trade_rows)

    return {
        "trades": trade_rows,
        "open_positions": open_rows,
        "stats": {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "breakeven": total - wins - losses,
            "win_rate": round(wins / total * 100, 1) if total else 0.0,
            "total_pnl_cents": pnl,
        },
    }


HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coby Trading Bot</title><style>
body{margin:0;padding:18px;background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.card,.trade{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}.trade{margin-bottom:10px}.big{font-size:26px;font-weight:700}.label,.small{color:#8b949e;font-size:13px}.green{color:#3fb950}.red{color:#f85149}.section{margin-top:22px}.trade-head{display:flex;justify-content:space-between;font-weight:700}.status{border-radius:20px;padding:4px 9px;font-size:12px;font-weight:700}.open{color:#d29922}
</style></head><body><h1>Coby Trading Bot</h1><div class="small">Confirmed live fills only</div>
<div class="grid"><div class="card"><div class="label">Total P&L</div><div id="pnl" class="big">$0.00</div></div>
<div class="card"><div class="label">Win Rate</div><div id="wr" class="big">0%</div></div>
<div class="card"><div class="label">Trades</div><div id="tr" class="big">0</div></div>
<div class="card"><div class="label">Record</div><div id="rec" class="big">0-0</div></div></div>
<div class="section"><h2>Open Position</h2><div id="op"></div></div>
<div class="section"><h2>Trade History</h2><div id="hist"></div></div>
<script>
const esc=v=>String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
const money=c=>((c>0?"+":"")+"$"+(c/100).toFixed(2));
async function refresh(){try{const d=await(await fetch("/api/data")).json(),s=d.stats;
let p=document.getElementById("pnl");p.textContent=money(s.total_pnl_cents);p.className="big "+(s.total_pnl_cents>0?"green":s.total_pnl_cents<0?"red":"");
wr.textContent=s.win_rate.toFixed(1)+"%";tr.textContent=s.total_trades;rec.textContent=s.wins+"-"+s.losses;
op.innerHTML=d.open_positions.length?d.open_positions.map(x=>`<div class="trade"><div class="trade-head"><span>${esc(x.side)}</span><span class="status open">OPEN</span></div><div class="small">Entry: ${x.entry_price}Â¢<br>Contracts: ${x.count}<br>Seconds left at entry: ${x.seconds_left===null?"-":Number(x.seconds_left).toFixed(1)}<br>${esc(x.ticker)}</div></div>`).join(""):'<div class="card small">No open position</div>';
hist.innerHTML=d.trades.length?d.trades.map(x=>`<div class="trade"><div class="trade-head"><span>${esc(x.side)} ${x.entry_price}Â¢ â ${x.exit_price}Â¢</span><span class="${x.pnl_cents>0?"green":x.pnl_cents<0?"red":""}">${money(x.pnl_cents)}</span></div><div class="small">${esc(x.reason)} | Contracts: ${x.count}<br>${esc(x.ticker)}</div></div>`).join(""):'<div class="card small">No completed trades yet</div>';
}catch(e){console.error(e)}}refresh();setInterval(refresh,3000);
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/data":
            payload = json.dumps(_dashboard_data()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/" or self.path.startswith("/?"):
            payload = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_dashboard() -> None:
    global _server_started
    if _server_started:
        return
    _server_started = True
    _init_db()
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="coby-dashboard",
    ).start()
