"""Live Coby model dashboard and session telemetry."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

_VOLUME = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
_DEFAULT_DB = str(Path(_VOLUME) / "coby_eth_dashboard.db") if _VOLUME else "/tmp/coby_eth_dashboard.db"
DB_PATH = Path(os.getenv("COBY_DASHBOARD_DB", _DEFAULT_DB))
_db_lock = threading.RLock()
_server_started = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock, _db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL, side TEXT, entry_price INTEGER,
            exit_price INTEGER, reason TEXT, count INTEGER DEFAULT 1,
            pnl_cents INTEGER, total_pnl_cents INTEGER,
            entry_time TEXT, exit_time TEXT, execution_mode TEXT,
            stop_price INTEGER, take_profit INTEGER, seconds_left REAL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS open_positions (
            ticker TEXT PRIMARY KEY, side TEXT, entry_price INTEGER,
            count INTEGER DEFAULT 1, seconds_left REAL, entry_time TEXT,
            stop_price INTEGER, take_profit INTEGER, execution_mode TEXT)"""
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(open_positions)")}
        for name, kind in (
            ("stop_price", "INTEGER"),
            ("take_profit", "INTEGER"),
            ("execution_mode", "TEXT"),
        ):
            if name not in cols:
                conn.execute(f"ALTER TABLE open_positions ADD COLUMN {name} {kind}")
        trade_cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
        for name, kind in (
            ("execution_mode", "TEXT"),
            ("stop_price", "INTEGER"),
            ("take_profit", "INTEGER"),
            ("seconds_left", "REAL"),
        ):
            if name not in trade_cols:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {kind}")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS model_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, ticker TEXT,
            seconds_left REAL, bnb_price REAL, gold_price REAL, btc_price REAL, eth_price REAL, target_price REAL,
            separation REAL, yes_bid INTEGER, yes_ask INTEGER,
            no_bid INTEGER, no_ask INTEGER, model_yes REAL, model_no REAL,
            edge_yes REAL, edge_no REAL, momentum_15 REAL,
            momentum_60 REAL, volatility REAL, decision TEXT, reason TEXT)"""
        )
        model_cols = {r[1] for r in conn.execute("PRAGMA table_info(model_snapshots)")}
        if "bnb_price" not in model_cols:
            conn.execute("ALTER TABLE model_snapshots ADD COLUMN bnb_price REAL")
        if "eth_price" not in model_cols:
            conn.execute("ALTER TABLE model_snapshots ADD COLUMN eth_price REAL")
        if "btc_price" not in model_cols:
            conn.execute("ALTER TABLE model_snapshots ADD COLUMN btc_price REAL")
        if "gold_price" not in model_cols:
            conn.execute("ALTER TABLE model_snapshots ADD COLUMN gold_price REAL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_snapshots_id ON model_snapshots(id)")


def record_model_snapshot(**x: Any) -> None:
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    keys = [
        "ticker", "seconds_left", "gold_price", "target_price", "separation",
        "yes_bid", "yes_ask", "no_bid", "no_ask", "model_yes", "model_no",
        "edge_yes", "edge_no", "momentum_15", "momentum_60", "volatility",
        "decision", "reason",
    ]
    with _db_lock, _db() as conn:
        conn.execute(
            "INSERT INTO model_snapshots (ts," + ",".join(keys) + ") VALUES (" +
            ",".join("?" for _ in range(len(keys) + 1)) + ")",
            [now] + [x.get(k) for k in keys],
        )
        conn.execute(
            "DELETE FROM model_snapshots WHERE id < "
            "(SELECT COALESCE(MAX(id)-100000,0) FROM model_snapshots)"
        )


def record_entry(
    ticker: str,
    side: str,
    entry_price: int,
    count: int = 1,
    seconds_left: float | None = None,
    stop_price: int | None = None,
    take_profit: int | None = None,
    execution_mode: str | None = None,
) -> None:
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _db_lock, _db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO open_positions
            (ticker,side,entry_price,count,seconds_left,entry_time,
             stop_price,take_profit,execution_mode)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (ticker, side, entry_price, count, seconds_left, now,
             stop_price, take_profit, execution_mode),
        )


def get_open_position(ticker: str) -> dict[str, Any] | None:
    _init_db()
    with _db_lock, _db() as conn:
        row = conn.execute(
            "SELECT * FROM open_positions WHERE ticker=?", (ticker,)
        ).fetchone()
    return dict(row) if row else None


def update_open_count(ticker: str, count: int) -> None:
    _init_db()
    with _db_lock, _db() as conn:
        conn.execute("UPDATE open_positions SET count=? WHERE ticker=?", (count, ticker))


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
    with _db_lock, _db() as conn:
        row = conn.execute(
            "SELECT entry_time,execution_mode,stop_price,take_profit,seconds_left "
            "FROM open_positions WHERE ticker=?", (ticker,)
        ).fetchone()
        entry_time = row["entry_time"] if row else None
        execution_mode = row["execution_mode"] if row else None
        stop_price = row["stop_price"] if row else None
        take_profit = row["take_profit"] if row else None
        seconds_left = row["seconds_left"] if row else None
        conn.execute(
            """INSERT INTO trades
            (ticker,side,entry_price,exit_price,reason,count,pnl_cents,
             total_pnl_cents,entry_time,exit_time,execution_mode,stop_price,
             take_profit,seconds_left)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, side, entry_price, exit_price, reason, count,
             pnl_cents, total_pnl_cents, entry_time, now, execution_mode,
             stop_price, take_profit, seconds_left),
        )
        conn.execute("DELETE FROM open_positions WHERE ticker=?", (ticker,))


def _data() -> dict[str, Any]:
    _init_db()
    with _db_lock, _db() as conn:
        trades = [dict(r) for r in conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT 500"
        )]
        opens = [dict(r) for r in conn.execute(
            "SELECT * FROM open_positions ORDER BY entry_time DESC"
        )]
        snapshots = [dict(r) for r in conn.execute(
            "SELECT * FROM model_snapshots ORDER BY id DESC LIMIT 30"
        )]
        totals = conn.execute(
            """SELECT COUNT(*) AS total,
               SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN pnl_cents < 0 THEN 1 ELSE 0 END) AS losses,
               COALESCE(SUM(pnl_cents),0) AS pnl
               FROM trades"""
        ).fetchone()
    total = int(totals["total"] or 0)
    wins = int(totals["wins"] or 0)
    losses = int(totals["losses"] or 0)
    pnl = int(totals["pnl"] or 0)
    return {
        "trades": trades,
        "open_positions": opens,
        "latest": snapshots[0] if snapshots else None,
        "snapshots": snapshots,
        "stats": {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(100 * wins / total, 1) if total else 0,
            "total_pnl_cents": pnl,
        },
    }


def _csv(table: str) -> bytes:
    if table not in {"trades", "model_snapshots"}:
        raise ValueError("unsupported table")
    _init_db()
    with _db_lock, _db() as conn:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id")]
    if not rows:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


HTML = '''<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coby Gold Bot</title><style>body{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:18px}.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.c{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px;margin-bottom:8px}.v{font-size:24px;font-weight:700}.s{color:#8b949e;font-size:12px}a{color:#58a6ff}.green{color:#3fb950}.red{color:#f85149}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #30363d;text-align:left}</style><h1>Coby Gold Bot</h1><div><a href="/download/trades.csv">Download trades CSV</a> ÃÂÃÂ· <a href="/download/model.csv">Download model data CSV</a></div><h2>Live Model</h2><div class=g id=live></div><h2>Performance</h2><div class=g id=stats></div><h2>Open Position</h2><div id=open></div><h2>Recent Model Decisions</h2><div id=decisions></div><h2>Recent Trades</h2><div id=trades></div><script>const f=(x,n=1)=>x==null?'ÃÂ¢ÃÂÃÂ':Number(x).toFixed(n),card=(k,v)=>`<div class=c><div class=s>${k}</div><div class=v>${v}</div></div>`;async function r(){try{let d=await(await fetch('/api/data',{cache:'no-store'})).json(),x=d.latest||{},s=d.stats;live.innerHTML=card('Decision',x.decision||'WAIT')+card('Reason',x.reason||'ÃÂ¢ÃÂÃÂ')+card('Gold','$'+f(x.gold_price,2))+card('Target','$'+f(x.target_price,2))+card('Separation','$'+f(x.separation,2))+card('Seconds left',f(x.seconds_left,0))+card('Model YES',f(x.model_yes,1)+'%')+card('Model NO',f(x.model_no,1)+'%')+card('YES edge',f(x.edge_yes,1)+'ÃÂÃÂ¢')+card('NO edge',f(x.edge_no,1)+'ÃÂÃÂ¢')+card('15s momentum','$'+f(x.momentum_15,2))+card('60s momentum','$'+f(x.momentum_60,2))+card('Volatility',f(x.volatility,3));stats.innerHTML=card('P&L',(s.total_pnl_cents>=0?'+':'')+'$'+f(s.total_pnl_cents/100,2))+card('Win rate',f(s.win_rate,1)+'%')+card('Trades',s.total_trades)+card('Record',s.wins+'-'+s.losses);open.innerHTML=d.open_positions.length?d.open_positions.map(p=>`<div class=c>${p.side.toUpperCase()} @ ${p.entry_price}ÃÂÃÂ¢ ÃÂÃÂ· ${p.count} contracts ÃÂÃÂ· ${p.execution_mode||'unknown'}<br><span class=s>Stop ${p.stop_price}ÃÂÃÂ¢ ÃÂÃÂ· TP ${p.take_profit}ÃÂÃÂ¢ ÃÂÃÂ· ${p.ticker}</span></div>`).join(''):'<div class=c>No open position</div>';decisions.innerHTML=d.snapshots.slice(0,12).map(q=>`<div class=c><b>${q.decision||'WAIT'}</b> ÃÂÃÂ· ${q.reason||'ÃÂ¢ÃÂÃÂ'}<br><span class=s>Gold $${f(q.gold_price,2)} / target $${f(q.target_price,2)} ÃÂÃÂ· sep $${f(q.separation,2)} ÃÂÃÂ· ${f(q.seconds_left,0)}s ÃÂÃÂ· YES ${f(q.model_yes,1)}% ÃÂÃÂ· edge ${f(q.edge_yes,1)}ÃÂÃÂ¢</span></div>`).join('');trades.innerHTML='<table><tr><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr>'+d.trades.slice(0,25).map(t=>`<tr><td>${t.side}</td><td>${t.entry_price}ÃÂÃÂ¢</td><td>${t.exit_price}ÃÂÃÂ¢</td><td>${(t.pnl_cents/100).toFixed(2)}</td><td>${t.reason}<br><span class=s>${t.execution_mode||'unknown'} ÃÂÃÂ· stop ${t.stop_price??'ÃÂ¢ÃÂÃÂ'}ÃÂÃÂ¢ ÃÂÃÂ· TP ${t.take_profit??'ÃÂ¢ÃÂÃÂ'}ÃÂÃÂ¢</span></td></tr>`).join('')+'</table>'}catch(e){console.error(e)}}r();setInterval(r,2000)</script>'''


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/data":
            return self._send(json.dumps(_data()).encode(), "application/json")
        if self.path == "/download/trades.csv":
            return self._send(_csv("trades"), "text/csv", "trades.csv")
        if self.path == "/download/model.csv":
            return self._send(_csv("model_snapshots"), "text/csv", "model_data.csv")
        if self.path == "/" or self.path.startswith("/?"):
            return self._send(HTML.encode(), "text/html; charset=utf-8")
        self.send_response(404)
        self.end_headers()

    def _send(self, payload: bytes, content_type: str, name: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        if name:
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        return


def start_dashboard() -> None:
    global _server_started
    if _server_started:
        return
    _server_started = True
    _init_db()
    server = ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="coby-dashboard").start()
