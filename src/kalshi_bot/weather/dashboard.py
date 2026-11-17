"""Phone-friendly web dashboard for the weather bot.

Started by ``weather-run`` in the same process, on ``$PORT`` (Railway sets it),
default 8080. Shows P&L, open bets, every trade, settlements, and the latest
model-vs-market view for each open event. Refreshes every 15 seconds.

Routes: ``/`` page, ``/api/data`` JSON, ``/download/trades.csv``,
``/download/settlements.csv``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kalshi_bot.telemetry.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.weather.engine import WeatherEngine

logger = get_logger(__name__)


def recent_rows(log_dir: Path, types: set[str], max_files: int = 45) -> list[dict[str, Any]]:
    """Rows of the given types from the newest ``max_files`` daily log files, newest first."""
    out: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.jsonl"))[-max_files:]:
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if '"type"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") in types:
                        out.append(row)
        except OSError:
            continue
    out.sort(key=lambda r: str(r.get("t", "")), reverse=True)
    return out


def build_data(engine: WeatherEngine) -> dict[str, Any]:
    now = datetime.now(UTC)
    snaps = list(engine.last_snapshots)
    quotes: dict[str, dict[str, Any]] = {}
    for s in snaps:
        for b in s["brackets"]:
            quotes[b["ticker"]] = {**b, "city": s["station"], "kind": s["kind"], "day": s["day"]}

    rows = recent_rows(engine.datalog.dir, {"order", "settle"})
    orders = [r for r in rows if r["type"] == "order" and r.get("filled")]
    settles = [r for r in rows if r["type"] == "settle"]
    wins = sum(1 for r in settles if (r.get("pnl") or 0) > 0)
    losses = sum(1 for r in settles if (r.get("pnl") or 0) <= 0)

    positions: list[dict[str, Any]] = []
    account: dict[str, Any]
    if engine.mode == "live" and engine.live_account is not None:
        raw = engine.live_account["positions"]
        account = {"balance_cents": engine.live_account["balance_cents"]}
    else:
        led = engine.ledger
        raw = [
            {"ticker": t, "count": p.count, "cost_cents": p.cost_cents}
            for t, p in led.positions.items()
        ]
        account = {
            "starting_cents": led.starting_cents,
            "realized_cents": led.realized_cents,
            "fees_cents": led.fees_cents,
            "open_cost_cents": led.open_cost_cents(),
            "bankroll_cents": led.bankroll_cents(),
        }

    unrealized = 0
    marked = True
    for p in raw:
        q = quotes.get(p["ticker"], {})
        count = int(p["count"])
        side = "YES" if count > 0 else "NO"
        if side == "YES":
            bid = q.get("yes_bid")
            p_win = q.get("p")
        else:
            bid = q.get("no_bid") or (100 - q["yes_ask"] if q.get("yes_ask") else None)
            p_win = None if q.get("p") is None else 1 - q["p"]
        value = None if bid is None else abs(count) * bid
        if value is None:
            marked = False
        else:
            unrealized += value - int(p["cost_cents"])
        positions.append(
            {
                "ticker": p["ticker"],
                "city": q.get("city"),
                "kind": q.get("kind"),
                "day": q.get("day"),
                "label": q.get("label"),
                "side": side,
                "count": abs(count),
                "cost_cents": int(p["cost_cents"]),
                "avg_price": round(int(p["cost_cents"]) / abs(count), 1) if count else None,
                "bid": bid,
                "value_cents": value,
                "p_win": None if p_win is None else round(p_win, 3),
            }
        )
    account["unrealized_cents"] = unrealized
    account["fully_marked"] = marked

    return {
        "now": now.isoformat(timespec="seconds"),
        "mode": engine.mode,
        "stations": [s.code for s in engine.stations],
        "kinds": [k.value for k in engine.kinds],
        "started_at": engine.started_at.isoformat(timespec="seconds"),
        "last_cycle_at": None
        if engine.last_cycle_at is None
        else engine.last_cycle_at.isoformat(timespec="seconds"),
        "cycles": engine.cycles,
        "last_error": engine.last_error,
        "min_edge": engine.ws.min_edge,
        "account": account,
        "stats": {
            "trades": len(orders),
            "settled": len(settles),
            "wins": wins,
            "losses": losses,
            "win_rate": round(100 * wins / len(settles), 1) if settles else None,
        },
        "positions": positions,
        "markets": snaps,
        "orders": orders[:60],
        "settles": settles[:60],
    }


def to_csv(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


ORDER_COLS = ["t", "mode", "kind", "event", "ticker", "action", "side", "count", "limit",
              "filled", "fill_price", "p_model", "p_used", "edge", "reason"]  # fmt: skip
SETTLE_COLS = ["t", "ticker", "result", "pnl"]

HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weather Bot</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--text:#e6edf3;--dim:#8b949e;
--good:#3fb950;--bad:#f85149;--warn:#d29922;--link:#58a6ff}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);
color:var(--text);margin:0;padding:14px 16px 40px;max-width:900px;margin-inline:auto}
h1{font-size:22px;margin:4px 0}h2{font-size:16px;margin:22px 0 8px;color:var(--dim);
text-transform:uppercase;letter-spacing:.04em}
.top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{padding:3px 9px;border-radius:999px;font-size:12px;font-weight:700;background:#1f6feb33;
color:var(--link)}.badge.live{background:#f8514933;color:var(--bad)}
.badge.paper{background:#3fb95033;color:var(--good)}
.sub{color:var(--dim);font-size:13px;margin:4px 0}
.err{background:#f8514922;border:1px solid var(--bad);border-radius:10px;padding:10px;
font-size:13px;margin-top:10px;white-space:pre-wrap;word-break:break-word}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px}
.c{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.k{color:var(--dim);font-size:12px}.v{font-size:21px;font-weight:700;margin-top:2px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.list .c{margin-bottom:8px;font-size:14px}.s{color:var(--dim);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
td:first-child,th:first-child{text-align:left}th{color:var(--dim);font-weight:500}
tr.hot td{background:#3fb95018}.bar{height:6px;background:#1f6feb;border-radius:3px}
.mk{margin-bottom:10px}.mk h3{font-size:15px;margin:0 0 4px}
details summary{cursor:pointer;list-style:none}details summary::-webkit-details-marker{display:none}
a{color:var(--link)}.scroll{overflow-x:auto}.empty{color:var(--dim);font-size:14px}
</style></head><body>
<div class=top><h1>🌤️ Weather Bot</h1><span id=mode class=badge>…</span></div>
<div class=sub id=status>Loading…</div><div id=err></div>
<h2>Money</h2><div class=g id=money></div>
<h2>Results</h2><div class=g id=stats></div>
<h2>Open bets</h2><div class=list id=open></div>
<h2>Markets right now</h2><div class=sub>Model % vs Kalshi price. Green rows = bot sees an edge.
Tap a city to expand.</div><div id=markets></div>
<h2>Recent trades</h2><div id=trades></div>
<h2>Settled</h2><div id=settles></div>
<p class=sub><a href="/download/trades.csv">Download trades CSV</a> ·
<a href="/download/settlements.csv">Download settlements CSV</a></p>
<script>
const $=id=>document.getElementById(id);
const usd=c=>c==null?'-':(c<0?'-':'')+'$'+(Math.abs(c)/100).toFixed(2);
const sgn=c=>c==null?'-':(c>0?'+':'')+usd(c);
const cls=c=>c>0?'pos':c<0?'neg':'';
const pct=p=>p==null?'-':(p*100).toFixed(0)+'%';
const card=(k,v,c='')=>`<div class=c><div class=k>${k}</div><div class="v ${c}">${v}</div></div>`;
const ago=t=>{if(!t)return'never';const s=(Date.now()-Date.parse(t))/1000;
return s<90?Math.round(s)+'s ago':s<5400?Math.round(s/60)+' min ago':Math.round(s/3600)+' h ago'};
const day=d=>d?new Date(d+'T12:00:00').toLocaleDateString([], {month:'short',day:'numeric'}):'';
const when=t=>t?new Date(t).toLocaleString([], {month:'short',day:'numeric',hour:'numeric',
minute:'2-digit'}):'';
const open_={};
async function r(){try{
const d=await(await fetch('/api/data',{cache:'no-store'})).json(),a=d.account,s=d.stats;
$('mode').textContent=d.mode==='log'?'LOG ONLY':d.mode.toUpperCase();
$('mode').className='badge '+d.mode;
$('status').textContent=`${d.stations.join(', ')} · ${d.kinds.join(' & ')} · last update `+
`${ago(d.last_cycle_at)} · ${d.cycles} cycles`;
$('err').innerHTML=d.last_error?`<div class=err>Last error ${d.last_error}</div>`:'';
if(d.mode==='live'){$('money').innerHTML=card('Kalshi balance',usd(a.balance_cents))+
card('Open bets (mark)',sgn(a.unrealized_cents),cls(a.unrealized_cents));}
else{const tot=a.realized_cents+a.unrealized_cents;
$('money').innerHTML=card('Total P&L',sgn(tot),cls(tot))+
card('Realized (after fees)',sgn(a.realized_cents),cls(a.realized_cents))+
card('Open bets (mark)',sgn(a.unrealized_cents),cls(a.unrealized_cents))+
card('Cash available',usd(a.bankroll_cents))+card('In open bets',usd(a.open_cost_cents))+
card('Fees paid',usd(a.fees_cents));}
$('stats').innerHTML=card('Trades',s.trades)+card('Settled',s.settled)+
card('Record',s.wins+'–'+s.losses)+card('Win rate',s.win_rate==null?'-':s.win_rate+'%');
$('open').innerHTML=d.positions.length?d.positions.map(p=>{
const pl=p.value_cents==null?null:p.value_cents-p.cost_cents;
return `<div class=c><b>${p.city||''} ${p.kind||''} ${day(p.day)} · ${p.label||p.ticker}°</b>
 — ${p.side} ×${p.count} @ ${p.avg_price}¢<br><span class=s>cost ${usd(p.cost_cents)} · now
 ${p.bid??'-'}¢ · model win chance ${pct(p.p_win)} · </span><span class="${cls(pl)}">${sgn(pl)}
 </span></div>`}).join(''):'<div class=empty>No open bets.</div>';
$('markets').innerHTML=d.markets.length?d.markets.map(m=>{
const top=m.brackets.reduce((x,b)=>b.p>x.p?b:x,m.brackets[0]);
const hot=m.brackets.some(b=>Math.max(b.edge_yes??-1,b.edge_no??-1)>=d.min_edge);
const obs=m.obs_hi!=null?` · ${m.kind} so far ~${Math.round((m.obs_lo+m.obs_hi)/2)}°`:'';
const rows=m.brackets.map(b=>{const e=Math.max(b.edge_yes??-1,b.edge_no??-1);
return `<tr class="${e>=d.min_edge?'hot':''}"><td>${b.label}°</td><td>${pct(b.p)}</td>
<td><div class=bar style="width:${Math.round(b.p*60)}px;margin-left:auto"></div></td>
<td>${b.yes_bid??'-'}/${b.yes_ask??'-'}¢</td><td>${e>-1?Math.round(e*100)+0+'¢':'-'}</td></tr>`}).join('');
const key=m.event;return `<details class="c mk" ${open_[key]?'open':''}
 ontoggle="open_['${key}']=this.open"><summary><h3>${hot?'🟢 ':''}${m.city} · ${m.kind} ·
 ${day(m.day)}</h3><div class=s>model ${m.mean}° ±${m.std} · most likely ${top.label}°
 (${pct(top.p)})${obs}${m.intraday?' · today':''}</div></summary>
<table><tr><th>Bracket</th><th>Model</th><th></th><th>Bid/Ask</th><th>Best edge</th></tr>${rows}
</table>${m.decisions.length?'<div class=s>Bot action: '+m.decisions.join('; ')+'</div>':''}
</details>`}).join(''):'<div class=empty>No markets modeled yet. Give it a minute.</div>';
$('trades').innerHTML=d.orders.length?'<div class=scroll><table><tr><th>When</th>'+
'<th>Market</th><th>Side</th>'+
'<th>Qty</th><th>Price</th><th>Edge</th></tr>'+d.orders.map(o=>`<tr><td>${when(o.t)}</td>
<td>${o.ticker.split('-').slice(-2).join(' ')}</td><td>${o.action} ${o.side}</td>
<td>${o.filled}</td><td>${o.fill_price}¢</td><td>${o.edge==null?'-':(o.edge*100).toFixed(0)+'¢'}
</td></tr>`).join('')+'</table></div>':'<div class=empty>No trades yet.</div>';
$('settles').innerHTML=d.settles.length?'<table><tr><th>When</th><th>Market</th><th>Result</th>'+
'<th>P&L</th></tr>'+d.settles.map(x=>`<tr><td>${when(x.t)}</td><td>${x.ticker.split('-')
.slice(-2).join(' ')}</td><td>${x.result.toUpperCase()}</td><td class="${cls(x.pnl)}">
${sgn(x.pnl)}</td></tr>`).join('')+'</table>':'<div class=empty>Nothing settled yet.</div>';
}catch(e){$('status').textContent='Dashboard can\\'t reach the bot: '+e}}
r();setInterval(r,15000);
</script></body></html>"""  # noqa: RUF001


class _Handler(BaseHTTPRequestHandler):
    engine: WeatherEngine

    def do_GET(self) -> None:
        try:
            path = self.path.split("?", 1)[0]
            if path == "/api/data":
                return self._send(
                    json.dumps(build_data(self.engine), default=str).encode(), "application/json"
                )
            if path == "/download/trades.csv":
                rows = recent_rows(self.engine.datalog.dir, {"order"}, max_files=10_000)
                return self._send(to_csv(rows, ORDER_COLS), "text/csv", "weather_trades.csv")
            if path == "/download/settlements.csv":
                rows = recent_rows(self.engine.datalog.dir, {"settle"}, max_files=10_000)
                return self._send(to_csv(rows, SETTLE_COLS), "text/csv", "weather_settlements.csv")
            if path in ("/", "/index.html"):
                return self._send(HTML.encode(), "text/html; charset=utf-8")
            if path == "/health":
                return self._send(b"ok", "text/plain")
        except Exception:
            logger.exception("dashboard request failed")
            self.send_response(500)
            self.end_headers()
            return None
        self.send_response(404)
        self.end_headers()
        return None

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


def start_weather_dashboard(engine: WeatherEngine, port: int | None = None) -> ThreadingHTTPServer:
    """Serve the dashboard in a background thread; returns the server."""
    handler = type("WeatherDashboardHandler", (_Handler,), {"engine": engine})
    server = ThreadingHTTPServer(("0.0.0.0", port or int(os.getenv("PORT", "8080"))), handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="weather-dashboard").start()
    logger.warning("Weather dashboard on port %d", server.server_address[1])
    return server
