"""Calibration jobs and the "does the model beat the market?" report."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.telemetry.logging import get_logger
from kalshi_bot.weather.calibration import (
    SIGMA_FLOOR,
    CalibrationStore,
    KindCalibration,
    fit_history,
    robust_std,
    trimmed_mean,
)
from kalshi_bot.weather.climate import fetch_cli
from kalshi_bot.weather.datalog import DataLog
from kalshi_bot.weather.fees import fee_per_contract
from kalshi_bot.weather.forecasts import (
    HISTORICAL_FORECAST_URL,
    OpenMeteoClient,
    window_extreme,
)
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind, Station, climate_day_window

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CalibrationOutcome:
    station: str
    kind: str
    n_days: int
    source_bias: dict[str, float]
    bias: float
    sigma: dict[str, float]
    applied: bool
    note: str = ""


def _sigma_from(s: float) -> dict[str, float]:
    return {
        "d0": round(max(SIGMA_FLOOR["d0"], s), 2),
        "d1": round(max(SIGMA_FLOOR["d1"], 1.3 * s), 2),
        "d2": round(max(SIGMA_FLOOR["d2"], 1.7 * s), 2),
    }


async def calibrate_history(
    stations: list[Station],
    kinds: list[Kind],
    ws: WeatherSettings,
    store: CalibrationStore,
    days: int = 180,
    min_days: int = 30,
) -> list[CalibrationOutcome]:
    """Fit source biases + spread from archived forecasts vs. CLI actuals."""
    end_day = datetime.now(UTC).date() - timedelta(days=2)
    start_day = end_day - timedelta(days=days)
    groups: list[list[str]] = []
    for g in [*ws.deterministic_models, *ws.ensemble_models]:
        if g and g not in groups:
            groups.append(g)

    om = OpenMeteoClient(ws.openmeteo_api_key, timeout=60.0)
    out: list[CalibrationOutcome] = []
    try:
        for st in stations:
            actuals_raw = await fetch_cli(
                st.icao, list(range(start_day.year, end_day.year + 1)), ws.user_agent
            )
            series: dict[str, Any] = {}
            for g in groups:
                errors: list[str] = []
                cur = start_day
                chunks = []
                while cur <= end_day:
                    chunk_end = min(end_day + timedelta(days=1), cur + timedelta(days=89))
                    hs = await om.fetch_group(
                        HISTORICAL_FORECAST_URL,
                        st,
                        g,
                        "det",
                        errors,
                        start_date=cur.isoformat(),
                        end_date=chunk_end.isoformat(),
                    )
                    if hs is not None:
                        chunks.append(hs)
                    cur = chunk_end + timedelta(days=1)
                if chunks:
                    model_id = chunks[0].source.split(":", 1)[1]
                    series[model_id] = chunks
                elif errors:
                    logger.warning("%s history %s: %s", st.code, g[0], errors[-1])

            for kind in kinds:
                is_high = kind is Kind.HIGH
                forecasts: dict[date, dict[str, float]] = defaultdict(dict)
                actuals: dict[date, float] = {}
                d = start_day
                while d <= end_day:
                    hi, lo = actuals_raw.get(d, (None, None))
                    a = hi if is_high else lo
                    if a is not None:
                        actuals[d] = a
                        w0, w1 = climate_day_window(st, d)
                        for model_id, chunks in series.items():
                            for hs in chunks:
                                v = window_extreme(hs, w0, w1, is_high)
                                if v.size and np.isfinite(v[0]):
                                    forecasts[d][model_id] = float(v[0])
                                    break
                    d += timedelta(days=1)
                biases, residuals = fit_history(dict(forecasts), actuals, min_n=min_days)
                cal = store.get(st.code, kind)
                applied = len(residuals) >= min_days and bool(biases)
                if applied:
                    cal = KindCalibration(
                        bias=0.0,
                        sigma=_sigma_from(robust_std(residuals)),
                        source_bias={**cal.source_bias, **biases},
                        n=len(residuals),
                        method="history",
                    )
                    store.put(st.code, kind, cal)
                out.append(
                    CalibrationOutcome(
                        st.code,
                        kind.value,
                        len(residuals),
                        biases,
                        cal.bias,
                        cal.sigma,
                        applied,
                        "" if applied else f"need >= {min_days} matched days",
                    )
                )
    finally:
        await om.aclose()
    store.save()
    return out


async def calibrate_from_log(
    stations: list[Station],
    kinds: list[Kind],
    ws: WeatherSettings,
    store: CalibrationStore,
    datalog: DataLog,
    min_n: int = 20,
) -> list[CalibrationOutcome]:
    """Refine station bias and per-lead spread from the bot's own logged forecasts."""
    rows = [r for r in datalog.rows("model") if not r.get("intraday")]
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in rows:
        latest[(r["station"], r["kind"], r["day"], r["bucket"])] = r
    out: list[CalibrationOutcome] = []
    cutoff = datetime.now(UTC).date() - timedelta(days=1)
    for st in stations:
        days = sorted({date.fromisoformat(k[2]) for k in latest if k[0] == st.code})
        days = [d for d in days if d < cutoff]
        if not days:
            continue
        actuals_raw = await fetch_cli(st.icao, sorted({d.year for d in days}), ws.user_agent)
        for kind in kinds:
            resid: dict[str, list[float]] = defaultdict(list)
            for (code, k, day_s, bucket), r in latest.items():
                if code != st.code or k != kind.value:
                    continue
                hi, lo = actuals_raw.get(date.fromisoformat(day_s), (None, None))
                actual = hi if kind is Kind.HIGH else lo
                if actual is not None and r.get("mu") is not None:
                    resid[bucket].append(actual - float(r["mu"]))
            cal = store.get(st.code, kind)
            pooled = resid["d0"] + resid["d1"]
            applied = len(pooled) >= min_n
            if applied:
                cal.bias = round(cal.bias + trimmed_mean(pooled), 2)
                for b, vals in resid.items():
                    if len(vals) >= min_n:
                        cal.sigma[b] = round(max(SIGMA_FLOOR[b], robust_std(vals)), 2)
                cal.n = len(pooled)
                cal.method = "log"
                store.put(st.code, kind, cal)
            out.append(
                CalibrationOutcome(
                    st.code,
                    kind.value,
                    len(pooled),
                    cal.source_bias,
                    cal.bias,
                    cal.sigma,
                    applied,
                    "" if applied else f"need >= {min_n} logged days",
                )
            )
    store.save()
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BucketScore:
    n: int = 0
    brier_model: float = 0.0
    brier_market: float = 0.0
    logloss_model: float = 0.0
    logloss_market: float = 0.0


@dataclass(slots=True)
class EvalReport:
    buckets: dict[str, BucketScore] = field(default_factory=dict)
    reliability: list[tuple[float, float, int]] = field(default_factory=list)
    signals: int = 0
    signal_pnl_cents: float = 0.0
    signal_wins: int = 0
    tickers_scored: int = 0


async def _results(client: KalshiClient, tickers: set[str], cache_path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for t in sorted(tickers - set(cache)):
        try:
            m = await client.get_market(t)
        except Exception as exc:
            logger.debug("result lookup %s failed: %s", t, exc)
            continue
        if m.result in ("yes", "no"):
            cache[t] = m.result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def score_rows(
    rows: list[dict[str, Any]],
    results: dict[str, str],
    min_edge: float,
    fee_rate: float,
) -> EvalReport:
    rep = EvalReport()
    eps = 1e-3
    bins: dict[int, list[float]] = defaultdict(list)
    seen_signal: set[str] = set()
    scored: set[str] = set()
    for r in rows:
        bucket = "intraday" if r.get("intraday") else str(r.get("bucket"))
        for b in r.get("brackets", []):
            res = results.get(b["ticker"])
            if res is None:
                continue
            y = 1.0 if res == "yes" else 0.0
            p = float(b["p"])
            bid, ask = b.get("yes_bid"), b.get("yes_ask")
            if not (bid and ask and ask > bid):
                continue
            q = (bid + ask) / 200.0
            s = rep.buckets.setdefault(bucket, BucketScore())
            s.n += 1
            s.brier_model += (p - y) ** 2
            s.brier_market += (q - y) ** 2
            pc, qc = min(max(p, eps), 1 - eps), min(max(q, eps), 1 - eps)
            s.logloss_model -= y * math.log(pc) + (1 - y) * math.log(1 - pc)
            s.logloss_market -= y * math.log(qc) + (1 - y) * math.log(1 - qc)
            bins[min(9, int(p * 10))].append(y)
            scored.add(b["ticker"])

            # First qualifying signal per ticker, held to settlement, 1 contract.
            if b["ticker"] in seen_signal:
                continue
            no_ask = b.get("no_ask") or (100 - bid)
            e_yes = p - ask / 100 - fee_per_contract(ask, fee_rate)
            e_no = (1 - p) - no_ask / 100 - fee_per_contract(no_ask, fee_rate)
            if max(e_yes, e_no) >= min_edge:
                seen_signal.add(b["ticker"])
                rep.signals += 1
                if e_yes >= e_no:
                    pnl = (100 if y else 0) - ask - 100 * fee_per_contract(ask, fee_rate)
                else:
                    pnl = (0 if y else 100) - no_ask - 100 * fee_per_contract(no_ask, fee_rate)
                rep.signal_pnl_cents += pnl
                rep.signal_wins += int(pnl > 0)
    for s in rep.buckets.values():
        if s.n:
            s.brier_model /= s.n
            s.brier_market /= s.n
            s.logloss_model /= s.n
            s.logloss_market /= s.n
    rep.reliability = [
        ((i + 0.5) / 10, float(np.mean(v)), len(v)) for i, v in sorted(bins.items()) if v
    ]
    rep.tickers_scored = len(scored)
    return rep


async def evaluate(
    client: KalshiClient,
    datalog: DataLog,
    data_dir: Path,
    ws: WeatherSettings,
) -> EvalReport:
    rows = list(datalog.rows("model"))
    cutoff = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    rows = [r for r in rows if r.get("day", "9999") < cutoff]
    tickers = {b["ticker"] for r in rows for b in r.get("brackets", [])}
    results = await _results(client, tickers, data_dir / "results.json")
    return score_rows(rows, results, ws.min_edge, ws.fee_rate)
