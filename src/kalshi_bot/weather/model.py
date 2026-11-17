"""Probability distribution for a station's daily high/low, priced per bracket.

How a distribution is built (Monte Carlo, whole-°F outcomes):

1. **Center.** Each source gives a forecast extreme for the climate day: each
   ensemble's member mean, deterministic models (NBM, GFS, ECMWF) and the NWS
   gridpoint forecast. Each is bias-corrected (see :mod:`calibration`) and
   they are combined as a weighted average, then a station-level bias is
   added.
2. **Spread.** Ensemble members, re-centered on that blend, give the shape.
   Each sample also gets kernel noise so the total spread is at least the
   calibrated error for this lead time. A ``tail_weight`` share of samples
   uses double spread so rare outcomes are never priced at zero.
3. **Same-day conditioning.** Once the climate day has started, the final
   value is ``max(observed so far, remaining-hours forecast)`` for highs (min
   for lows). The remaining-hours forecast is nudged by how far today's
   latest observation is running from the forecast.
4. Samples are rounded to whole °F, as the climate report is.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from kalshi_bot.weather.brackets import Bracket
from kalshi_bot.weather.calibration import KindCalibration, lead_bucket
from kalshi_bot.weather.forecasts import ForecastBundle, HourlySet, window_extreme
from kalshi_bot.weather.observations import ObsSummary
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind, Station, climate_day_window


@dataclass(slots=True)
class ModelResult:
    station: str
    kind: Kind
    day: date
    lead_hours: float
    bucket: str
    mu: float
    sigma_cal: float
    ens_std: float
    n_members: int
    intraday: bool
    centers: dict[str, float]
    values: np.ndarray = field(repr=False)
    obs_lo: float | None = None
    obs_hi: float | None = None
    anomaly: float | None = None
    remaining_hours: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        return float(self.values.std())

    def prob(self, bracket: Bracket) -> float:
        mask = np.ones(self.values.shape[0], dtype=bool)
        if bracket.lo is not None:
            mask &= self.values >= bracket.lo
        if bracket.hi is not None:
            mask &= self.values <= bracket.hi
        return float(mask.mean())

    def summary(self) -> dict[str, object]:
        return {
            "station": self.station,
            "kind": self.kind.value,
            "day": self.day.isoformat(),
            "lead_hours": round(self.lead_hours, 2),
            "bucket": self.bucket,
            "mu": round(self.mu, 2),
            "model_mean": round(self.mean, 2),
            "model_std": round(self.std, 2),
            "sigma_cal": round(self.sigma_cal, 2),
            "ens_std": round(self.ens_std, 2),
            "n_members": self.n_members,
            "intraday": self.intraday,
            "obs_lo": None if self.obs_lo is None else round(self.obs_lo, 2),
            "obs_hi": None if self.obs_hi is None else round(self.obs_hi, 2),
            "anomaly": None if self.anomaly is None else round(self.anomaly, 2),
            "remaining_hours": self.remaining_hours,
            "centers": {k: round(v, 2) for k, v in self.centers.items()},
            "notes": self.notes,
        }


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def _source_weight(src: str, ws: WeatherSettings) -> float:
    if src == "nws":
        return ws.nws_weight
    if "nbm" in src:
        return ws.nbm_weight
    if src.startswith("ens:"):
        return ws.ensemble_weight
    return ws.deterministic_weight


def _sample_groups(
    rng: np.random.Generator,
    groups: list[np.ndarray],
    n: int,
) -> np.ndarray:
    """Equal weight per group, uniform member within group."""
    g = rng.integers(len(groups), size=n)
    out = np.empty(n)
    for i, arr in enumerate(groups):
        sel = g == i
        k = int(sel.sum())
        if k:
            out[sel] = arr[rng.integers(arr.shape[0], size=k)]
    return out


def _value_near(hs: HourlySet, t: datetime) -> float | None:
    ts = int(t.timestamp())
    idx = int(np.argmin(np.abs(hs.times - ts)))
    if abs(int(hs.times[idx]) - ts) > 3600:
        return None
    col = hs.values[:, idx]
    col = col[np.isfinite(col)]
    return float(col.mean()) if col.size else None


def build_distribution(
    station: Station,
    kind: Kind,
    day: date,
    now: datetime,
    bundle: ForecastBundle,
    calib: KindCalibration,
    ws: WeatherSettings,
    obs: ObsSummary | None = None,
) -> ModelResult | None:
    """Return the outcome distribution, or ``None`` if there is no usable data."""
    start, end = climate_day_window(station, day)
    lead_h = (end - now).total_seconds() / 3600.0
    if lead_h <= 0:
        return None
    bucket = lead_bucket(lead_h)
    sigma_cal = calib.sigma_for(bucket)
    is_high = kind is Kind.HIGH
    notes: list[str] = []

    # --- 1. centers -------------------------------------------------------
    centers: dict[str, float] = {}
    weights: dict[str, float] = {}
    ens_groups: list[np.ndarray] = []
    for hs in bundle.hourly:
        ext = window_extreme(hs, start, end, is_high)
        ext = ext[np.isfinite(ext)]
        if ext.size == 0:
            continue
        adj = ext + calib.source_offset(hs.source, kind)
        centers[hs.source] = float(adj.mean())
        weights[hs.source] = _source_weight(hs.source, ws)
        if hs.is_ensemble and adj.size >= 5:
            ens_groups.append(adj)
    nws_val = (bundle.nws_highs if is_high else bundle.nws_lows).get(day)
    if nws_val is not None:
        centers["nws"] = nws_val + calib.source_offset("nws", kind)
        weights["nws"] = ws.nws_weight
    total_w = sum(weights[s] for s in centers)
    if not centers or total_w <= 0:
        return None
    mu = sum(centers[s] * weights[s] for s in centers) / total_w + calib.bias

    # --- 2. day-ahead samples ---------------------------------------------
    rng = np.random.default_rng(_seed(station.code, kind.value, day, int(now.timestamp() // 60)))
    n = ws.samples
    if ens_groups:
        gmeans = np.array([g.mean() for g in ens_groups])
        ens_var = float(np.mean([g.var() for g in ens_groups]) + gmeans.var())
        shift = mu - float(gmeans.mean())
        base = _sample_groups(rng, ens_groups, n) + shift
        n_members = int(sum(g.size for g in ens_groups))
    else:
        ens_var, shift, n_members = 0.0, 0.0, 0
        base = np.full(n, mu)
        notes.append("no ensemble; normal around blended center")
    sigma_k = math.sqrt(max(sigma_cal**2 - ens_var, (0.5 * sigma_cal) ** 2))
    tail = rng.random(n) < ws.tail_weight
    sig = np.where(tail, 2.0 * sigma_cal, sigma_k)
    day_samples = base + rng.standard_normal(n) * sig

    result_kwargs: dict[str, object] = {}
    final = day_samples
    intraday = now >= start

    # --- 3. same-day conditioning -----------------------------------------
    if intraday:
        bounds: tuple[float, float] | None = None
        if obs is not None:
            final_ext = obs.final_max if is_high else obs.final_min
            if final_ext is not None:
                bounds = final_ext
                notes.append("24h extreme group reported")
            elif is_high and obs.max_lo is not None and obs.max_hi is not None:
                bounds = (obs.max_lo, obs.max_hi)
            elif not is_high and obs.min_lo is not None and obs.min_hi is not None:
                bounds = (obs.min_lo, obs.min_hi)
        if bounds is None:
            notes.append("intraday without observations; using full-day forecast")
        else:
            lo_b, hi_b = min(bounds), max(bounds)
            m_samples = rng.uniform(lo_b, hi_b, n) if hi_b > lo_b else np.full(n, lo_b)
            result_kwargs.update(obs_lo=lo_b, obs_hi=hi_b)

            after = now.timestamp() + 1
            rem_groups: list[np.ndarray] = []
            rem_hours = 0
            anomaly_vals: list[float] = []
            for hs in bundle.hourly:
                mask = (hs.times >= after) & (hs.times < end.timestamp())
                rem_hours = max(rem_hours, int(mask.sum()))
                if not mask.any():
                    continue
                vals = hs.values[:, mask]
                with np.errstate(all="ignore"):
                    ext = np.nanmax(vals, axis=1) if is_high else np.nanmin(vals, axis=1)
                ext = ext[np.isfinite(ext)] + calib.source_offset(hs.source, kind)
                if ext.size:
                    rem_groups.append(ext)
                if obs is not None and obs.latest_time is not None:
                    v = _value_near(hs, obs.latest_time)
                    if v is not None:
                        anomaly_vals.append(v)
            if "24h extreme group reported" in notes:
                rem_hours = 0
            result_kwargs["remaining_hours"] = rem_hours

            if rem_hours == 0:
                final = m_samples
            elif not rem_groups:
                notes.append("remaining-hours forecast missing; using full-day forecast")
                final = (
                    np.maximum(m_samples, day_samples)
                    if is_high
                    else np.minimum(m_samples, day_samples)
                )
            else:
                anomaly: float | None = None
                if (
                    obs is not None
                    and obs.latest_temp_f is not None
                    and obs.latest_time is not None
                    and anomaly_vals
                    and (now - obs.latest_time).total_seconds() < 2 * 3600
                ):
                    anomaly = obs.latest_temp_f - float(np.mean(anomaly_vals))
                    nudge = ws.obs_persistence * anomaly
                    result_kwargs["anomaly"] = anomaly
                else:
                    nudge = shift
                    notes.append("no recent observation; remaining hours use forecast shift")
                sig_rem = max(0.7, calib.sigma_for("d0") * math.sqrt(min(rem_hours, 24) / 24.0))
                y = _sample_groups(rng, rem_groups, n) + nudge
                y += rng.standard_normal(n) * np.where(tail, 2.0 * sig_rem, sig_rem)
                final = np.maximum(m_samples, y) if is_high else np.minimum(m_samples, y)

    values = np.floor(final + 0.5).astype(np.int64)
    return ModelResult(
        station=station.code,
        kind=kind,
        day=day,
        lead_hours=lead_h,
        bucket=bucket,
        mu=mu,
        sigma_cal=sigma_cal,
        ens_std=math.sqrt(ens_var),
        n_members=n_members,
        intraday=intraday,
        centers=centers,
        values=values,
        notes=notes,
        **result_kwargs,  # type: ignore[arg-type]
    )
