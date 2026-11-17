"""Station observations -> bounds on today's observed high / low.

The settlement value is a whole °F from the climate report. Live data comes
in several precisions, and each reading is kept as a ``[lo, hi]`` °F interval:

* METAR ``T`` group (e.g. ``T01720094``): temperature to 0.1 °C.
* 6-hour max/min remark groups (``1snTTT`` / ``2snTTT``): the true extreme
  over the previous 6 hours, to 0.1 °C. These catch peaks that fall between
  hourly readings.
* 24-hour max/min group (``4snTTTsnTTT``), sent near midnight local standard
  time: the climate-day extremes themselves.
* Plain ``temperature`` values without a ``T`` group (e.g. 5-minute data):
  whole °C, so ±0.5 °C (about ±0.9 °F).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

_T_GROUP = re.compile(r"(?<=\s)T([01])(\d{3})(?:[01]\d{3})?(?=\s|$)")
_MAX6 = re.compile(r"(?<=\s)1([01])(\d{3})(?=\s|$)")
_MIN6 = re.compile(r"(?<=\s)2([01])(\d{3})(?=\s|$)")
_EXT24 = re.compile(r"(?<=\s)4([01])(\d{3})([01])(\d{3})(?=\s|$)")

WINDOW_TOLERANCE = timedelta(minutes=10)


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _tenths(sign: str, digits: str) -> float:
    v = int(digits) / 10.0
    return -v if sign == "1" else v


def _interval_f(c: float, half_width_c: float) -> tuple[float, float]:
    return c_to_f(c - half_width_c), c_to_f(c + half_width_c)


@dataclass(slots=True)
class ObsReading:
    """One bound-carrying reading.

    ``kind``: ``inst`` (instantaneous), ``max6``/``min6`` (6h extreme ending at
    ``time``), ``max24``/``min24`` (climate-day extreme).
    """

    time: datetime
    kind: str
    lo_f: float
    hi_f: float

    @property
    def mid_f(self) -> float:
        return (self.lo_f + self.hi_f) / 2.0


def parse_observation(props: dict[str, Any]) -> list[ObsReading]:
    """Parse one api.weather.gov observation ``properties`` object."""
    ts = props.get("timestamp")
    if not ts:
        return []
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return []

    raw = " " + str(props.get("rawMessage") or "") + " "
    rmk = raw.split(" RMK ", 1)[1] if " RMK " in raw else ""
    rmk = " " + rmk + " "
    out: list[ObsReading] = []

    if m := _T_GROUP.search(raw):
        c = _tenths(m.group(1), m.group(2))
        out.append(ObsReading(t, "inst", *_interval_f(c, 0.05)))
    else:
        temp = props.get("temperature") or {}
        value = temp.get("value")
        unit = str(temp.get("unitCode") or "degC")
        if value is not None:
            v = float(value)
            c = (v - 32.0) * 5.0 / 9.0 if unit.endswith("degF") else v
            out.append(ObsReading(t, "inst", *_interval_f(c, 0.5)))

    if rmk.strip():
        if m := _MAX6.search(rmk):
            c = _tenths(m.group(1), m.group(2))
            out.append(ObsReading(t, "max6", *_interval_f(c, 0.05)))
        if m := _MIN6.search(rmk):
            c = _tenths(m.group(1), m.group(2))
            out.append(ObsReading(t, "min6", *_interval_f(c, 0.05)))
        if m := _EXT24.search(rmk):
            cmax = _tenths(m.group(1), m.group(2))
            cmin = _tenths(m.group(3), m.group(4))
            out.append(ObsReading(t, "max24", *_interval_f(cmax, 0.05)))
            out.append(ObsReading(t, "min24", *_interval_f(cmin, 0.05)))
    return out


@dataclass(slots=True)
class ObsSummary:
    """What observations say about the climate day so far."""

    window_start: datetime
    window_end: datetime
    # Interval containing the max / min observed so far (None if no data).
    max_lo: float | None = None
    max_hi: float | None = None
    min_lo: float | None = None
    min_hi: float | None = None
    latest_time: datetime | None = None
    latest_temp_f: float | None = None
    n_readings: int = 0
    final_max: tuple[float, float] | None = None  # from 24h group
    final_min: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)


def summarize(readings: list[ObsReading], start: datetime, end: datetime) -> ObsSummary:
    """Combine readings into bounds for the climate day ``[start, end)``."""
    s = ObsSummary(window_start=start, window_end=end)
    max_los: list[float] = []
    max_his: list[float] = []
    min_los: list[float] = []
    min_his: list[float] = []
    for r in readings:
        if r.kind == "inst" and start <= r.time < end:
            max_los.append(r.lo_f)
            max_his.append(r.hi_f)
            min_los.append(r.lo_f)
            min_his.append(r.hi_f)
            if s.latest_time is None or r.time > s.latest_time:
                s.latest_time, s.latest_temp_f = r.time, r.mid_f
        elif r.kind in ("max6", "min6"):
            w0 = r.time - timedelta(hours=6)
            if w0 >= start - WINDOW_TOLERANCE and r.time <= end + WINDOW_TOLERANCE:
                if r.kind == "max6":
                    max_los.append(r.lo_f)
                    max_his.append(r.hi_f)
                else:
                    min_los.append(r.lo_f)
                    min_his.append(r.hi_f)
        elif r.kind in ("max24", "min24") and abs(r.time - end) <= timedelta(minutes=40):
            if r.kind == "max24":
                s.final_max = (r.lo_f, r.hi_f)
            else:
                s.final_min = (r.lo_f, r.hi_f)
    s.n_readings = len(max_los)
    if max_los:
        s.max_lo, s.max_hi = max(max_los), max(max_his)
        s.min_lo, s.min_hi = min(min_los), min(min_his)
    return s
