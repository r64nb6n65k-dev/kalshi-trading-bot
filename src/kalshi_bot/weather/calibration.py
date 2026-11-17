"""Per-station calibration: forecast bias by source and error spread by lead.

Stored as JSON at ``<data_dir>/calibration.json``. Until you run
``kalshi-bot weather-calibrate``, conservative priors are used:

* Open-Meteo hourly-derived extremes miss the true peak between hours, so
  they get a +0.8 °F prior for highs and -0.8 °F for lows.
* NWS gridpoint extremes are forecasts of the true extreme, so their prior is 0.
* Spread: 1.8 / 2.6 / 3.4 °F for the same day / next day / two or more days out.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from kalshi_bot.weather.stations import Kind

DEFAULT_SIGMA = {"d0": 1.8, "d1": 2.6, "d2": 3.4}
SIGMA_FLOOR = {"d0": 1.0, "d1": 1.6, "d2": 2.2}


def lead_bucket(lead_hours: float) -> str:
    """Hours until the climate day ends -> ``d0``/``d1``/``d2``."""
    if lead_hours <= 30:
        return "d0"
    if lead_hours <= 54:
        return "d1"
    return "d2"


def default_source_bias(source: str, kind: Kind) -> float:
    if source.startswith(("ens:", "det:")):
        return 0.8 if kind is Kind.HIGH else -0.8
    return 0.0


@dataclass(slots=True)
class KindCalibration:
    bias: float = 0.0
    sigma: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGMA))
    source_bias: dict[str, float] = field(default_factory=dict)
    n: int = 0
    method: str = "prior"
    updated: str = ""

    def source_offset(self, source: str, kind: Kind) -> float:
        # Calibrations are keyed by model id without the ens/det prefix so a
        # deterministic-history fit also applies to that model's ensemble.
        bare = source.split(":", 1)[-1]
        if source in self.source_bias:
            return self.source_bias[source]
        if bare in self.source_bias:
            return self.source_bias[bare]
        return default_source_bias(source, kind)

    def sigma_for(self, bucket: str) -> float:
        return max(SIGMA_FLOOR.get(bucket, 1.0), self.sigma.get(bucket, DEFAULT_SIGMA[bucket]))


class CalibrationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, KindCalibration]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for st, kinds in raw.items():
            self._data[st] = {k: KindCalibration(**v) for k, v in kinds.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {st: {k: asdict(v) for k, v in kinds.items()} for st, kinds in self._data.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, station: str, kind: Kind) -> KindCalibration:
        return self._data.get(station, {}).get(kind.value) or KindCalibration()

    def put(self, station: str, kind: Kind, cal: KindCalibration) -> None:
        cal.updated = datetime.now(UTC).isoformat(timespec="seconds")
        self._data.setdefault(station, {})[kind.value] = cal


def trimmed_mean(values: list[float], trim: float = 0.1) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    k = int(len(arr) * trim)
    core = arr[k : len(arr) - k] if len(arr) - 2 * k > 0 else arr
    return float(core.mean())


def robust_std(values: list[float]) -> float:
    """Std estimate resistant to a few busted forecasts (1.4826 * MAD),
    blended 50/50 with the plain std so real fat tails still count."""
    arr = np.asarray(values, dtype=float)
    mad = float(np.median(np.abs(arr - np.median(arr)))) * 1.4826
    return 0.5 * mad + 0.5 * float(arr.std(ddof=1))


def fit_history(
    forecasts: dict[date, dict[str, float]],
    actuals: dict[date, float],
    min_n: int = 30,
) -> tuple[dict[str, float], list[float]]:
    """Fit per-source bias and return blended residuals.

    ``forecasts[day][source]`` is a forecast extreme; ``actuals[day]`` the CLI
    value. Returns ``(biases, residuals)`` where each residual is the actual
    minus the equal-weight mean of all bias-corrected sources that day.
    """
    errors: dict[str, list[float]] = {}
    for day, row in forecasts.items():
        actual = actuals.get(day)
        if actual is None:
            continue
        for src, f in row.items():
            if math.isfinite(f):
                errors.setdefault(src, []).append(actual - f)
    biases = {
        src: round(trimmed_mean(errs), 2) for src, errs in errors.items() if len(errs) >= min_n
    }
    residuals: list[float] = []
    for day, row in sorted(forecasts.items()):
        actual = actuals.get(day)
        corrected = [f + biases[s] for s, f in row.items() if s in biases and math.isfinite(f)]
        if actual is not None and corrected:
            residuals.append(actual - float(np.mean(corrected)))
    return biases, residuals
