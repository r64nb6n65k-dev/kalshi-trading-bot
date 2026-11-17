"""Distribution model: day-ahead, same-day conditioning, lows."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kalshi_bot.weather.brackets import Bracket, group_events
from kalshi_bot.weather.calibration import KindCalibration, lead_bucket
from kalshi_bot.weather.forecasts import ForecastBundle
from kalshi_bot.weather.model import build_distribution
from kalshi_bot.weather.observations import ObsSummary
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind, climate_day_window
from tests.unit.weather_helpers import DAY, NYC, bundle, ladder

UTC = UTC
WS = WeatherSettings(samples=20_000)


def test_lead_buckets() -> None:
    assert lead_bucket(10) == "d0"
    assert lead_bucket(41) == "d1"
    assert lead_bucket(70) == "d2"


def test_day_ahead_distribution_is_centered_and_sums_to_one() -> None:
    now = datetime(2026, 10, 27, 12, tzinfo=UTC)
    res = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(70.0), KindCalibration(), WS)
    assert res is not None
    assert res.bucket == "d1" and not res.intraday
    # Hourly-derived extremes +0.8 prior; NWS gridpoint given as 70.8.
    assert res.mu == pytest.approx(70.8, abs=0.7)
    assert res.mean == pytest.approx(res.mu, abs=0.5)
    assert 2.0 < res.std < 4.0
    ev = group_events(ladder())[0]
    total = sum(res.prob(b.bracket) for b in ev.brackets)
    assert total == pytest.approx(1.0)


def test_deterministic_for_same_minute() -> None:
    now = datetime(2026, 10, 27, 12, tzinfo=UTC)
    a = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(), KindCalibration(), WS)
    b = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(), KindCalibration(), WS)
    assert a is not None and b is not None
    assert a.prob(Bracket(70, 71)) == b.prob(Bracket(70, 71))


def test_calibration_bias_shifts_center() -> None:
    now = datetime(2026, 10, 27, 12, tzinfo=UTC)
    base = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(), KindCalibration(), WS)
    shifted = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(), KindCalibration(bias=2.0), WS)
    assert base is not None and shifted is not None
    assert shifted.mean - base.mean == pytest.approx(2.0, abs=0.3)


def test_locked_high_after_day_is_effectively_over() -> None:
    start, end = climate_day_window(NYC, DAY)
    now = datetime(2026, 10, 29, 4, 30, tzinfo=UTC)  # 11:30 PM EST, no hours left
    obs = ObsSummary(
        window_start=start,
        window_end=end,
        max_lo=79.9,
        max_hi=80.1,
        min_lo=50.0,
        min_hi=50.2,
        latest_time=now,
        latest_temp_f=55.0,
        n_readings=40,
    )
    res = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(70.0), KindCalibration(), WS, obs)
    assert res is not None and res.intraday
    assert res.remaining_hours == 0
    assert res.prob(Bracket(80, 80)) == pytest.approx(1.0)


def test_observed_high_floors_the_distribution() -> None:
    start, end = climate_day_window(NYC, DAY)
    now = datetime(2026, 10, 28, 21, tzinfo=UTC)  # 4 PM EST, peak passed
    obs = ObsSummary(
        window_start=start,
        window_end=end,
        max_lo=74.0,
        max_hi=74.2,
        min_lo=60.0,
        min_hi=60.2,
        latest_time=now,
        latest_temp_f=73.0,
        n_readings=40,
    )
    res = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(70.0), KindCalibration(), WS, obs)
    assert res is not None
    assert res.prob(Bracket(None, 73)) == 0.0  # already observed 74
    assert res.prob(Bracket(74, 75)) > 0.8


def test_warm_anomaly_raises_remaining_forecast() -> None:
    start, end = climate_day_window(NYC, DAY)
    now = datetime(2026, 10, 28, 15, tzinfo=UTC)  # 10 AM EST, before the peak

    def run(latest: float) -> float:
        obs = ObsSummary(
            window_start=start,
            window_end=end,
            max_lo=60.0,
            max_hi=60.2,
            min_lo=55.0,
            min_hi=55.2,
            latest_time=now,
            latest_temp_f=latest,
            n_readings=10,
        )
        r = build_distribution(NYC, Kind.HIGH, DAY, now, bundle(70.0), KindCalibration(), WS, obs)
        assert r is not None and r.anomaly is not None
        return r.mean

    assert run(68.0) - run(64.0) == pytest.approx(0.6 * 4.0, abs=0.4)


def test_low_temperature_distribution() -> None:
    now = datetime(2026, 10, 27, 12, tzinfo=UTC)
    res = build_distribution(NYC, Kind.LOW, DAY, now, bundle(70.0), KindCalibration(), WS)
    assert res is not None
    # Diurnal min = 54; hourly-derived low gets a -0.8 prior; NWS low 53.2.
    assert res.mean == pytest.approx(53.2, abs=0.8)


def test_no_data_returns_none() -> None:
    empty = ForecastBundle(station=NYC, fetched_at=datetime(2026, 10, 27, tzinfo=UTC))
    now = datetime(2026, 10, 27, 12, tzinfo=UTC)
    assert build_distribution(NYC, Kind.HIGH, DAY, now, empty, KindCalibration(), WS) is None


def test_past_day_returns_none() -> None:
    now = datetime(2026, 10, 30, tzinfo=UTC)
    assert build_distribution(NYC, Kind.HIGH, DAY, now, bundle(), KindCalibration(), WS) is None
