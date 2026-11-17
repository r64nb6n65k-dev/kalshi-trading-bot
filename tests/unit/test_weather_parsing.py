"""Brackets, stations, observations, NWS grid and fee parsing."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from kalshi_bot.weather.brackets import (
    Bracket,
    bracket_from_strikes,
    bracket_from_text,
    group_events,
    parse_bracket,
    parse_event_date,
    validate_partition,
)
from kalshi_bot.weather.climate import parse_cli
from kalshi_bot.weather.fees import fee_cents
from kalshi_bot.weather.nws import extremes_by_climate_day, parse_valid_time
from kalshi_bot.weather.observations import parse_observation, summarize
from kalshi_bot.weather.stations import (
    STATIONS,
    climate_day_window,
    climate_today,
    resolve_stations,
)
from tests.unit.weather_helpers import ladder, market

UTC = UTC


# --- brackets ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("65° to 66°", Bracket(65, 66)),
        ("65-66°", Bracket(65, 66)),
        ("73° or above", Bracket(73, None)),
        ("64° or below", Bracket(None, 64)),
        ("-5° to -4°", Bracket(-5, -4)),
        ("greater than 80°", Bracket(81, None)),
        ("nonsense", None),
    ],
)
def test_bracket_from_text(text: str, expected: Bracket | None) -> None:
    assert bracket_from_text(text) == expected


def test_bracket_from_strikes() -> None:
    assert bracket_from_strikes(
        market("X", "", strike_type="between", floor=65, cap=66)
    ) == Bracket(65, 66)
    assert bracket_from_strikes(market("X", "", strike_type="greater", floor=72)) == Bracket(
        73, None
    )
    assert bracket_from_strikes(market("X", "", strike_type="less", cap=64)) == Bracket(None, 63)
    assert bracket_from_strikes(
        market("X", "", strike_type="greater_or_equal", floor=73)
    ) == Bracket(73, None)


def test_label_and_strike_disagreement_is_rejected() -> None:
    m = market("X", "73° or above", strike_type="greater", floor=73)  # strikes say >=74
    br, problem = parse_bracket(m)
    assert br is None and "label" in problem


def test_partition_validation() -> None:
    assert validate_partition([Bracket(None, 63), Bracket(64, 65), Bracket(66, None)]) == ""
    assert "gap" in validate_partition([Bracket(None, 63), Bracket(65, 66), Bracket(67, None)])
    assert "tail" in validate_partition([Bracket(64, 65), Bracket(66, None)])


def test_group_events_valid_and_dated() -> None:
    evs = group_events(ladder())
    assert len(evs) == 1
    ev = evs[0]
    assert ev.valid, ev.problem
    assert ev.day == date(2026, 10, 28)
    assert ev.brackets[0].bracket == Bracket(None, 63)
    assert ev.brackets[-1].bracket == Bracket(76, None)


def test_group_events_flags_gaps() -> None:
    rows = [m for m in ladder() if not m.ticker.endswith("B68.5")]
    ev = group_events(rows)[0]
    assert not ev.valid and "gap" in ev.problem


def test_parse_event_date() -> None:
    assert parse_event_date("KXHIGHNY-26OCT28") == date(2026, 10, 28)
    assert parse_event_date("KXHIGHNY-26OCT28-B65.5") == date(2026, 10, 28)
    assert parse_event_date("KXHIGHNY") is None


# --- stations ---------------------------------------------------------------
def test_climate_day_is_fixed_utc_window_even_in_dst() -> None:
    nyc = STATIONS["NYC"]
    start, end = climate_day_window(nyc, date(2026, 7, 4))  # EDT in effect
    assert start == datetime(2026, 7, 4, 5, tzinfo=UTC)  # 1:00 AM EDT
    assert end == datetime(2026, 7, 5, 5, tzinfo=UTC)
    # 12:30 AM EDT on Jul 5 still belongs to the Jul 4 climate day.
    assert climate_today(nyc, datetime(2026, 7, 5, 4, 30, tzinfo=UTC)) == date(2026, 7, 4)


def test_series_override() -> None:
    (st,) = resolve_stations(["nyc"], {"NYC:low": "KXLOWNEW"})
    assert st.low_series == "KXLOWNEW" and st.high_series == "KXHIGHNY"
    with pytest.raises(ValueError):
        resolve_stations(["XXX"])


# --- observations -----------------------------------------------------------
def _obs(ts: str, raw: str, temp: float | None = None) -> dict[str, object]:
    return {
        "timestamp": ts,
        "rawMessage": raw,
        "temperature": {"value": temp, "unitCode": "wmoUnit:degC"},
    }


def test_parse_t_group_and_six_hour_max() -> None:
    rd = parse_observation(
        _obs(
            "2026-10-28T18:51:00+00:00",
            "KNYC 281851Z 27008KT 10SM CLR 21/09 A3002 RMK AO2 SLP165 T02110094 10217 20150 58012",
            21.0,
        )
    )
    kinds = {r.kind: r for r in rd}
    assert kinds["inst"].lo_f == pytest.approx(21.05 * 0 + (21.1 - 0.05) * 1.8 + 32)
    assert kinds["max6"].mid_f == pytest.approx(21.7 * 1.8 + 32)
    assert kinds["min6"].mid_f == pytest.approx(15.0 * 1.8 + 32)


def test_peak_wind_not_mistaken_for_min_group() -> None:
    rd = parse_observation(
        _obs(
            "2026-10-28T18:51:00+00:00",
            "KNYC 281851Z 21015G28KT 10SM CLR 21/09 A3002 RMK AO2 PK WND 21028/1832 T02110094",
        )
    )
    assert [r.kind for r in rd] == ["inst"]


def test_negative_temps_and_24h_group() -> None:
    rd = parse_observation(
        _obs(
            "2026-01-15T04:51:00+00:00",
            "KNYC 150451Z 00000KT 10SM CLR M05/M10 A3030 RMK AO2 T10501100 401001055",
        )
    )
    k = {r.kind: r for r in rd}
    assert k["inst"].mid_f == pytest.approx(-5.0 * 1.8 + 32)
    assert k["max24"].mid_f == pytest.approx(10.0 * 1.8 + 32)
    assert k["min24"].mid_f == pytest.approx(-5.5 * 1.8 + 32)


def test_whole_degree_obs_get_wide_interval() -> None:
    (r,) = parse_observation(_obs("2026-10-28T18:55:00+00:00", "", 21.0))
    assert r.hi_f - r.lo_f == pytest.approx(1.8)


def test_summarize_uses_only_in_window_data() -> None:
    nyc = STATIONS["NYC"]
    start, end = climate_day_window(nyc, date(2026, 10, 28))
    rd = []
    # 6h max ending 10:51Z starts 04:51Z (9 min early) -> allowed by tolerance
    rd += parse_observation(_obs("2026-10-28T10:51:00+00:00", "X RMK AO2 T01000050 10120"))
    # 6h max ending 08:51Z starts 02:51Z -> previous day mostly, excluded
    rd += parse_observation(_obs("2026-10-28T08:51:00+00:00", "X RMK AO2 T00900050 10250"))
    # instantaneous before window start -> excluded
    rd += parse_observation(_obs("2026-10-28T04:51:00+00:00", "X RMK AO2 T02000050"))
    s = summarize(rd, start, end)
    assert s.max_lo == pytest.approx((12.0 - 0.05) * 1.8 + 32)
    assert s.latest_time == datetime(2026, 10, 28, 10, 51, tzinfo=UTC)


# --- NWS grid ---------------------------------------------------------------
def test_grid_extremes_map_to_climate_days() -> None:
    nyc = STATIONS["NYC"]
    props = {
        "maxTemperature": {
            "uom": "wmoUnit:degC",
            "values": [
                {"validTime": "2026-10-28T11:00:00+00:00/PT13H", "value": 20.0},
            ],
        },
        "minTemperature": {
            "uom": "wmoUnit:degC",
            "values": [
                {"validTime": "2026-10-28T23:00:00+00:00/PT14H", "value": 10.0},
            ],
        },
    }
    highs = extremes_by_climate_day(props, nyc, "maxTemperature")
    lows = extremes_by_climate_day(props, nyc, "minTemperature")
    assert highs == {date(2026, 10, 28): pytest.approx(68.0)}
    # Overnight period ending the morning of the 29th belongs to the 29th.
    assert lows == {date(2026, 10, 29): pytest.approx(50.0)}
    assert parse_valid_time("2026-10-28T11:00:00+00:00/P1DT2H") is not None


def test_parse_cli_handles_missing_and_trace() -> None:
    out = parse_cli(
        {
            "results": [
                {"valid": "2026-10-27", "high": 66, "low": 51},
                {"valid": "2026-10-28", "high": "M", "low": 50},
            ]
        }
    )
    assert out[date(2026, 10, 27)] == (66.0, 51.0)
    assert out[date(2026, 10, 28)] == (None, 50.0)


# --- fees ------------------------------------------------------------------
def test_fee_formula() -> None:
    assert fee_cents(50, 100) == 175  # 0.07 * 100 * 0.25 = $1.75
    assert fee_cents(1, 1) == 1  # rounds up to a full cent
    assert fee_cents(50, 0) == 0
