"""Weather dashboard: data assembly and HTTP routes."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalshi_bot.weather.dashboard import build_data, start_weather_dashboard
from kalshi_bot.weather.settings import DEFAULT_DATA_DIR, WeatherSettings
from tests.unit.test_weather_engine import _engine
from tests.unit.weather_helpers import ladder


def test_build_data_after_paper_cycle(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", ladder({"70": (2, 5)}), result="yes")
    asyncio.run(eng.cycle())
    d = build_data(eng)
    assert d["mode"] == "paper" and d["cycles"] == 1
    assert d["markets"] and d["markets"][0]["city"].startswith("New York")
    assert d["stats"]["trades"] >= 1
    pos = d["positions"]
    assert pos and pos[0]["label"] == "70-71" and pos[0]["side"] == "YES"
    assert pos[0]["bid"] == 2  # marked at the current YES bid
    assert d["account"]["unrealized_cents"] < 0  # bought at 5c, bid 2c
    json.dumps(d)  # must be JSON-serialisable

    eng.now_fn = lambda: datetime(2026, 10, 29, 12, tzinfo=UTC)
    eng._last_settle = float("-inf")
    asyncio.run(eng._settle_paper())
    d2 = build_data(eng)
    assert d2["stats"]["settled"] >= 1 and d2["stats"]["wins"] >= 1
    assert not d2["positions"]


def test_http_routes(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", ladder({"70": (2, 5)}))
    asyncio.run(eng.cycle())
    server = start_weather_dashboard(eng, port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "Weather Bot" in html
        data = json.loads(urllib.request.urlopen(base + "/api/data").read())
        assert data["mode"] == "paper"
        csv_text = urllib.request.urlopen(base + "/download/trades.csv").read().decode()
        assert csv_text.startswith("t,mode,kind") and "B70.5" in csv_text
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/nope")
    finally:
        server.shutdown()


def test_data_path_prefers_railway_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path / "vol"))
    assert WeatherSettings().data_path() == tmp_path / "vol" / "weather"
    custom = WeatherSettings(data_dir=str(tmp_path / "mine"))
    assert custom.data_path() == tmp_path / "mine"
    monkeypatch.delenv("RAILWAY_VOLUME_MOUNT_PATH")
    monkeypatch.chdir(tmp_path)
    assert WeatherSettings().data_path() == Path(DEFAULT_DATA_DIR)
