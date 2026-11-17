"""Historical NWS climate-report values (the settlement truth) via IEM.

The Iowa Environmental Mesonet archives parsed NWS CLI products:
``https://mesonet.agron.iastate.edu/json/cli.py?station=KNYC&year=2026``.
Missing values come back as ``"M"`` and trace as ``"T"``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, str):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_cli(payload: dict[str, Any]) -> dict[date, tuple[float | None, float | None]]:
    out: dict[date, tuple[float | None, float | None]] = {}
    for row in payload.get("results") or []:
        try:
            day = date.fromisoformat(str(row.get("valid"))[:10])
        except ValueError:
            continue
        out[day] = (_num(row.get("high")), _num(row.get("low")))
    return out


async def fetch_cli(
    icao: str,
    years: list[int],
    user_agent: str,
    timeout: float = 30.0,
) -> dict[date, tuple[float | None, float | None]]:
    out: dict[date, tuple[float | None, float | None]] = {}
    async with httpx.AsyncClient(
        timeout=timeout, headers={"User-Agent": user_agent}, follow_redirects=True
    ) as client:
        for year in sorted(set(years)):
            resp = await client.get(IEM_CLI_URL, params={"station": icao, "year": year})
            resp.raise_for_status()
            out.update(parse_cli(resp.json()))
    return out
