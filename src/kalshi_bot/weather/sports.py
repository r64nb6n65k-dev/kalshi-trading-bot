"""Sports: bet game-winner markets once the live score has decided the game.

Same idea as the weather bot: no predicting, no beating Kalshi's traders.
The bot reads ESPN's free live scoreboard and only bets when a game is over
or the lead can't realistically be overcome in the time left:

* Final: ESPN says the game is over. Bet the winner (99.5%).
* Basketball (NBA, WNBA, college): Bill James's "safe lead" rule, with the
  lead minus 3, squared, more than the seconds left.
* Football (NFL, college): 17+ points with 7 minutes or less left, or 9+
  points with 100 seconds or less left.
* Hockey: 3+ goals with 10 minutes or less left, or 2+ goals with 90 seconds
  or less left.
* Baseball: 4+ runs in the 9th or later, or 6+ runs in the 8th.
* Soccer: 2+ goals from the 85th minute.

Decided games get YES on the leader's market and NO on the other side (and
NO on "Tie" in soccer), at 97% confidence. The same price rules as weather
apply: Kalshi must also price our side at 60c+, and we never pay above 97c.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.weather.live import Verdict


@dataclass(frozen=True, slots=True)
class League:
    code: str
    espn_path: str  # e.g. "basketball/nba"
    sport: str  # basketball | football | hockey | baseball | soccer
    periods: int = 4
    period_seconds: int = 720


LEAGUES: dict[str, League] = {
    "NFL": League("NFL", "football/nfl", "football", 4, 900),
    "NCAAF": League("NCAAF", "football/college-football", "football", 4, 900),
    "NBA": League("NBA", "basketball/nba", "basketball", 4, 720),
    "WNBA": League("WNBA", "basketball/wnba", "basketball", 4, 600),
    "NCAAMB": League("NCAAMB", "basketball/mens-college-basketball", "basketball", 2, 1200),
    "NCAAWB": League("NCAAWB", "basketball/womens-college-basketball", "basketball", 4, 600),
    "NHL": League("NHL", "hockey/nhl", "hockey", 3, 1200),
    "MLB": League("MLB", "baseball/mlb", "baseball", 9, 0),
    "EPL": League("EPL", "soccer/eng.1", "soccer", 2, 2700),
    "MLS": League("MLS", "soccer/usa.1", "soccer", 2, 2700),
    "UCL": League("UCL", "soccer/uefa.champions", "soccer", 2, 2700),
    "LALIGA": League("LALIGA", "soccer/esp.1", "soccer", 2, 2700),
    "SERIEA": League("SERIEA", "soccer/ita.1", "soccer", 2, 2700),
    "BUNDESLIGA": League("BUNDESLIGA", "soccer/ger.1", "soccer", 2, 2700),
    "LIGUE1": League("LIGUE1", "soccer/fra.1", "soccer", 2, 2700),
}

# ESPN abbreviation -> other codes Kalshi uses for the same team.
ALIASES: dict[str, tuple[str, ...]] = {
    "GS": ("GSW",), "NY": ("NYK", "NYR", "NYY", "NYM", "NYG", "NYJ"), "SA": ("SAS",),
    "NO": ("NOP", "NOS"), "UTAH": ("UTA",), "WSH": ("WAS",), "LA": ("LAR", "LAK", "LAD"),
    "LAR": ("LA",), "JAX": ("JAC",), "TB": ("TBL", "TBB"), "NJ": ("NJD",), "SJ": ("SJS",),
    "CHW": ("CWS",), "KC": ("KCC", "KCR"), "SF": ("SFG",), "SD": ("SDP",),
    "PHX": ("PHO",), "BKN": ("BRK", "BKN"), "CHA": ("CHO",), "VGK": ("VEG",),
}  # fmt: skip


def league_for(series_ticker: str) -> League | None:
    m = re.fullmatch(r"KX([A-Z0-9]+?)GAME", series_ticker.upper())
    return LEAGUES.get(m.group(1)) if m else None


@dataclass(slots=True)
class Team:
    abbr: str
    names: list[str]
    score: int
    home: bool

    def codes(self) -> set[str]:
        a = self.abbr.upper()
        return {a, *ALIASES.get(a, ())}

    def named(self, text: str) -> bool:
        t = text.lower().strip()
        return bool(t) and any(t == n or t in n or n in t for n in self.names if len(n) >= 3)


@dataclass(slots=True)
class Game:
    state: str  # pre | in | post
    period: int
    clock: float  # seconds left in period (seconds elapsed for soccer)
    detail: str
    day: date  # US Eastern date
    teams: list[Team] = field(default_factory=list)

    def leader(self) -> tuple[Team | None, int]:
        a, b = self.teams
        if a.score == b.score:
            return None, 0
        lead = abs(a.score - b.score)
        return (a if a.score > b.score else b), lead


def _eastern_day(when: datetime) -> date:
    from kalshi_bot.weather.markets import _eastern_offset

    return (when + timedelta(hours=_eastern_offset(when.date()))).date()


def parse_scoreboard(payload: dict[str, Any]) -> list[Game]:
    games: list[Game] = []
    for ev in payload.get("events") or []:
        try:
            comp = (ev.get("competitions") or [{}])[0]
            status = comp.get("status") or ev.get("status") or {}
            stype = status.get("type") or {}
            when = datetime.fromisoformat(str(ev["date"]).replace("Z", "+00:00"))
            teams = []
            for c in comp.get("competitors") or []:
                t = c.get("team") or {}
                names = [
                    str(t.get(k, "")).lower()
                    for k in ("displayName", "shortDisplayName", "name", "location", "nickname")
                    if t.get(k)
                ]
                teams.append(
                    Team(
                        abbr=str(t.get("abbreviation", "")),
                        names=names,
                        score=int(float(c.get("score") or 0)),
                        home=c.get("homeAway") == "home",
                    )
                )
            if len(teams) != 2:
                continue
            games.append(
                Game(
                    state=str(stype.get("state", "pre")),
                    period=int(status.get("period") or 0),
                    clock=float(status.get("clock") or 0.0),
                    detail=str(stype.get("shortDetail") or stype.get("detail") or ""),
                    day=_eastern_day(when.astimezone(UTC)),
                    teams=teams,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return games


_EVENT_RE = re.compile(r"-\d{2}[A-Z]{3}\d{2}(?:\d{4})?([A-Z0-9]+)$")


def match_game(
    event_ticker: str, markets: list[Market], games: list[Game]
) -> tuple[Game, dict[str, Team | None]] | None:
    """Find the ESPN game for a Kalshi game event; map each market ticker to its team.

    ``None`` maps to a tie market. Matching uses the team codes in the event
    ticker (with aliases), then team names in the markets' labels.
    """
    from kalshi_bot.weather.brackets import parse_event_date

    day = parse_event_date(event_ticker)
    m = _EVENT_RE.search(event_ticker.upper())
    codes = m.group(1) if m else ""
    for g in games:
        if day is not None and abs((g.day - day).days) > 1:
            continue
        a, b = g.teams
        ok = any(
            (codes[:i] in x.codes() and codes[i:] in y.codes())
            for i in range(2, max(2, len(codes) - 1))
            for x, y in ((a, b), (b, a))
        )
        mapping: dict[str, Team | None] = {}
        for mk in markets:
            suffix = mk.ticker.rsplit("-", 1)[-1].upper()
            label = mk.yes_sub_title or mk.subtitle or ""
            if suffix in ("TIE", "DRAW") or label.lower() in ("tie", "draw"):
                mapping[mk.ticker] = None
            elif suffix in a.codes():
                mapping[mk.ticker] = a
            elif suffix in b.codes():
                mapping[mk.ticker] = b
            elif a.named(label) != b.named(label):  # name matches exactly one team
                mapping[mk.ticker] = a if a.named(label) else b
        teams_seen = {id(t) for t in mapping.values() if t is not None}
        if not ok and teams_seen != {id(a), id(b)}:
            continue
        if mapping:
            return g, mapping
    return None


def seconds_left(league: League, g: Game) -> float:
    """Regulation time left (overtime: time left in the period)."""
    if g.period > league.periods:
        return g.clock
    return g.clock + (league.periods - g.period) * league.period_seconds


def decided(league: League, g: Game) -> tuple[Team | None, float, str] | None:
    """(winner or None for a tie, confidence, why) if the game is decided."""
    leader, lead = g.leader()
    score = f"{g.teams[0].abbr} {g.teams[0].score}-{g.teams[1].score} {g.teams[1].abbr}"
    if g.state == "post":
        return leader, 0.995, f"final: {score}"
    if g.state != "in" or leader is None:
        return None
    if g.clock <= 0 and league.sport not in ("soccer", "baseball"):
        return None  # between periods: ESPN can show the next period at 0:00
    why = f"{score}, {g.detail}"
    s = league.sport
    if s == "basketball":
        t = seconds_left(league, g)
        if lead >= 5 and (lead - 3) ** 2 > t:
            return leader, 0.97, why
    elif s == "football":
        t = seconds_left(league, g)
        if (lead >= 17 and t <= 420) or (lead >= 9 and t <= 100):
            return leader, 0.97, why
    elif s == "hockey":
        if g.period > league.periods:
            return None  # overtime / shootout
        t = seconds_left(league, g)
        if (lead >= 3 and t <= 600) or (lead >= 2 and t <= 90):
            return leader, 0.97, why
    elif s == "baseball":
        if (g.period >= 9 and lead >= 4) or (g.period == 8 and lead >= 6):
            return leader, 0.97, why
    elif s == "soccer":
        minute = g.clock / 60.0
        if g.period >= 2 and minute >= 85 and lead >= 2:
            return leader, 0.97, why
    return None


def verdicts_for(
    mapping: dict[str, Team | None], outcome: tuple[Team | None, float, str]
) -> dict[str, Verdict]:
    winner, p, why = outcome
    out: dict[str, Verdict] = {}
    for ticker, team in mapping.items():
        if winner is None:  # finished level: only the tie market wins
            out[ticker] = Verdict(Side.YES if team is None else Side.NO, p, why)
        else:
            out[ticker] = Verdict(Side.YES if team is winner else Side.NO, p, why)
    return out


class EspnClient:
    """ESPN's public scoreboard (no key). Cached for ``ttl`` seconds per league."""

    BASE = "https://site.api.espn.com/apis/site/v2/sports"

    def __init__(self, ttl: float = 15.0, timeout: float = 15.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._cache: dict[str, tuple[float, list[Game]]] = {}
        self.ttl = ttl

    async def aclose(self) -> None:
        await self._client.aclose()

    async def games(self, league: League) -> list[Game]:
        hit = self._cache.get(league.espn_path)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        resp = await self._client.get(f"{self.BASE}/{league.espn_path}/scoreboard")
        resp.raise_for_status()
        games = parse_scoreboard(resp.json())
        self._cache[league.espn_path] = (time.monotonic(), games)
        return games
