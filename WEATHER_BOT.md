# Weather Bot: Kalshi daily high/low temperature markets

This bot prices every bracket of Kalshi's daily temperature markets with a
forecast model, compares those prices with the order book, and trades only
when the edge clears fees by a margin. It starts in **log-only** or **paper**
mode. Live trading takes two explicit switches.

Initial universe: **NYC (Central Park), Chicago (Midway), Austin, Miami,
Denver**, both highs and lows.

---

## 1. Setup

```bash
python -m pip install -e ".[dev]"
cp .env.example .env    # then fill in the weather section
```

Minimum `.env` additions:

```bash
KALSHI_WEATHER__USER_AGENT="kalshi-weather-bot (you@example.com)"   # required by api.weather.gov
```

No new Python dependencies. The bot uses `httpx` and `numpy`, which the repo
already has.

## 2. Recommended rollout

| Step | Command | Why |
|---|---|---|
| 1 | `kalshi-bot weather-check` | Confirms every data source works and that Kalshi's brackets parse. **Run this first.** It shows which Open-Meteo model IDs resolved, and warns if a series ticker or station looks wrong. |
| 2 | `kalshi-bot weather-calibrate --source history --days 180` | Fits each model's warm/cold bias and the forecast error spread per station, using archived forecasts against the official climate reports. |
| 3 | `kalshi-bot weather-show` | Shows model vs. market for every open event right now, with the YES/NO edge after fees. |
| 4 | `kalshi-bot weather-run --log-only` (leave running 2-3 weeks) | Pure data collection: model probabilities and full order-book quotes every 10 min per event. |
| 5 | `kalshi-bot weather-report` | Scores the model against the market on settled events (Brier score and log loss by lead time), with a reliability table and the P&L of the trades the bot *would* have made. |
| 6 | `kalshi-bot weather-calibrate --source log` | Refines station bias and spread from the bot's own forecasts. |
| 7 | `kalshi-bot weather-run` | Paper trading. The ledger in `data/weather/paper_ledger.json` settles against real Kalshi results. |
| 8 | `KALSHI_DRY_RUN=false kalshi-bot weather-run --live` | Real orders, only once steps 5 and 7 look good. |

**The go/no-go gate is step 5.** If `Brier model` isn't clearly below `Brier
market` for a lead bucket, the model has no edge there, so don't trade it.

## Dashboard

`weather-run` serves a phone-friendly dashboard on `$PORT` (Railway sets it;
default 8080). It shows P&L, open bets marked to the current bid, the
model-vs-market table for every open event, recent trades and settlements,
plus CSV downloads. Use `--no-dashboard` to turn it off.

On Railway, attach a **volume**: the bot then keeps its logs, paper ledger and
calibration in `<volume>/weather`, so they survive redeploys. Without one,
every redeploy starts from zero.

## 3. How the model works

Code: `src/kalshi_bot/weather/model.py`

1. **Climate day.** The NWS report covers midnight to midnight *local standard
   time*, so during daylight saving the day runs 1:00 AM to 12:59 AM. The bot
   always uses this window (`stations.climate_day_window`), for forecasts and
   observations alike.
2. **Center.** A weighted blend of:
   * Ensembles (GFS, ECMWF, ICON, GEM via Open-Meteo): the mean of each one's members.
   * NBM and deterministic models (Open-Meteo).
   * The NWS gridpoint max/min forecast (api.weather.gov).

   Each source is bias-corrected per station. Before calibration, the defaults
   are +0.8 °F for hourly-model highs (hourly data misses the peak between
   hours), −0.8 °F for lows, and 0 for NWS.
3. **Spread.** Ensemble members, re-centered on the blend, give the shape.
   Kernel noise brings the total spread up to the calibrated error for the
   lead time. 10% of samples get double spread (fat tails), so long-shot
   brackets are never priced at zero.
4. **Same-day tracking.** Once the climate day starts, live observations
   are pulled every 5 minutes:
   * METAR `T` groups (0.1 °C), 6-hour max/min groups, and the 24-hour
     group are parsed as exact bounds. Whole-°C 5-minute readings get a
     ±0.9 °F band.
   * Final high = max(observed so far, remaining-hours forecast). The
     remaining-hours forecast shifts by 60% of how far the latest reading
     is running from the forecast.
   * Brackets below an already-observed high go to 0.
5. Samples are rounded to whole °F, matching the climate report.

## 4. Strategy

Code: `src/kalshi_bot/weather/strategy.py`

* **Probability used for trading** = `confidence × model + (1 − confidence) × market mid`
  (default 0.7). This hedges against model error and keeps Kelly sizing sane.
* **Entry:** buy YES if `p − ask − fee ≥ min_edge` (default 6c), or buy NO if
  `(1 − p) − no_ask − fee ≥ min_edge`. Fee = `0.07 × P × (1 − P)` per contract.
* **Size:** quarter-Kelly (from `KALSHI_RISK__KELLY_FRACTION`), then capped by:
  * 50 contracts per order;
  * $15 per market, $25 per event, $100 per date across **all** cities,
    since one weather system moves several cities at once;
  * the displayed ask size.
* **Exit:** sells back (reduce-only) if held contracts bid more than the model
  thinks they're worth, after fees and a 4c margin.
* **Partition arbitrage:** if the YES asks across all brackets plus fees cost
  less than 100c, buy the full set. The NO-side mirror is handled too. In live
  mode each leg is a separate immediate-or-cancel order, so a partial fill can
  leave a small unhedged leg.
* All orders are **immediate-or-cancel limits at the touch**, so nothing rests
  on the book. Trading stops 5 minutes before each market closes.
* **Station safety check:** if a market's rules text doesn't mention the
  expected station (e.g. "Midway" for Chicago), that series isn't traded.

## 5. Configuration (`KALSHI_WEATHER__*`)

| Variable | Default | Meaning |
|---|---|---|
| `STATIONS` | `NYC,CHI,AUS,MIA,DEN` | Cities to model/trade |
| `KINDS` | `high,low` | Market types |
| `SERIES_OVERRIDES` | `{}` | JSON, e.g. `{"NYC:low":"KXLOWTNYC"}` if a ticker differs |
| `USER_AGENT` | placeholder | **Set this** (NWS requires contact info) |
| `DATA_DIR` | `./data/weather` | Logs, ledger, calibration |
| `OPENMETEO_API_KEY` | empty | Paid key (see data-source notes) |
| `MIN_EDGE` | `0.06` | Edge after fees required to trade |
| `CONFIDENCE` | `0.7` | Model vs. market-mid weight |
| `MIN_PRICE` / `MAX_PRICE` | `3` / `95` | Don't trade outside this price range (cents) |
| `MAX_CONTRACTS_PER_ORDER` | `50` | |
| `MAX_COST_PER_MARKET_CENTS` | `1500` | |
| `MAX_COST_PER_EVENT_CENTS` | `2500` | |
| `MAX_COST_PER_DAY_CENTS` | `10000` | Correlation guard across cities |
| `PAPER_BANKROLL_CENTS` | `50000` | Paper starting bankroll |
| `EXIT_ON_REVERSAL` | `true` | |
| `PARTITION_ARB` | `true` | |
| `FEE_RATE` | `0.07` | Update if Kalshi's fee schedule changes |
| `POLL_INTERVAL` / `FORECAST_REFRESH` / `OBS_REFRESH` | `60` / `3600` / `300` | Seconds |
| `MAX_LEAD_HOURS` | `52` | Only events whose climate day ends within this |
| `ENSEMBLE_MODELS` / `DETERMINISTIC_MODELS` | see `settings.py` | JSON list of alternate-ID groups |

The existing `KALSHI_RISK__*` limits also apply to every order.

## 6. Data sources and their limits

* **Kalshi:** markets and orders through the existing client. Settled
  results come from `GET /markets/{ticker}`.
* **api.weather.gov:** free, no key, but a descriptive `User-Agent` is required.
* **Open-Meteo:** the free tier is for **non-commercial use** with roughly
  10k calls a day, and ensemble calls count extra. Trading for profit is
  arguably commercial, so consider a paid plan and set
  `KALSHI_WEATHER__OPENMETEO_API_KEY`. The bot then uses the
  `customer-*` hosts automatically. Default usage is about 35 requests per
  hour for 5 cities.
* **IEM (Iowa Environmental Mesonet):** archived NWS climate reports, used
  for calibration.

Model IDs on Open-Meteo change from time to time. Each entry in
`ENSEMBLE_MODELS` is a list of alternates, and `weather-check` shows which
one resolved.

## 7. Adding a city

Add a `Station` to `STATIONS` in `src/kalshi_bot/weather/stations.py`, with:
* the ICAO code of the *settlement* station (check the market rules);
* its coordinates;
* its **standard-time** UTC offset;
* both series tickers;
* a word from the rules text for the safety check.

Then run `weather-check` and `weather-calibrate`.

## 8. Files

```
src/kalshi_bot/weather/
  stations.py      station registry + climate-day windows
  settings.py      WeatherSettings (KALSHI_WEATHER__*)
  brackets.py      market -> integer bracket parsing + partition validation
  forecasts.py     Open-Meteo ensembles / deterministic / historical
  nws.py           api.weather.gov gridpoint extremes + observations
  observations.py  METAR remark parsing -> observed high/low bounds
  climate.py       IEM climate-report archive (settlement truth)
  calibration.py   bias/spread store and fitting helpers
  model.py         Monte Carlo distribution -> bracket probabilities
  fees.py          Kalshi fee math
  strategy.py      entries, exits, partition arb, budgets
  ledger.py        paper ledger with settlement
  datalog.py       JSONL logging
  engine.py        orchestration loop (log / paper / live)
  report.py        calibration jobs + model-vs-market scoring
  cli_commands.py  weather-* CLI commands
data/weather/        (created at runtime)
  log/*.jsonl  calibration.json  paper_ledger.json  results.json
```

*Educational software. No guarantee of profit. Weather markets are thin and
fees are material. Start small.*
