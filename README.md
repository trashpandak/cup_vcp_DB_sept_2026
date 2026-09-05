# NSE Multi-Pattern Scanner

Multi-timeframe technical pattern scanner for NSE equities, built on
William O'Neil's CANSLIM entry/exit framework. Detects **Cup & Handle,
Double Bottom, VCP (Volatility Contraction Pattern), and Ascending
Triangle**, each with bullish-confirmation-candle analysis at the
breakout and multi-timeframe confluence. Detection is deliberately
loose (maximum sensitivity); entry and exit rules are strict (capital
protection). See `spec_v3_full.txt` for the original design rationale.

## What's new (v4 — multi-pattern extension)

- **3 new detectors**: `double_bottom_detector.py`, `vcp_detector.py`,
  `ascending_triangle_detector.py`. Each emits a `pattern_common.
  PatternSignal` that reuses Cup & Handle's field names with
  generalized meanings (see that file's module docstring), so
  `readiness.py` and `entry_exit.py` work against all 4 pattern types
  unmodified.
- **`candlestick_patterns.py`**: 9 bullish confirmation patterns
  (Bullish Engulfing, Morning Star, Three White Soldiers, Marubozu,
  Piercing Line, Hammer, Dragonfly Doji, Gap-Up, Solid Candle),
  evaluated on the breakout bar to flag genuine buying conviction vs. a
  thin poke through the pivot.
- **Multi-timeframe confluence** is now generalized per pattern family
  (same pattern confirmed on 2+ timeframes), plus a new **cross-pattern
  confluence** badge (different pattern types lining up across
  timeframes for the same symbol).
- **`chart_viewer/template.html`** has a new pattern-type tab row (All
  / Cup & Handle / Double Bottom / VCP / Ascending Triangle) and the
  chart overlay drawing is fully generic — it renders whichever
  pattern's structure lines a card supplies, not just cup/handle boxes.
- `database.py` migrates existing `signals.db` files automatically
  (new columns via `ALTER TABLE`, run once at `init_db()`).
- `generate_demo_data.py` / `run_demo.py`: synthetic demo universe and
  a driver that runs the real pipeline against it with zero network
  access — useful for testing changes without waiting on NSE downloads.

## How it works

1. **Detection** (`cup_handle_detector.py` + the 3 new detector files)
   — scans daily, weekly, and monthly price series for each pattern
   shape with a single hard gate per pattern (see each file's
   docstring). Everything else — prior trend, depth, shape, final-base
   quality — is scored and tagged, never used to reject a pattern.
2. **Breakout Readiness** (`readiness.py`) — a separate 0-100% score
   for signals near their pivot, answering "is this actionable now?"
3. **Confirmation Candle** (`candlestick_patterns.py`) — is the
   breakout bar itself a real bullish candle, or a thin push through
   the level worth waiting on?
4. **Entry/Exit** (`entry_exit.py`) — strict O'Neil rules: volume
   confirmation, 8% max stop loss, 20%/measured-move/Fibonacci
   targets, position sizing, and a sell-rule checklist.
5. **Report** (`report.py`) — a multi-sheet Excel workbook spanning
   all 4 pattern types: Today's Signals, Near Breakout Watchlist,
   Active Tracking, Historical Signals, and Strategy Summary.
6. **HTML Dashboard** (`chart_export.py` + `chart_viewer/template.html`)
   — a single self-contained HTML file with pattern-type tabs, live
   charts, and full trade-plan detail per signal.

## Local setup

```bash
pip install -r requirements.txt
python main.py                       # incremental scan (downloads only new bars)
python main.py --full-refresh        # wipe and redownload full history
python main.py --refresh-universe    # force-refresh the NSE symbol list

# No internet? Try it against synthetic data:
python generate_demo_data.py         # writes data/daily/*.parquet + demo meta
python run_demo.py                   # runs the real pipeline against it
```

python main.py --debug-symbol TCS.NS # verbose single-symbol diagnosis
```

Reports are written to `reports/cup_handle_report_YYYY-MM-DD.xlsx`.

## GitHub Actions

`.github/workflows/daily_scan.yml` runs the scanner automatically at
4:00 PM IST on trading days. It:

- Restores the previous run's Parquet data from cache (so only new
  bars get downloaded — first run downloads everything, every run
  after that is incremental)
- Runs the full scan
- Saves the updated data cache for next time
- Uploads the Excel report as a workflow artifact (90-day retention)

Trigger manually from the Actions tab with `workflow_dispatch` to set
`full_refresh`, `refresh_universe`, or `debug_symbol`.

## Configuration

All thresholds live in `config.py` — portfolio size, risk per trade,
detection lookback windows, entry/exit rules, and Breakout Readiness
weights. Detection thresholds are commented with the rationale for
keeping them loose; entry/exit thresholds are commented with the
rationale for keeping them strict. Read the comments before changing
either.

## File overview

| File | Purpose |
|---|---|
| `config.py` | All tuneable constants |
| `logger_utils.py` | Shared logging setup |
| `universe.py` | NSE symbol list fetcher/cache |
| `downloader.py` | Parquet data download, incremental updates |
| `indicators.py` | RSI, ADX, ATR, MAs, RS Rating |
| `pattern_common.py` | Shared multi-pattern infrastructure (`PatternSignal`, zigzag swing detector, generalized scoring) |
| `cup_handle_detector.py` | Cup & Handle detection engine |
| `double_bottom_detector.py` | Double Bottom ("W") detection engine |
| `vcp_detector.py` | VCP (Volatility Contraction Pattern) detection engine |
| `ascending_triangle_detector.py` | Ascending Triangle detection engine |
| `candlestick_patterns.py` | Bullish confirmation-candle detection at breakouts |
| `readiness.py` | Breakout Readiness scoring |
| `entry_exit.py` | Strict entry/exit/position-sizing calculator |
| `explain.py` | Pattern-aware natural-language explanations for the dashboard |
| `database.py` | SQLite persistence for signals and tracking |
| `report.py` | Excel report generator |
| `chart_export.py` | Builds the HTML dashboard's data payload |
| `chart_viewer/template.html` | The dashboard itself (single self-contained HTML file) |
| `main.py` | Orchestrator / CLI entry point |
| `generate_demo_data.py` / `run_demo.py` | Synthetic demo universe + pipeline driver (no network needed) |
