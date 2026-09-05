"""
NSE Cup & Handle Scanner - Data Downloader
=============================================
Downloads/updates Parquet OHLCV files for every symbol in the universe.

Key behaviour (this is what makes "download once, update daily" work):
  - On first run (no Parquet file / full_refresh): downloads full
    history from 2000-01-01.
  - On every subsequent run: reads the last cached date per symbol and
    only requests bars after that date, then merges into the existing
    Parquet file. This is what GitHub Actions caching of data/daily/
    relies on — see daily_scan.yml.

Resilience features carried over from the proven Darvas scanner:
  - Batch download with per-symbol incremental start dates
  - Separate fast retry queue for timeouts vs slow exponential-backoff
    queue for rate limits
  - Robust extraction that handles both flat and MultiIndex yfinance
    responses
  - Per-symbol fallback via Ticker.history() for benchmarks and final
    retries
"""

from __future__ import annotations

import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    BATCH_DELAY_SECONDS, BATCH_SIZE, DAILY_DIR,
    EXPONENTIAL_BASE, MAX_RETRIES,
    RATELIMIT_RETRY_WAIT_MIN, TIMEOUT_RETRY_WAIT_SEC,
    NIFTY50_SYMBOL, NIFTY500_SYMBOL,
)
from logger_utils import get_logger

log = get_logger("scanner")

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ─── Public API ───────────────────────────────────────────────────────────

def run_download(symbols: list[str], full_refresh: bool = False) -> None:
    """Main entry point. Downloads/updates Parquet files for all symbols."""
    benchmarks = [NIFTY50_SYMBOL, NIFTY500_SYMBOL]

    equity = list(symbols)
    random.shuffle(equity)   # break alphabetical clustering across batches
    all_targets = benchmarks + equity

    if full_refresh:
        log.info("FULL REFRESH — removing existing Parquet cache")
        for sym in all_targets:
            p = _parquet_path(sym)
            if p.exists():
                p.unlink()

    for bm in benchmarks:
        _download_single_with_retry(bm)

    batches = _make_batches(equity, BATCH_SIZE)
    log.info("Downloading %d equity symbols in %d batches (size=%d)",
              len(equity), len(batches), BATCH_SIZE)

    timeout_queue: list[str] = []
    ratelimit_queue: list[str] = []

    for i, batch in enumerate(batches, 1):
        log.info("Batch %d/%d – %d symbols", i, len(batches), len(batch))
        saved, timed_out, rate_limited = _download_batch(batch)
        log.info("  Batch %d result: saved=%d  timeouts=%d  rate_limited=%d",
                  i, saved, len(timed_out), len(rate_limited))
        timeout_queue.extend(timed_out)
        ratelimit_queue.extend(rate_limited)

        if i < len(batches):
            time.sleep(BATCH_DELAY_SECONDS)

    if timeout_queue:
        log.info("Retrying %d timeout symbols after %ds ...",
                  len(timeout_queue), TIMEOUT_RETRY_WAIT_SEC)
        time.sleep(TIMEOUT_RETRY_WAIT_SEC)
        still_failed = _retry_individually(timeout_queue, max_attempts=3, wait_sec=30)
        if still_failed:
            log.warning("Permanently timed out: %s", still_failed[:10])

    if ratelimit_queue:
        _retry_ratelimited(ratelimit_queue)

    log.info("Download pipeline complete.")
    _log_data_completeness_summary(equity)


def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV DataFrame from Parquet cache."""
    p = _parquet_path(symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df.sort_index(inplace=True)
        available = [c for c in OHLCV_COLS if c in df.columns]
        return df[available] if available else None
    except Exception as e:
        log.error("Load failed %s: %s", symbol, e)
        return None


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly OHLCV, grouped by ISO calendar week (Monday-Sunday) rather
    than pandas' 'W-FRI' resample anchor.

    Why not plain `.resample("W-FRI", label="left", closed="left")`:
    that combination labels each week's bar with the *previous*
    Friday (a full week earlier than the week it actually
    summarizes) — e.g. the Mon Jan 1 - Fri Jan 5 week gets labeled
    "2023-12-29". This silently shifts every weekly candle back by
    about a week versus what any real charting platform (TradingView,
    NSE's own charts) shows for the same stock, which is exactly the
    mismatch reported when comparing this scanner's weekly chart
    against TradingView.

    This groups by (ISO year, ISO week) directly and labels each
    resulting bar with the first trading day actually present in that
    week — so a normal week is labeled Monday, and a week where
    Monday was an NSE holiday is correctly labeled with Tuesday (or
    whichever day trading actually resumed), rather than assuming
    Monday always exists.
    """
    if daily is None or daily.empty:
        return daily

    iso = daily.index.isocalendar()
    work = daily.copy()
    work["_iso_year"] = iso["year"].values
    work["_iso_week"] = iso["week"].values

    grouped = work.groupby(["_iso_year", "_iso_week"], sort=True)
    weekly = grouped.agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"),
    )
    first_trading_day = grouped.apply(lambda g: g.index[0], include_groups=False)
    weekly.index = pd.DatetimeIndex(first_trading_day.values)
    weekly.index.name = "Date"

    return weekly.dropna(subset=["Close"]).sort_index()


def resample_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly OHLCV via resampling."""
    return (
        daily.resample("MS")
        .agg({"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )


# ─── Data completeness summary ─────────────────────────────────────────────

def _log_data_completeness_summary(symbols: list[str]) -> None:
    buckets = {
        "5y+ (1250+ bars)":     0,
        "2-5y (500-1249)":      0,
        "1-2y (250-499)":       0,
        "150-249 (min usable)": 0,
        "<150 (too short)":     0,
        "missing":              0,
    }
    total_rows = 0
    oldest_date = None
    newest_date = None

    for sym in symbols:
        p = _parquet_path(sym)
        if not p.exists():
            buckets["missing"] += 1
            continue
        try:
            df = pd.read_parquet(p, columns=["Close"])
            n = len(df)
            total_rows += n
            if n >= 1250:
                buckets["5y+ (1250+ bars)"] += 1
            elif n >= 500:
                buckets["2-5y (500-1249)"] += 1
            elif n >= 250:
                buckets["1-2y (250-499)"] += 1
            elif n >= 150:
                buckets["150-249 (min usable)"] += 1
            else:
                buckets["<150 (too short)"] += 1

            df.index = pd.to_datetime(df.index)
            if len(df) > 0:
                first, last = df.index[0], df.index[-1]
                if oldest_date is None or first < oldest_date:
                    oldest_date = first
                if newest_date is None or last > newest_date:
                    newest_date = last
        except Exception:
            buckets["missing"] += 1

    log.info("=" * 60)
    log.info("DATA COMPLETENESS SUMMARY (%d symbols)", len(symbols))
    for label, count in buckets.items():
        pct = count / len(symbols) * 100 if symbols else 0
        log.info("  %-24s: %5d  (%.1f%%)", label, count, pct)
    if oldest_date is not None:
        log.info("  Oldest data point      : %s", oldest_date.date())
        log.info("  Newest data point      : %s", newest_date.date())
    avg_rows = total_rows / len(symbols) if symbols else 0
    log.info("  Average rows/symbol     : %.0f", avg_rows)
    log.info("=" * 60)


# ─── Batch download internals ───────────────────────────────────────────────

def _download_batch(batch: list[str]) -> tuple[int, list[str], list[str]]:
    """Download a batch. Returns (n_saved, timed_out_syms, rate_limited_syms)."""
    starts = {sym: _incremental_start(sym) for sym in batch}
    end = (date.today() + timedelta(days=1)).isoformat()
    min_start = min(starts.values())

    saved = 0
    timed_out: list[str] = []
    rate_limited: list[str] = []

    try:
        raw = yf.download(
            batch, start=min_start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception as exc:
        err = str(exc).lower()
        if "timeout" in err or "timed out" in err or "curl: (28)" in err:
            log.warning("Batch timeout — queuing for fast retry: %s...", batch[:3])
            return 0, batch, []
        if "429" in err or "rate" in err or "too many" in err:
            log.warning("Rate limit hit — queuing for slow retry")
            return 0, [], batch
        log.error("Batch exception: %s", exc)
        return 0, batch, []

    if raw is None or raw.empty:
        log.warning("Empty response for batch — queuing all for retry")
        return 0, batch, []

    for sym in batch:
        sym_start = starts[sym]
        df = _extract_symbol_robust(raw, sym, len(batch))
        if df is None or df.empty:
            timed_out.append(sym)
            continue

        df = df[df.index >= pd.Timestamp(sym_start)]
        if df.empty:
            saved += 1   # already up to date — not a failure
            continue

        _merge_and_save(sym, df)
        saved += 1

    return saved, timed_out, rate_limited


def _extract_symbol_robust(raw: pd.DataFrame, symbol: str, n_syms: int) -> Optional[pd.DataFrame]:
    """Robustly extract single-symbol OHLCV from a yfinance response."""
    try:
        if n_syms == 1 or raw.columns.nlevels == 1:
            df = raw.copy()
        else:
            if symbol not in raw.columns.get_level_values(1):
                return None
            df = raw.xs(symbol, axis=1, level=1).copy()

        df.columns = [str(c).strip() for c in df.columns]
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        elif "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])

        present = [c for c in OHLCV_COLS if c in df.columns]
        if len(present) < 4:
            return None
        df = df[present].dropna(how="all")
        df.index = pd.to_datetime(df.index)
        df = df[df["Close"] > 0]
        return df if not df.empty else None

    except Exception as e:
        log.debug("Extraction error %s: %s", symbol, e)
        return None


# ─── Single-symbol download (benchmarks + fallback) ───────────────────────

def _download_single_with_retry(symbol: str) -> bool:
    sym_start = _incremental_start(symbol)
    for attempt in range(1, 4):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=sym_start,
                end=(date.today() + timedelta(days=1)).isoformat(),
                auto_adjust=True,
            )
            if df.empty:
                log.debug("%s: empty history (attempt %d)", symbol, attempt)
                time.sleep(5 * attempt)
                continue

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [str(c) for c in df.columns]
            present = [c for c in OHLCV_COLS if c in df.columns]
            df = df[present].dropna(how="all")
            df = df[df["Close"] > 0]

            _merge_and_save(symbol, df)
            log.info("Downloaded %s: %d rows", symbol, len(df))
            return True

        except Exception as e:
            log.warning("%s attempt %d failed: %s", symbol, attempt, e)
            time.sleep(10 * attempt)

    log.error("Failed to download %s after 3 attempts", symbol)
    return False


# ─── Retry queues ─────────────────────────────────────────────────────────

def _retry_individually(symbols: list[str], max_attempts: int, wait_sec: int) -> list[str]:
    still_failed = []
    for sym in symbols:
        success = _download_single_with_retry(sym)
        if not success:
            still_failed.append(sym)
        time.sleep(wait_sec)
    return still_failed


def _retry_ratelimited(symbols: list[str]) -> None:
    remaining = list(symbols)
    for attempt in range(1, MAX_RETRIES + 1):
        if not remaining:
            break
        wait = RATELIMIT_RETRY_WAIT_MIN * 60 * (EXPONENTIAL_BASE ** (attempt - 1))
        log.info("Rate-limit retry %d/%d — waiting %.0fs for %d symbols",
                  attempt, MAX_RETRIES, wait, len(remaining))
        time.sleep(wait)

        mini_batches = _make_batches(remaining, 10)
        still_bad = []
        for mb in mini_batches:
            saved, to, rl = _download_batch(mb)
            still_bad.extend(to + rl)
            time.sleep(5)

        log.info("Rate-limit retry %d: %d recovered, %d still failing",
                  attempt, len(remaining) - len(still_bad), len(still_bad))
        remaining = still_bad

    if remaining:
        log.error("Permanently failed %d symbols: %s", len(remaining), remaining[:10])


# ─── Helpers ──────────────────────────────────────────────────────────────

def _parquet_path(symbol: str) -> Path:
    safe = symbol.replace("^", "_").replace(".", "_")
    return DAILY_DIR / f"{safe}.parquet"


def _incremental_start(symbol: str) -> str:
    p = _parquet_path(symbol)
    if not p.exists():
        return "2000-01-01"
    try:
        df = pd.read_parquet(p, columns=["Close"])
        df.index = pd.to_datetime(df.index)
        last = df.index[-1].date()
        return (last + timedelta(days=1)).isoformat()
    except Exception:
        return "2000-01-01"


def _merge_and_save(symbol: str, new_df: pd.DataFrame) -> None:
    p = _parquet_path(symbol)
    if p.exists():
        try:
            existing = pd.read_parquet(p)
            existing.index = pd.to_datetime(existing.index)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        except Exception:
            combined = new_df.sort_index()
    else:
        combined = new_df.sort_index()

    cols = [c for c in OHLCV_COLS if c in combined.columns]
    combined[cols].to_parquet(p, index=True, compression="snappy")
    log.debug("Saved %s — %d rows (latest: %s)",
               symbol, len(combined), combined.index[-1].date())


def _make_batches(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]
