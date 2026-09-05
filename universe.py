"""
NSE Cup & Handle Scanner - Symbol Universe
============================================
Fetches the complete list of NSE-listed equity symbols, formatted for
yfinance (e.g. 'RELIANCE.NS'). Caches locally; refreshes every 24h or
on demand via force_refresh.

Falls back to a stale cache if the live NSE fetch fails (NSE's site
blocks naive requests fairly often, so resilience here matters more
than freshness — a 2-day-old symbol list is still 99% correct).
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR
from logger_utils import get_logger

log = get_logger("scanner")

CACHE_FILE = DATA_DIR / "nse_symbols.json"
META_FILE = DATA_DIR / "nse_symbol_meta.json"   # symbol -> {name, sector}

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

CACHE_MAX_AGE_HOURS = 24


def fetch_nse_symbols(force_refresh: bool = False) -> list[str]:
    """Return a list of NSE equity symbols, e.g. ['RELIANCE.NS', ...]."""
    if not force_refresh and CACHE_FILE.exists():
        age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_hours < CACHE_MAX_AGE_HOURS:
            symbols = _load_cache()
            log.info("Using cached NSE symbol list (%d symbols, %.1fh old)",
                      len(symbols), age_hours)
            return symbols

    log.info("Fetching NSE equity master list …")
    symbols, meta = _fetch_live()
    if symbols:
        _save_cache(symbols)
        if meta:
            _save_meta(meta)
        log.info("Fetched %d NSE symbols from live source", len(symbols))
        return symbols

    if CACHE_FILE.exists():
        log.warning("Live fetch failed — using stale cached symbol list")
        return _load_cache()

    raise RuntimeError("Could not obtain NSE symbol list from any source.")


def get_symbol_meta(symbol: str) -> dict:
    """Return {'name': ..., 'sector': ...} for a symbol, best-effort."""
    if not META_FILE.exists():
        return {"name": "", "sector": ""}
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
        return meta.get(symbol, {"name": "", "sector": ""})
    except Exception:
        return {"name": "", "sector": ""}


# ─── Private helpers ──────────────────────────────────────────────────────

def _fetch_live() -> tuple[list[str], dict]:
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        resp = session.get(NSE_EQUITY_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        sym_col = next(c for c in df.columns if "SYMBOL" in c.upper())
        name_col = next(
            (c for c in df.columns if "NAME" in c.upper()), None
        )
        series_col = next(
            (c for c in df.columns if c.strip().upper() == "SERIES"), None
        )

        # Keep only the main EQ series if the column exists — avoids
        # picking up BE/BZ/illiquid series duplicates of the same company
        if series_col is not None:
            df = df[df[series_col].astype(str).str.strip().str.upper() == "EQ"]

        df = df.dropna(subset=[sym_col])
        symbols = sorted({f"{str(s).strip()}.NS" for s in df[sym_col]})

        meta: dict[str, dict] = {}
        if name_col is not None:
            for _, row in df.iterrows():
                sym = f"{str(row[sym_col]).strip()}.NS"
                meta[sym] = {
                    "name": str(row.get(name_col, "")).strip(),
                    "sector": "",   # NSE equity master doesn't include sector;
                                    # left blank, report shows "" gracefully
                }

        return symbols, meta

    except Exception as exc:
        log.error("Live NSE fetch failed: %s", exc)
        return [], {}


def _load_cache() -> list[str]:
    with open(CACHE_FILE) as f:
        return json.load(f)


def _save_cache(symbols: list[str]) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(symbols, f)


def _save_meta(meta: dict) -> None:
    try:
        with open(META_FILE, "w") as f:
            json.dump(meta, f)
    except Exception as e:
        log.debug("Could not save symbol meta: %s", e)
