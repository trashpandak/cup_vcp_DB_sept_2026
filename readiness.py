"""
NSE Cup & Handle Scanner - Breakout Readiness Score
======================================================
Implements spec v3 Part 3I.

Quality Score (in cup_handle_detector.py) answers "is this historically
a good-looking pattern?" Breakout Readiness answers "is this stock
actionable RIGHT NOW?" — a separate, narrower question computed only
for signals close enough to their pivot to matter (NEAR BREAKOUT,
BASING, BREAKOUT NOW). EARLY STAGE and CUP ONLY signals get None.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import config as cfg
from cup_handle_detector import CupHandleSignal
from indicators import atr as atr_series_fn, rsi as rsi_fn


READY_SIGNAL_TYPES = {"NEAR BREAKOUT", "BASING", "BREAKOUT NOW"}


def compute_breakout_readiness(
    sig: CupHandleSignal,
    df: pd.DataFrame,
    rs_trend: str,
) -> dict:
    """
    Returns a dict with:
      readiness_pct      : float 0-100, or None if not applicable
      reasons_str         : "✔ Near Pivot | ✘ Tight Handle | ..."
      near_pivot          : bool
      tight_handle        : bool
      rising_rs           : bool
      atr_contracting     : bool
      above_50ma          : bool
    """
    if sig.signal_type not in READY_SIGNAL_TYPES:
        return {
            "readiness_pct": None,
            "reasons_str": "N/A — pattern not yet near pivot",
            "near_pivot": False,
            "tight_handle": False,
            "rising_rs": False,
            "atr_contracting": False,
            "above_50ma": False,
        }

    near_pivot = _check_near_pivot(sig)
    tight_handle = _check_tight_handle(sig)
    rising_rs = rs_trend == "Improving"
    atr_contracting = _check_atr_contracting(sig, df)
    above_50ma = _check_above_50ma(sig, df)

    weights = cfg.READINESS_WEIGHTS
    readiness_pct = 0
    checks = [
        ("near_pivot", near_pivot, "Near Pivot"),
        ("tight_handle", tight_handle, "Tight Handle"),
        ("rising_rs", rising_rs, "Rising RS"),
        ("atr_contracting", atr_contracting, "ATR Contracting"),
        ("above_50ma", above_50ma, "Above 50 MA"),
    ]

    reasons_parts = []
    for key, passed, label in checks:
        if passed:
            readiness_pct += weights[key]
        mark = "✔" if passed else "✘"
        reasons_parts.append(f"{mark} {label}")

    return {
        "readiness_pct": readiness_pct,
        "reasons_str": " | ".join(reasons_parts),
        "near_pivot": near_pivot,
        "tight_handle": tight_handle,
        "rising_rs": rising_rs,
        "atr_contracting": atr_contracting,
        "above_50ma": above_50ma,
    }


def readiness_band(readiness_pct: Optional[float]) -> str:
    """For report colour-coding."""
    if readiness_pct is None:
        return "n/a"
    if readiness_pct >= cfg.READINESS_BAND_HIGH:
        return "high"
    if readiness_pct >= cfg.READINESS_BAND_MEDIUM:
        return "medium"
    return "low"


# ─── Individual checks ──────────────────────────────────────────────────────

def _check_near_pivot(sig: CupHandleSignal) -> bool:
    return abs(sig.price_vs_pivot_pct) <= cfg.READINESS_NEAR_PIVOT_PCT


def _check_tight_handle(sig: CupHandleSignal) -> bool:
    if not sig.has_handle:
        return False
    # "tightness sub-score >= 3/5 pts" per spec — our tightness scoring
    # awards 5 (full), 3 (partial), or 0 (none) — so >=3 means full or partial
    return sig.handle_tightness_pts >= 3


def _check_atr_contracting(sig: CupHandleSignal, df: pd.DataFrame) -> bool:
    if sig.atr_at_handle_start is None or sig.atr_at_handle_start <= 0:
        return False
    try:
        current_atr = atr_series_fn(df, period=cfg.ATR_PERIOD).iloc[-1]
    except Exception:
        return False
    if pd.isna(current_atr):
        return False
    return float(current_atr) < sig.atr_at_handle_start


def _check_above_50ma(sig: CupHandleSignal, df: pd.DataFrame) -> bool:
    """
    Daily: 50-day MA. Weekly: 10-week MA. Monthly: 3-month MA.
    These approximate the same ~50-trading-day window on each timeframe.
    """
    ma_period = {"daily": 50, "weekly": 10, "monthly": 3}.get(sig.timeframe, 50)
    close = df["Close"]
    if len(close) < ma_period:
        return False
    ma_val = close.rolling(window=ma_period, min_periods=ma_period).mean().iloc[-1]
    if pd.isna(ma_val):
        return False
    return sig.current_price > float(ma_val)
