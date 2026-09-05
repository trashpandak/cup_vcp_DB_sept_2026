"""
NSE Cup & Handle Scanner - Technical Indicators
=================================================
Pure functions operating on pandas Series/DataFrames. No I/O, no
side effects — easy to unit-test in isolation.

Includes:
  - RSI, ADX, ATR (Wilder's smoothing, the standard method)
  - Simple moving averages
  - O'Neil-style cross-sectional RS Rating (percentile rank vs the
    whole universe, NOT a fixed-curve comparison against a benchmark)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import RS_WEIGHTS


# ─── RSI ──────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.fillna(100)  # avg_loss == 0 means pure uptrend -> RSI 100
    return out


# ─── ATR ──────────────────────────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr_at(df: pd.DataFrame, idx, period: int = 14) -> Optional[float]:
    """
    Convenience: compute ATR series once and return the value at a
    specific index label (used to capture ATR at handle start for the
    Breakout Readiness 'ATR Contracting' check).
    """
    series = atr(df, period)
    if idx not in series.index:
        return None
    val = series.loc[idx]
    return float(val) if pd.notna(val) else None


# ─── ADX ──────────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr_smooth.replace(0, np.nan)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr_smooth.replace(0, np.nan)
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx_val.fillna(0.0)


# ─── Moving averages ────────────────────────────────────────────────────────

def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


# ─── RS Rating (O'Neil cross-sectional percentile method) ────────────────

def raw_weighted_return(close: pd.Series) -> Optional[float]:
    """
    Compute a single symbol's weighted return used as the RAW input
    to cross-sectional RS Rating ranking. NOT a percentile itself —
    that happens once we have every symbol's raw value (see
    cross_sectional_rs_rating below).

    weighted_return = 0.4*(3m) + 0.2*(6m) + 0.2*(9m) + 0.2*(12m)

    Returns None if there isn't enough history for the longest lookback.
    """
    if close is None or len(close) == 0:
        return None

    last_price = close.iloc[-1]
    if pd.isna(last_price) or last_price <= 0:
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    have_any = False

    for _label, (weight, lookback_bars) in RS_WEIGHTS.items():
        if len(close) <= lookback_bars:
            continue
        past_price = close.iloc[-1 - lookback_bars]
        if pd.isna(past_price) or past_price <= 0:
            continue
        period_return = (last_price / past_price) - 1.0
        weighted_sum += weight * period_return
        total_weight += weight
        have_any = True

    if not have_any or total_weight == 0:
        return None

    # Renormalise in case some lookbacks were unavailable (newer listings)
    return weighted_sum / total_weight


def cross_sectional_rs_rating(weighted_returns: pd.Series) -> pd.Series:
    """
    Convert a Series of {symbol: raw_weighted_return} into O'Neil-style
    RS Ratings (1-99) by percentile-ranking every symbol against every
    OTHER symbol in the same universe. This is what makes it a true
    "Relative Strength" rating rather than a comparison to a single
    benchmark curve.
    """
    valid = weighted_returns.dropna()
    if valid.empty:
        return pd.Series(dtype=float)

    ranks = valid.rank(pct=True, method="average")
    rs_rating = (ranks * 98 + 1).round(0)  # scale to 1-99
    return rs_rating


def rs_rating_trend(
    rs_rating_now: float,
    rs_rating_n_weeks_ago: Optional[float],
) -> str:
    """
    Classify RS Rating trend over the lookback window used for the
    Breakout Readiness 'Rising RS' check. Informational tag only.
    """
    if rs_rating_n_weeks_ago is None or pd.isna(rs_rating_n_weeks_ago):
        return "Unknown"
    delta = rs_rating_now - rs_rating_n_weeks_ago
    if delta >= 5:
        return "Improving"
    if delta <= -5:
        return "Declining"
    return "Flat"


# ─── Volume cleaning ────────────────────────────────────────────────────────

def clean_volume(volume: pd.Series, ffill_limit: int = 5) -> pd.Series:
    """
    NSE data via yfinance occasionally has zero-volume bars (holidays,
    circuit freezes). Per spec, replace zeros with NaN, forward-fill a
    short gap, then fall back to the symbol's own median volume for
    anything still missing. Never drop rows — that would shift bar
    alignment for pattern detection.
    """
    v = volume.copy()
    v = v.replace(0, np.nan)
    v = v.ffill(limit=ffill_limit)
    if v.isna().any():
        median_vol = v.median()
        if pd.isna(median_vol):
            median_vol = 0.0
        v = v.fillna(median_vol)
    return v


# ─── Composite indicator bundle (convenience for detector/entry_exit) ────

def compute_indicator_bundle(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Compute the full set of indicators needed downstream in one pass,
    returning a dict of Series keyed by name. Avoids re-computing the
    same ATR/RSI/ADX multiple times across detector + readiness +
    entry_exit for the same symbol/timeframe.
    """
    from config import MA_PERIODS_DAILY, MA_PERIODS_WEEKLY, MA_PERIODS_MONTHLY

    close = df["Close"]
    vol_clean = clean_volume(df["Volume"]) if "Volume" in df.columns else None

    ma_periods = {
        "daily": MA_PERIODS_DAILY,
        "weekly": MA_PERIODS_WEEKLY,
        "monthly": MA_PERIODS_MONTHLY,
    }.get(timeframe, MA_PERIODS_DAILY)

    bundle = {
        "rsi": rsi(close, period=14),
        "atr": atr(df, period=14),
        "adx": adx(df, period=14),
        "volume_clean": vol_clean,
        "ma": {p: sma(close, p) for p in ma_periods},
    }
    return bundle
