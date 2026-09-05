"""
NSE Multi-Pattern Scanner - Bullish Confirmation Candles
============================================================
A breakout above a pivot is more trustworthy when the breakout bar
itself (or the bar right after it) shows real buying conviction, not
just a thin poke through the level. This module scores the bar(s)
around a detected breakout against classic bullish candlestick
patterns and returns the single strongest match.

Deliberately informational, exactly like Breakout Readiness — this
NEVER gates pattern detection. A pattern with no bullish confirmation
candle is still a valid, tradeable signal; it's just flagged so the
trader knows the breakout bar itself wasn't a strong one and a
same-day/next-day confirmation bar may be worth waiting for.

Checked strongest-to-weakest, first match wins:
  Three White Soldiers > Morning Star > Bullish Engulfing >
  Bullish Marubozu > Piercing Line > Hammer > Dragonfly Doji >
  Gap-Up Bullish > Solid Bullish Candle > No Bullish Confirmation
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import config as cfg


# ─── Per-bar geometry ───────────────────────────────────────────────────────

def _row_ohlc(df: pd.DataFrame, pos: int) -> tuple[float, float, float, float]:
    r = df.iloc[pos]
    return float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])


def _metrics(o: float, h: float, l: float, c: float) -> dict:
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "range": rng, "body": body, "body_pct": body / rng,
        "upper_wick": upper, "upper_pct": upper / rng,
        "lower_wick": lower, "lower_pct": lower / rng,
        "bullish": c > o, "bearish": c < o,
        "close_pos": (c - l) / rng,   # 0 = closed at the low, 1 = closed at the high
    }


# ─── Individual pattern checks ─────────────────────────────────────────────

def is_bullish_marubozu(df: pd.DataFrame, pos: int) -> bool:
    o, h, l, c = _row_ohlc(df, pos)
    m = _metrics(o, h, l, c)
    return (
        m["bullish"]
        and m["upper_pct"] <= cfg.CANDLE_MARUBOZU_WICK_MAX_PCT
        and m["lower_pct"] <= cfg.CANDLE_MARUBOZU_WICK_MAX_PCT
    )


def is_hammer(df: pd.DataFrame, pos: int) -> bool:
    o, h, l, c = _row_ohlc(df, pos)
    m = _metrics(o, h, l, c)
    if m["body"] <= 0:
        return False
    return (
        m["lower_wick"] >= cfg.CANDLE_HAMMER_LOWER_WICK_MIN_RATIO * m["body"]
        and m["upper_wick"] <= cfg.CANDLE_HAMMER_UPPER_WICK_MAX_RATIO * m["body"]
        and m["close_pos"] >= 0.5
    )


def is_dragonfly_doji(df: pd.DataFrame, pos: int) -> bool:
    o, h, l, c = _row_ohlc(df, pos)
    m = _metrics(o, h, l, c)
    return (
        m["body_pct"] <= cfg.CANDLE_DOJI_BODY_MAX_PCT
        and m["lower_pct"] >= 0.6
        and m["upper_pct"] <= 0.15
    )


def is_bullish_engulfing(df: pd.DataFrame, pos: int) -> tuple[bool, str]:
    if pos < 1:
        return False, ""
    o0, h0, l0, c0 = _row_ohlc(df, pos - 1)
    o1, h1, l1, c1 = _row_ohlc(df, pos)
    m0, m1 = _metrics(o0, h0, l0, c0), _metrics(o1, h1, l1, c1)
    if not (m0["bearish"] and m1["bullish"]):
        return False, ""
    if not (o1 <= c0 and c1 >= o0):
        return False, ""
    strength = "Strong" if m1["body"] >= cfg.CANDLE_ENGULF_SIZE_MIN_RATIO * max(m0["body"], 1e-9) else "Moderate"
    return True, strength


def is_piercing_line(df: pd.DataFrame, pos: int) -> bool:
    if pos < 1:
        return False
    o0, h0, l0, c0 = _row_ohlc(df, pos - 1)
    o1, h1, l1, c1 = _row_ohlc(df, pos)
    m0, m1 = _metrics(o0, h0, l0, c0), _metrics(o1, h1, l1, c1)
    if not (m0["bearish"] and m1["bullish"]) or m0["body"] <= 0:
        return False
    midpoint0 = (o0 + c0) / 2.0
    return (o1 < c0) and (c1 > midpoint0) and (c1 < o0)


def is_morning_star(df: pd.DataFrame, pos: int) -> bool:
    if pos < 2:
        return False
    o0, h0, l0, c0 = _row_ohlc(df, pos - 2)
    o1, h1, l1, c1 = _row_ohlc(df, pos - 1)
    o2, h2, l2, c2 = _row_ohlc(df, pos)
    m0, m1, m2 = _metrics(o0, h0, l0, c0), _metrics(o1, h1, l1, c1), _metrics(o2, h2, l2, c2)
    if not m0["bearish"] or not m2["bullish"]:
        return False
    if m1["body_pct"] > 0.40:          # middle candle must be small-bodied (the "star")
        return False
    midpoint0 = (o0 + c0) / 2.0
    return c2 > midpoint0 and max(o1, c1) <= max(o0, c0) + m0["range"] * 0.05


def is_three_white_soldiers(df: pd.DataFrame, pos: int) -> bool:
    if pos < 2:
        return False
    rows = [_row_ohlc(df, pos - 2), _row_ohlc(df, pos - 1), _row_ohlc(df, pos)]
    metrics = [_metrics(*r) for r in rows]
    if not all(m["bullish"] and m["body_pct"] >= 0.45 for m in metrics):
        return False
    closes = [r[3] for r in rows]
    if not (closes[0] < closes[1] < closes[2]):
        return False
    for i in (1, 2):
        prev_o, prev_c = rows[i - 1][0], rows[i - 1][3]
        this_o = rows[i][0]
        if this_o < min(prev_o, prev_c) * 0.99 or this_o > max(prev_o, prev_c) * 1.01:
            return False   # opened with a big gap away from the prior body — not "soldiers"
    return all(m["upper_pct"] <= 0.25 for m in metrics)


def is_gap_up_bullish(df: pd.DataFrame, pos: int) -> bool:
    if pos < 1:
        return False
    o0, h0, l0, c0 = _row_ohlc(df, pos - 1)
    o1, h1, l1, c1 = _row_ohlc(df, pos)
    return l1 > h0 and c1 > o1


def is_solid_bullish_candle(df: pd.DataFrame, pos: int) -> bool:
    o, h, l, c = _row_ohlc(df, pos)
    m = _metrics(o, h, l, c)
    return (
        m["bullish"]
        and m["body_pct"] >= cfg.CANDLE_SOLID_BODY_MIN_PCT
        and m["close_pos"] >= cfg.CANDLE_SOLID_CLOSE_POS_MIN
    )


# ─── Master entry point ────────────────────────────────────────────────────

def detect_confirmation_candle(df: pd.DataFrame, breakout_pos: Optional[int] = None) -> dict:
    """
    Examine the bar at `breakout_pos` (default: the most recent bar)
    for a recognised bullish confirmation pattern, using up to 2 prior
    bars for multi-candle patterns.

    Returns:
      {
        "found":    bool,
        "pattern":  str,           # e.g. "Bullish Engulfing"
        "strength": str,           # "Strong" / "Moderate" / "Weak" / "None"
        "date":     str or None,   # ISO date of the confirmation bar
        "bars_from_end": int,      # 0 = the last available bar
        "detail":   str,           # one-line human explanation
      }
    """
    n = len(df)
    if n == 0:
        return _none_result()
    pos = n - 1 if breakout_pos is None else max(0, min(breakout_pos, n - 1))

    try:
        if is_three_white_soldiers(df, pos):
            return _result(True, "Three White Soldiers", "Strong", df, pos,
                "Three consecutive strong bullish candles, each closing higher than the last "
                "— sustained buying pressure carrying straight through the breakout.")

        if is_morning_star(df, pos):
            return _result(True, "Morning Star", "Strong", df, pos,
                "A bearish candle, a small-bodied pause, then a strong bullish candle closing "
                "back into the first candle's body — a classic three-bar reversal.")

        engulf, engulf_strength = is_bullish_engulfing(df, pos)
        if engulf:
            return _result(True, "Bullish Engulfing", engulf_strength, df, pos,
                "The breakout candle's real body fully engulfs the prior candle's body "
                "— buyers decisively overwhelmed sellers on the move.")

        if is_bullish_marubozu(df, pos):
            return _result(True, "Bullish Marubozu", "Strong", df, pos,
                "A near full-bodied bullish candle with almost no wicks (open near the low, "
                "close near the high) — one-sided conviction all session.")

        if is_piercing_line(df, pos):
            return _result(True, "Piercing Line", "Moderate", df, pos,
                "Opened below the prior candle's close but fought back above its midpoint "
                "by the close — a meaningful swing in buyers' favour.")

        if is_hammer(df, pos):
            return _result(True, "Hammer", "Moderate", df, pos,
                "A long lower wick with a small body near the top of the range — sellers "
                "pushed price down intraday and buyers took it right back.")

        if is_dragonfly_doji(df, pos):
            return _result(True, "Dragonfly Doji", "Moderate", df, pos,
                "Open and close nearly identical with a long lower wick — the session's "
                "indecision resolved firmly in buyers' favour by the close.")

        if is_gap_up_bullish(df, pos):
            return _result(True, "Gap-Up Bullish", "Moderate", df, pos,
                "Price gapped above the prior candle's high and held the gap into a "
                "higher close — a sign of urgency among buyers.")

        if is_solid_bullish_candle(df, pos):
            return _result(True, "Solid Bullish Candle", "Weak", df, pos,
                "A clean bullish candle with a real body closing near its high — healthy, "
                "though it doesn't form one of the named reversal patterns.")

        return _result(False, "No Bullish Confirmation", "None", df, pos,
            "The breakout bar itself is not a strong bullish candle (small body, closed "
            "off its high, or outright red) — consider waiting for a stronger confirmation "
            "bar before sizing up.")
    except Exception:
        return _none_result()


def _result(found: bool, pattern: str, strength: str, df: pd.DataFrame, pos: int, detail: str) -> dict:
    try:
        d = pd.Timestamp(df.index[pos]).strftime("%Y-%m-%d")
    except Exception:
        d = None
    return {
        "found": found,
        "pattern": pattern,
        "strength": strength,
        "date": d,
        "bars_from_end": (len(df) - 1 - pos),
        "detail": detail,
    }


def _none_result() -> dict:
    return {
        "found": False, "pattern": "N/A", "strength": "None",
        "date": None, "bars_from_end": None,
        "detail": "Not enough data to assess the breakout candle.",
    }
