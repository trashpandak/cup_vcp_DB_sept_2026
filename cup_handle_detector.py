"""
NSE Cup & Handle Scanner - Pattern Detection Engine
=====================================================
Implements spec v3 Part 3: maximum-sensitivity Cup & Handle detection.

Philosophy: detection has exactly ONE hard gate (50% recovery from cup
bottom to right rim). Everything else — prior uptrend, cup depth,
shape, volume behaviour, RS Rating — is computed and attached as
INFORMATIONAL tags and quality-score contributions. Nothing else
disqualifies a pattern. See config.py for every threshold used here.

Returns a list of CupHandleSignal dataclass instances per (symbol,
timeframe) call — normally 0 or 1, since we only want the MOST RECENT
valid pattern, but the function is structured to make extending to
multiple non-overlapping patterns straightforward later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from indicators import clean_volume


# ─── Result container ──────────────────────────────────────────────────────

@dataclass
class CupHandleSignal:
    symbol: str
    timeframe: str                  # 'daily' / 'weekly' / 'monthly'

    # Cup geometry
    cup_start_date: date
    cup_bottom_date: date
    cup_end_date: date              # = right rim date
    left_rim_price: float
    cup_bottom_price: float
    right_rim_price: float
    cup_depth_pct: float
    cup_duration_bars: int
    cup_shape: str                  # 'U-Shape' / 'V-Shape' / 'Irregular'
    recovery_pct: float

    # Cup context tags (informational, never gate detection)
    prior_uptrend_pct: float
    prior_uptrend_tag: str          # Strong / Moderate / Possible Bottoming
    cup_depth_class: str            # Shallow / Classic / Deep / Very Deep / Crash
    cup_depth_verify_flag: bool

    # Handle geometry (None if no handle found)
    has_handle: bool
    handle_start_date: Optional[date] = None
    handle_end_date: Optional[date] = None
    handle_low_price: Optional[float] = None
    handle_depth_pct: Optional[float] = None
    handle_duration_bars: Optional[int] = None

    # Handle quality sub-score (0-20, components broken out for the report)
    handle_quality_subscore: float = 0.0
    handle_vol_dryup_pts: float = 0.0
    handle_tightness_pts: float = 0.0
    handle_higher_lows_pts: float = 0.0
    handle_close_high_pts: float = 0.0

    # ATR captured at handle start (or at right rim if no handle) — used
    # downstream by readiness.py for the "ATR Contracting" check
    atr_at_handle_start: Optional[float] = None

    # Pivot / current state
    pivot_point: float = 0.0
    current_price: float = 0.0
    price_vs_pivot_pct: float = 0.0
    signal_type: str = ""           # BREAKOUT NOW / NEAR BREAKOUT / BASING /
                                     # EARLY STAGE / CUP ONLY
    pattern_type: str = ""          # Cup & Handle / Cup Only / Pre-Handle
    already_breaking_out_handle: bool = False

    # Quality score (0-100)
    quality_score: float = 0.0
    rim_symmetry_pct: float = 0.0   # right_rim / left_rim ratio, for display

    # Multi-timeframe confluence — populated later by main.py after all
    # timeframes for a symbol have been scanned
    mtf_confluence: bool = False
    mtf_timeframes: str = ""

    # Raw bar index positions (internal use for ATR-at-handle-start lookup
    # and for downstream entry/exit ATR calc) — not persisted to DB
    _df_ref: object = field(default=None, repr=False, compare=False)


# ─── Public entry point ────────────────────────────────────────────────────

def detect_cup_handle(
    df: pd.DataFrame,
    timeframe: str,
    symbol: str,
    rs_rating: float = 50.0,
) -> list[CupHandleSignal]:
    """
    Scan df (OHLCV, datetime index, ascending) for the most recent valid
    Cup & Handle (or Cup Only) pattern on the given timeframe.

    Returns a list with 0 or 1 CupHandleSignal. (List return type kept
    for forward-compatibility with multi-pattern detection.)
    """
    min_bars_cfg = cfg.MIN_BARS_BY_TIMEFRAME.get(timeframe)
    if min_bars_cfg is None or len(df) < min_bars_cfg:
        return []

    min_w, max_w = cfg.CUP_LOOKBACKS[timeframe]
    if len(df) < min_w:
        return []

    # We want the MOST RECENT pattern. Search windows ending at the last
    # bar, trying progressively larger windows up to max_w, and keep the
    # best (most recent right-rim, then highest recovery) valid candidate.
    n = len(df)
    candidates: list[dict] = []

    # To bound runtime, we don't slide an arbitrary start for every
    # window size — instead we anchor the window END at the last bar
    # and vary the START, since "most recent" patterns are what we want.
    # This is O(max_w) per symbol/timeframe, not O(n * max_w).
    for window_size in range(min_w, min(max_w, n) + 1):
        start_idx = n - window_size
        window = df.iloc[start_idx:]
        cand = _evaluate_window(window, timeframe)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return []

    # Prefer the candidate with the largest window (most established,
    # most complete pattern), tie-broken by best recovery_pct. Once the
    # right-rim search is bounded by handle-reserve (see
    # _evaluate_window), most window sizes that span the same
    # underlying cup will find the same right rim — so window size is
    # the more meaningful differentiator than recovery_pct alone.
    best = max(candidates, key=lambda c: (c["window_size"], round(c["recovery_pct"], 4)))

    sig = _build_signal(df, best, timeframe, symbol, rs_rating)
    return [sig] if sig is not None else []


# ─── Window evaluation: find left rim / bottom / right rim ────────────────

def _evaluate_window(window: pd.DataFrame, timeframe: str) -> Optional[dict]:
    """
    Given a candidate window (ending at the most recent bar), find the
    left rim / cup bottom / right rim. Returns None if the ONLY hard
    gate (50% recovery) fails or basic sanity checks fail.
    """
    close = window["Close"]
    high = window["High"]
    n = len(window)
    if n < 3:
        return None

    # STEP 1 — left rim: highest close in first 25% of window, using a
    # rolling-max to find local peaks, accepting any bar >= 90% of the
    # window's absolute high.
    left_zone_end = max(1, int(n * cfg.LEFT_RIM_WINDOW_FRACTION))
    left_zone = close.iloc[:left_zone_end]
    if left_zone.empty:
        return None

    window_high = close.iloc[:left_zone_end].max()
    rolling_max = left_zone.rolling(
        window=min(cfg.LEFT_RIM_ROLLING_BARS, len(left_zone)), min_periods=1
    ).max()
    eligible = left_zone[left_zone >= window_high * cfg.LEFT_RIM_MIN_PCT_OF_HIGH]
    if eligible.empty:
        return None
    left_rim_pos = left_zone.index.get_loc(eligible.index[0])
    left_rim_price = float(left_zone.iloc[left_rim_pos])
    left_rim_date = left_zone.index[left_rim_pos]

    # STEP 2 — cup bottom: lowest close AFTER left rim
    after_left = close.iloc[left_rim_pos + 1:]
    if after_left.empty:
        return None
    bottom_offset = int(np.argmin(after_left.values))
    bottom_pos = left_rim_pos + 1 + bottom_offset
    cup_bottom_price = float(after_left.values[bottom_offset])
    cup_bottom_date = after_left.index[bottom_offset]

    if cup_bottom_price <= 0:
        return None
    if cup_bottom_price >= left_rim_price:
        return None  # no actual decline — not a cup

    # STEP 3 — right rim: the recovery peak that forms the "lip" of the
    # cup, found BEFORE any handle pullback. We walk forward from the
    # cup bottom and find the FIRST genuine local maximum (a turning
    # point where price has been rising into it and then pulls back,
    # however briefly) whose recovery clears the 50% hard gate. This
    # is what distinguishes "the rim, followed by a handle pullback"
    # from "still rallying, no rim yet" — a true local max requires a
    # pullback after it; if price never pulls back, there's no rim yet
    # and the pattern is correctly classified as Cup Only using the
    # most recent bar as a provisional (still-forming) rim.
    after_bottom = close.iloc[bottom_pos + 1:]
    if after_bottom.empty:
        return None

    decline = left_rim_price - cup_bottom_price
    if decline <= 0:
        return None

    right_rim_pos, right_rim_price = _find_right_rim(
        close, bottom_pos, decline, cup_bottom_price, timeframe
    )
    if right_rim_pos is None:
        return None

    recovery = (right_rim_price - cup_bottom_price) / decline

    # ── THE ONLY HARD GATE (re-verified against the actual chosen peak) ──
    if recovery < cfg.CUP_MIN_RECOVERY_PCT:
        return None

    right_rim_date = close.index[right_rim_pos]

    # STEP 4 — sanity checks
    if not (left_rim_pos < bottom_pos < right_rim_pos):
        return None

    # Staleness check: current price shouldn't have fallen far below
    # the right rim (i.e. this should still look like a live setup,
    # not a cup that fully resolved and reversed long ago).
    current_price = float(close.iloc[-1])
    if current_price < right_rim_price * cfg.CURRENT_PRICE_MIN_PCT_OF_RIM:
        return None   # already crashed back down — not a current setup

    return {
        "window_size": n,
        "left_rim_pos": left_rim_pos,
        "left_rim_date": left_rim_date,
        "left_rim_price": left_rim_price,
        "bottom_pos": bottom_pos,
        "cup_bottom_date": cup_bottom_date,
        "cup_bottom_price": cup_bottom_price,
        "right_rim_pos": right_rim_pos,
        "right_rim_date": right_rim_date,
        "right_rim_price": right_rim_price,
        "recovery_pct": recovery,
        "cup_depth_pct": (decline / left_rim_price) * 100.0,
    }


def _find_right_rim(
    close: pd.Series,
    bottom_pos: int,
    decline: float,
    cup_bottom_price: float,
    timeframe: str,
) -> tuple[Optional[int], Optional[float]]:
    """
    Walk forward from the cup bottom and return the position/price of
    the right rim: the first genuine local maximum (price rises into
    it, then pulls back by at least a small confirmation margin) whose
    recovery clears the hard gate. A "local maximum" requires a
    pullback of at least a small confirmation margin after the
    candidate peak, confirming it's a real turning point and not just
    a noisy tick on the way up.

    If price keeps making new highs all the way to the last bar (no
    pullback ever confirms a peak), the most recent bar is returned as
    a PROVISIONAL right rim — this is the legitimate "still rallying,
    rim still forming, no handle yet" case, surfaced as Cup Only.
    """
    after_bottom = close.iloc[bottom_pos + 1:]
    if after_bottom.empty:
        return None, None

    # Confirmation margin: how far price must pull back from a
    # candidate peak before we treat it as confirmed. Loose on
    # purpose — small relative to cup height — since detection is
    # meant to be maximally sensitive; this is just enough to reject
    # single-bar noise spikes as the "rim".
    confirm_margin = max(decline * 0.015, cup_bottom_price * 0.003)

    running_max = -float("inf")
    running_max_pos = None
    candidate_pos = None
    candidate_price = None

    values = after_bottom.values
    # Positions are simply consecutive integers starting at bottom_pos+1
    # — no need for expensive index.get_loc() lookups per bar.
    start_pos = bottom_pos + 1

    running_max = -float("inf")
    running_max_pos = None
    candidate_pos = None
    candidate_price = None

    for offset, price in enumerate(values):
        pos = start_pos + offset
        if price > running_max:
            running_max = price
            running_max_pos = pos
            candidate_pos = pos
            candidate_price = price
        else:
            if candidate_pos is not None and (running_max - price) >= confirm_margin:
                recovery = (candidate_price - cup_bottom_price) / decline
                if recovery >= cfg.CUP_MIN_RECOVERY_PCT:
                    return candidate_pos, float(candidate_price)
                candidate_pos = None

    if running_max_pos is not None:
        recovery = (running_max - cup_bottom_price) / decline
        if recovery >= cfg.CUP_MIN_RECOVERY_PCT:
            return running_max_pos, float(running_max)

    return None, None


# ─── Build full signal from a winning candidate ────────────────────────────

def _build_signal(
    df: pd.DataFrame,
    cand: dict,
    timeframe: str,
    symbol: str,
    rs_rating: float,
) -> Optional[CupHandleSignal]:
    n = len(df)
    window_start_pos = n - cand["window_size"]
    window = df.iloc[window_start_pos:]

    left_rim_price = cand["left_rim_price"]
    cup_bottom_price = cand["cup_bottom_price"]
    right_rim_price = cand["right_rim_price"]
    cup_depth_pct = cand["cup_depth_pct"]
    recovery_pct = cand["recovery_pct"] * 100.0

    cup_start_date = cand["left_rim_date"]
    cup_bottom_date = cand["cup_bottom_date"]
    cup_end_date = cand["right_rim_date"]
    cup_duration_bars = cand["right_rim_pos"] - cand["left_rim_pos"]

    # ── Cup shape classification ──
    cup_shape = _classify_cup_shape(window, cand)

    # ── Prior uptrend tag ──
    prior_uptrend_pct = _prior_uptrend(df, window_start_pos, cand["left_rim_pos"], cup_duration_bars)
    prior_uptrend_tag = _prior_uptrend_tag(prior_uptrend_pct)

    # ── Cup depth classification ──
    cup_depth_class, depth_verify_flag = _classify_cup_depth(cup_depth_pct)

    # ── Handle detection ──
    handle_info = _detect_handle(df, cand, timeframe)

    # ── Pivot point ──
    if handle_info["has_handle"]:
        pivot_point = handle_info["handle_high"]
    else:
        # right rim high = max High in the bars around the right rim close
        right_rim_pos_abs = window_start_pos + cand["right_rim_pos"]
        pivot_point = float(df["High"].iloc[right_rim_pos_abs])

    current_price = float(df["Close"].iloc[-1])
    price_vs_pivot_pct = ((current_price - pivot_point) / pivot_point) * 100.0 if pivot_point else 0.0

    # ── Signal type classification ──
    signal_type = _classify_signal_type(
        current_price, pivot_point, df, handle_info["has_handle"]
    )

    pattern_type = "Cup & Handle" if handle_info["has_handle"] else "Cup Only"

    # ── ATR at handle start (for Breakout Readiness 'ATR Contracting') ──
    atr_at_handle_start = _atr_at_position(
        df, handle_info.get("handle_start_pos_abs"), timeframe
    )

    # ── Quality score ──
    rim_symmetry_pct = (right_rim_price / left_rim_price) * 100.0 if left_rim_price else 0.0
    quality_score = _compute_quality_score(
        rim_symmetry_pct=rim_symmetry_pct,
        cup_depth_pct=cup_depth_pct,
        prior_uptrend_pct=prior_uptrend_pct,
        cup_shape=cup_shape,
        rs_rating=rs_rating,
        handle_quality_subscore=handle_info["handle_quality_subscore"],
        volume_dryup_cup=handle_info.get("cup_volume_dryup_score", "none"),
        mtf_confluence=False,  # set later by main.py after MTF pass
    )

    sig = CupHandleSignal(
        symbol=symbol,
        timeframe=timeframe,
        cup_start_date=_to_date(cup_start_date),
        cup_bottom_date=_to_date(cup_bottom_date),
        cup_end_date=_to_date(cup_end_date),
        left_rim_price=round(left_rim_price, 2),
        cup_bottom_price=round(cup_bottom_price, 2),
        right_rim_price=round(right_rim_price, 2),
        cup_depth_pct=round(cup_depth_pct, 2),
        cup_duration_bars=cup_duration_bars,
        cup_shape=cup_shape,
        recovery_pct=round(recovery_pct, 2),
        prior_uptrend_pct=round(prior_uptrend_pct, 2),
        prior_uptrend_tag=prior_uptrend_tag,
        cup_depth_class=cup_depth_class,
        cup_depth_verify_flag=depth_verify_flag,
        has_handle=handle_info["has_handle"],
        handle_start_date=_to_date(handle_info.get("handle_start_date")),
        handle_end_date=_to_date(handle_info.get("handle_end_date")),
        handle_low_price=handle_info.get("handle_low_price"),
        handle_depth_pct=handle_info.get("handle_depth_pct"),
        handle_duration_bars=handle_info.get("handle_duration_bars"),
        handle_quality_subscore=handle_info["handle_quality_subscore"],
        handle_vol_dryup_pts=handle_info["handle_vol_dryup_pts"],
        handle_tightness_pts=handle_info["handle_tightness_pts"],
        handle_higher_lows_pts=handle_info["handle_higher_lows_pts"],
        handle_close_high_pts=handle_info["handle_close_high_pts"],
        atr_at_handle_start=atr_at_handle_start,
        pivot_point=round(pivot_point, 2),
        current_price=round(current_price, 2),
        price_vs_pivot_pct=round(price_vs_pivot_pct, 2),
        signal_type=signal_type,
        pattern_type=pattern_type,
        already_breaking_out_handle=handle_info.get("already_breaking_out", False),
        quality_score=round(quality_score, 1),
        rim_symmetry_pct=round(rim_symmetry_pct, 2),
    )
    sig._df_ref = df
    return sig


# ─── Cup shape classification ──────────────────────────────────────────────

def _classify_cup_shape(window: pd.DataFrame, cand: dict) -> str:
    """
    U-shape: price doesn't make a new low after the midpoint of the cup
    (i.e. the bottom third of the cup duration contains the actual low).
    V-shape: sharp, narrow bottom — accepted, just scores lower.
    Irregular: anything not cleanly U or V (e.g. W / double bottom).
    """
    left_pos = cand["left_rim_pos"]
    right_pos = cand["right_rim_pos"]
    bottom_pos = cand["bottom_pos"]

    cup_duration = right_pos - left_pos
    if cup_duration <= 0:
        return "Irregular"

    bottom_fraction = (bottom_pos - left_pos) / cup_duration

    close = window["Close"].iloc[left_pos:right_pos + 1]
    if len(close) < 3:
        return "Irregular"

    # Check for a double-bottom / W shape: a second trough within 5% of
    # the cup bottom price elsewhere in the cup
    cup_bottom_price = cand["cup_bottom_price"]
    near_bottom = close[close <= cup_bottom_price * 1.05]
    distinct_trough_clusters = _count_distinct_clusters(near_bottom.index, window.index)

    if distinct_trough_clusters >= 2:
        return "Irregular"   # W-shape / double bottom

    # U-shape: bottom sits roughly in the middle third (0.33-0.67) of
    # the cup duration, and the decline/recovery legs are each a
    # reasonably gradual fraction of the cup (not instantaneous)
    if 0.30 <= bottom_fraction <= 0.70:
        return "U-Shape"

    # V-shape: sharp, narrow bottom (bottom near one edge, fast reversal)
    return "V-Shape"


def _count_distinct_clusters(near_bottom_index: pd.Index, full_index: pd.Index) -> int:
    """Count distinct contiguous clusters of bars near the cup bottom —
    used to detect W-shapes / double bottoms."""
    if len(near_bottom_index) == 0:
        return 0
    positions = sorted(full_index.get_indexer(near_bottom_index))
    clusters = 1
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] > 2:
            clusters += 1
    return clusters


# ─── Prior uptrend ──────────────────────────────────────────────────────────

def _prior_uptrend(
    df: pd.DataFrame, window_start_pos: int, left_rim_pos_in_window: int, cup_duration_bars: int
) -> float:
    """
    % gain in the bars immediately before the left rim, lookback equal
    to the cup's own duration (capped per spec). Returns 0 if there's
    insufficient prior history (can't penalise — just show 0/unknown).
    """
    left_rim_abs_pos = window_start_pos + left_rim_pos_in_window
    lookback = min(cup_duration_bars, 252)  # cap, see spec 3C-1
    if lookback <= 0:
        return 0.0

    prior_start_pos = left_rim_abs_pos - lookback
    if prior_start_pos < 0:
        prior_start_pos = 0
    if prior_start_pos >= left_rim_abs_pos:
        return 0.0

    prior_start_price = float(df["Close"].iloc[prior_start_pos])
    left_rim_price = float(df["Close"].iloc[left_rim_abs_pos])
    if prior_start_price <= 0:
        return 0.0

    return ((left_rim_price - prior_start_price) / prior_start_price) * 100.0


def _prior_uptrend_tag(pct: float) -> str:
    if pct >= cfg.PRIOR_UPTREND_STRONG_PCT:
        return "Strong Continuation"
    if pct >= cfg.PRIOR_UPTREND_MODERATE_PCT:
        return "Moderate Continuation"
    return "Possible Bottoming Pattern"


# ─── Cup depth classification ──────────────────────────────────────────────

def _classify_cup_depth(depth_pct: float) -> tuple[str, bool]:
    band_label = "Crash Recovery"
    for label, lo, hi in cfg.CUP_DEPTH_BANDS:
        if lo <= depth_pct < hi:
            band_label = label
            break
    verify_flag = depth_pct < cfg.CUP_VERY_SHALLOW_THRESHOLD
    return band_label, verify_flag


# ─── Handle detection ───────────────────────────────────────────────────────

def _detect_handle(df: pd.DataFrame, cand: dict, timeframe: str) -> dict:
    """
    Search for a handle in the bars after the right rim. Relaxed
    geometry per spec 3D. Returns a dict describing the handle (or
    has_handle=False if none found), plus the handle quality sub-score.
    """
    n = len(df)
    window_start_pos = n - cand["window_size"]
    right_rim_abs_pos = window_start_pos + cand["right_rim_pos"]

    min_bars, max_bars = cfg.HANDLE_SEARCH_BARS[timeframe]
    search_end = min(n, right_rim_abs_pos + 1 + max_bars)
    search_start = right_rim_abs_pos + 1

    no_handle_result = {
        "has_handle": False,
        "handle_quality_subscore": 0.0,
        "handle_vol_dryup_pts": 0.0,
        "handle_tightness_pts": 0.0,
        "handle_higher_lows_pts": 0.0,
        "handle_close_high_pts": 0.0,
        "handle_start_pos_abs": right_rim_abs_pos,  # for ATR-at-start fallback
        "already_breaking_out": False,
    }

    if search_start >= n:
        return no_handle_result

    handle_zone = df.iloc[search_start:search_end]
    if len(handle_zone) < 1:
        return no_handle_result

    cup_bottom_price = cand["cup_bottom_price"]
    left_rim_price = cand["left_rim_price"]
    right_rim_price = cand["right_rim_price"]
    cup_height = left_rim_price - cup_bottom_price

    handle_low_price = float(handle_zone["Low"].min())
    handle_low_date = handle_zone["Low"].idxmin()
    handle_high_price = float(handle_zone["High"].max())

    # Reject as "handle" if it breaks the loose floor — but still no
    # hard detection failure; just means we treat it as Cup Only
    min_allowed_low = cup_bottom_price + cfg.HANDLE_MIN_PCT_FROM_BOTTOM * cup_height
    if handle_low_price < min_allowed_low:
        return no_handle_result

    handle_depth_pct = ((right_rim_price - handle_low_price) / right_rim_price) * 100.0
    cup_depth_pct_abs = ((left_rim_price - cup_bottom_price) / left_rim_price) * 100.0
    if cup_depth_pct_abs > 0 and (handle_depth_pct / cup_depth_pct_abs) > cfg.HANDLE_MAX_DEPTH_RATIO:
        return no_handle_result

    already_breaking_out = handle_high_price > right_rim_price * (1 + cfg.HANDLE_MAX_BREAKOUT_PCT)

    handle_start_date = handle_zone.index[0]
    handle_end_date = handle_zone.index[-1]
    handle_duration_bars = len(handle_zone)

    if handle_duration_bars < min_bars and not already_breaking_out:
        # Too short to be a meaningful handle yet — treat as Cup Only /
        # Pre-Handle (still surfaced, just without handle-specific levels)
        return no_handle_result

    # ── Handle quality sub-score (0-20 pts) ──
    vol_pts, vol_label = _handle_volume_dryup_score(df, cand, search_start, search_end)
    tight_pts, tight_label = _handle_tightness_score(df, cand, search_start, search_end)
    higher_lows_pts = _handle_higher_lows_score(handle_zone)
    close_high_pts = _handle_close_near_high_score(handle_zone)

    handle_quality_subscore = vol_pts + tight_pts + higher_lows_pts + close_high_pts

    return {
        "has_handle": True,
        "handle_start_date": handle_start_date,
        "handle_end_date": handle_end_date,
        "handle_low_price": round(handle_low_price, 2),
        "handle_high": round(handle_high_price, 2),
        "handle_depth_pct": round(handle_depth_pct, 2),
        "handle_duration_bars": handle_duration_bars,
        "handle_quality_subscore": handle_quality_subscore,
        "handle_vol_dryup_pts": vol_pts,
        "handle_tightness_pts": tight_pts,
        "handle_higher_lows_pts": higher_lows_pts,
        "handle_close_high_pts": close_high_pts,
        "handle_start_pos_abs": search_start,
        "already_breaking_out": already_breaking_out,
        "cup_volume_dryup_score": vol_label,
    }


def _handle_volume_dryup_score(df, cand, search_start, search_end) -> tuple[float, str]:
    try:
        volume = clean_volume(df["Volume"])
    except Exception:
        return 0.0, "none"

    n = len(df)
    window_start_pos = n - cand["window_size"]
    left_rim_abs = window_start_pos + cand["left_rim_pos"]
    right_rim_abs = window_start_pos + cand["right_rim_pos"]

    cup_vol = volume.iloc[left_rim_abs:right_rim_abs + 1]
    handle_vol = volume.iloc[search_start:search_end]

    if cup_vol.empty or handle_vol.empty:
        return 0.0, "none"

    cup_avg = cup_vol.mean()
    handle_avg = handle_vol.mean()
    if cup_avg <= 0 or pd.isna(cup_avg):
        return 0.0, "none"

    ratio = handle_avg / cup_avg
    if ratio <= cfg.HQ_VOL_DRYUP_FULL_RATIO:
        return cfg.HQ_VOL_DRYUP_PTS["full"], "full"
    if ratio <= cfg.HQ_VOL_DRYUP_PARTIAL_RATIO:
        return cfg.HQ_VOL_DRYUP_PTS["partial"], "partial"
    return cfg.HQ_VOL_DRYUP_PTS["none"], "none"


def _handle_tightness_score(df, cand, search_start, search_end) -> tuple[float, str]:
    n = len(df)
    window_start_pos = n - cand["window_size"]
    left_rim_abs = window_start_pos + cand["left_rim_pos"]
    right_rim_abs = window_start_pos + cand["right_rim_pos"]

    cup_bars = df.iloc[left_rim_abs:right_rim_abs + 1]
    handle_bars = df.iloc[search_start:search_end]

    if cup_bars.empty or handle_bars.empty:
        return 0.0, "none"

    cup_range_ratio = ((cup_bars["High"] - cup_bars["Low"]) / cup_bars["Close"].replace(0, np.nan)).mean()
    handle_range_ratio = ((handle_bars["High"] - handle_bars["Low"]) / handle_bars["Close"].replace(0, np.nan)).mean()

    if pd.isna(cup_range_ratio) or cup_range_ratio <= 0 or pd.isna(handle_range_ratio):
        return 0.0, "none"

    ratio = handle_range_ratio / cup_range_ratio
    if ratio <= cfg.HQ_TIGHTNESS_FULL_RATIO:
        return cfg.HQ_TIGHTNESS_PTS["full"], "full"
    if ratio <= cfg.HQ_TIGHTNESS_PARTIAL_RATIO:
        return cfg.HQ_TIGHTNESS_PTS["partial"], "partial"
    return cfg.HQ_TIGHTNESS_PTS["none"], "none"


def _handle_higher_lows_score(handle_zone: pd.DataFrame) -> float:
    lows = handle_zone["Low"].values
    if len(lows) < 2:
        return cfg.HQ_HIGHER_LOWS_PTS["none"]

    # Sequential higher lows: check if at least 2 consecutive increases
    # among the per-bar lows (loosely — doesn't need to be every bar)
    increases = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
    if increases >= 2:
        return cfg.HQ_HIGHER_LOWS_PTS["sequential"]

    min_pos = int(np.argmin(lows))
    if min_pos >= len(lows) / 2:
        return cfg.HQ_HIGHER_LOWS_PTS["second_half"]

    if lows[-1] < lows[0]:
        return cfg.HQ_HIGHER_LOWS_PTS["none"]

    return cfg.HQ_HIGHER_LOWS_PTS["second_half"]


def _handle_close_near_high_score(handle_zone: pd.DataFrame) -> float:
    rng = (handle_zone["High"] - handle_zone["Low"]).replace(0, np.nan)
    close_pos = (handle_zone["Close"] - handle_zone["Low"]) / rng
    avg_close_pos = close_pos.mean()

    if pd.isna(avg_close_pos):
        return cfg.HQ_CLOSE_HIGH_PTS["none"]
    if avg_close_pos >= cfg.HQ_CLOSE_HIGH_FULL:
        return cfg.HQ_CLOSE_HIGH_PTS["full"]
    if avg_close_pos >= cfg.HQ_CLOSE_HIGH_PARTIAL:
        return cfg.HQ_CLOSE_HIGH_PTS["partial"]
    return cfg.HQ_CLOSE_HIGH_PTS["none"]


# ─── ATR at handle start ────────────────────────────────────────────────────

def _atr_at_position(df: pd.DataFrame, pos_abs: Optional[int], timeframe: str) -> Optional[float]:
    if pos_abs is None or pos_abs < 0 or pos_abs >= len(df):
        return None
    from indicators import atr
    try:
        atr_series = atr(df, period=cfg.ATR_PERIOD)
        val = atr_series.iloc[pos_abs]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


# ─── Signal type classification ────────────────────────────────────────────

def _classify_signal_type(
    current_price: float, pivot_point: float, df: pd.DataFrame, has_handle: bool
) -> str:
    if pivot_point <= 0:
        return "EARLY STAGE"

    price_vs_pivot = (current_price - pivot_point) / pivot_point

    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price

    if current_price >= pivot_point and current_price > prev_close:
        return "BREAKOUT NOW"
    if not has_handle:
        return "CUP ONLY"
    if -cfg.NEAR_BREAKOUT_THRESHOLD <= price_vs_pivot < 0 or (
        current_price >= pivot_point
    ):
        return "NEAR BREAKOUT"
    if -cfg.BASING_THRESHOLD <= price_vs_pivot < -cfg.NEAR_BREAKOUT_THRESHOLD:
        return "BASING"
    return "EARLY STAGE"


# ─── Quality score ──────────────────────────────────────────────────────────

def _compute_quality_score(
    rim_symmetry_pct: float,
    cup_depth_pct: float,
    prior_uptrend_pct: float,
    cup_shape: str,
    rs_rating: float,
    handle_quality_subscore: float,
    volume_dryup_cup: str,
    mtf_confluence: bool,
) -> float:
    score = 0.0

    # Rim symmetry
    if rim_symmetry_pct >= 90:
        score += cfg.QS_RIM_SYMMETRY["high"]
    elif rim_symmetry_pct >= 75:
        score += cfg.QS_RIM_SYMMETRY["mid"]
    else:
        score += cfg.QS_RIM_SYMMETRY["low"]

    # Cup depth
    if 15 <= cup_depth_pct <= 35:
        score += cfg.QS_CUP_DEPTH["classic"]
    elif 8 <= cup_depth_pct < 15 or 35 < cup_depth_pct <= 50:
        score += cfg.QS_CUP_DEPTH["near"]
    elif cup_depth_pct > 70:
        score += cfg.QS_CUP_DEPTH["crash"]
    else:
        score += cfg.QS_CUP_DEPTH["far"]

    # Prior uptrend
    if prior_uptrend_pct >= cfg.PRIOR_UPTREND_STRONG_PCT:
        score += cfg.QS_PRIOR_UPTREND["strong"]
    elif prior_uptrend_pct >= cfg.PRIOR_UPTREND_MODERATE_PCT:
        score += cfg.QS_PRIOR_UPTREND["moderate"]
    else:
        score += cfg.QS_PRIOR_UPTREND["weak"]

    # Volume dry-up during cup
    if volume_dryup_cup == "full":
        score += cfg.QS_VOLUME_DRYUP["full"]
    elif volume_dryup_cup == "partial":
        score += cfg.QS_VOLUME_DRYUP["partial"]
    else:
        score += cfg.QS_VOLUME_DRYUP["none"]

    # Shape
    if cup_shape == "U-Shape":
        score += cfg.QS_SHAPE["u_shape"]
    elif cup_shape == "V-Shape":
        score += cfg.QS_SHAPE["v_shape"]
    else:
        score += cfg.QS_SHAPE["irregular"]

    # RS Rating
    if rs_rating >= cfg.RS_LEADER_THRESHOLD:
        score += cfg.QS_RS_RATING["leader"]
    elif rs_rating >= cfg.RS_RISING_THRESHOLD:
        score += cfg.QS_RS_RATING["rising"]
    else:
        score += cfg.QS_RS_RATING["lagging"]

    # MTF confluence (set to False here, added later by main.py if needed)
    if mtf_confluence:
        score += cfg.QS_MTF_CONFLUENCE

    # Handle quality sub-score
    score += min(handle_quality_subscore, cfg.QS_HANDLE_MAX)

    return min(score, cfg.QUALITY_SCORE_CAP)


def apply_mtf_confluence_bonus(score: float) -> float:
    """Called externally by main.py once confluence is known across
    timeframes for a symbol — adds the +10 bonus and re-caps."""
    return min(score + cfg.QS_MTF_CONFLUENCE, cfg.QUALITY_SCORE_CAP)


def quality_band(score: float) -> str:
    if score >= cfg.QUALITY_BAND_HIGH:
        return "High Quality"
    if score >= cfg.QUALITY_BAND_MEDIUM:
        return "Medium Quality"
    return "Low Quality"


# ─── Misc helpers ───────────────────────────────────────────────────────────

def _to_date(ts) -> Optional[date]:
    if ts is None:
        return None
    if isinstance(ts, date) and not isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).date()
