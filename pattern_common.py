"""
NSE Multi-Pattern Scanner - Shared Pattern Infrastructure
=============================================================
Everything Double Bottom, VCP, and Ascending Triangle detection have
in common, so each detector file only has to implement the geometry
that's actually unique to it.

── Why PatternSignal reuses CupHandleSignal's field names ──
readiness.py and entry_exit.py are written against `CupHandleSignal`,
but Python doesn't enforce that at runtime — both modules only ever
touch attributes (`sig.pivot_point`, `sig.has_handle`, `sig.timeframe`,
...), never `isinstance(sig, CupHandleSignal)`. So instead of writing
three more copies of "is this actionable now" (readiness.py) and
"where's the stop, target, position size" (entry_exit.py) logic,
`PatternSignal` below exposes the SAME attribute names, with the
concepts generalised:

  CupHandleSignal field      | generalised meaning for the new patterns
  ---------------------------|---------------------------------------------
  left_rim_price             | the structure's key HIGH reference (neckline
                              | for Double Bottom, base high for VCP, flat
                              | resistance for Ascending Triangle)
  cup_bottom_price           | the structure's key LOW reference, paired
                              | with left_rim_price for the measured-move
                              | target (entry + (left_rim - cup_bottom))
  right_rim_price            | the swing high right before the final base
  has_handle                 | whether a final tight base/pullback formed
                              | right before the pivot (always True for
                              | VCP/Triangle once the hard gate is met —
                              | their last contraction/approach IS that
                              | base; optional for Double Bottom)
  handle_low_price /         | the final base's low / start / end — used
  handle_start_date / etc.   | for the stop-loss and its quality subscore
  pivot_point                | breakout level (unchanged concept)

This is not a hack layered on top of unrelated fields — "resistance
minus deepest low, projected off the breakout" is a standard measured-
move target for double bottoms and triangles, and "stop below the
final, tightest contraction" is exactly how VCP/triangle breakouts are
traded in practice. The mapping is semantically correct, not just
attribute-compatible.

PatternSignal ALSO carries new, purely additive fields that
cup_handle_detector.py's CupHandleSignal doesn't declare:
`pattern_family` (stable key for UI tab filtering), `key_points` /
`trendlines` (generic chart annotations — see chart_export.py),
`confirmation_candle` (bullish candle at the breakout, from
candlestick_patterns.py), and `extra` (pattern-specific numbers for
the explanation text). `ensure_generic_fields()` retrofits these onto
a plain CupHandleSignal too (dataclasses without `slots=True` allow
arbitrary attribute assignment), so main.py can treat every detector's
output uniformly without ever touching cup_handle_detector.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from indicators import clean_volume


# ─── Generalised signal container ──────────────────────────────────────────

@dataclass
class PatternSignal:
    symbol: str
    timeframe: str
    pattern_type: str               # display name, e.g. "VCP (3 Contractions)"
    pattern_family: str             # stable key: double_bottom / vcp / ascending_triangle

    cup_start_date: date
    cup_bottom_date: date
    cup_end_date: date
    left_rim_price: float
    cup_bottom_price: float
    right_rim_price: float
    cup_depth_pct: float
    cup_duration_bars: int
    cup_shape: str
    recovery_pct: float

    prior_uptrend_pct: float
    prior_uptrend_tag: str
    cup_depth_class: str
    cup_depth_verify_flag: bool

    has_handle: bool
    handle_start_date: Optional[date] = None
    handle_end_date: Optional[date] = None
    handle_low_price: Optional[float] = None
    handle_depth_pct: Optional[float] = None
    handle_duration_bars: Optional[int] = None

    handle_quality_subscore: float = 0.0
    handle_vol_dryup_pts: float = 0.0
    handle_tightness_pts: float = 0.0
    handle_higher_lows_pts: float = 0.0
    handle_close_high_pts: float = 0.0

    atr_at_handle_start: Optional[float] = None

    pivot_point: float = 0.0
    current_price: float = 0.0
    price_vs_pivot_pct: float = 0.0
    signal_type: str = ""
    already_breaking_out_handle: bool = False

    quality_score: float = 0.0
    rim_symmetry_pct: float = 0.0

    mtf_confluence: bool = False
    mtf_timeframes: str = ""

    # ── Generic annotations (consumed by chart_export.py / template.html) ──
    key_points: list = field(default_factory=list)     # [{date,price,label,role}]
    trendlines: list = field(default_factory=list)      # [{role,style,color,points:[[date,price],...]}]
    structure_note: str = ""
    confirmation_candle: Optional[dict] = None
    extra: dict = field(default_factory=dict)

    _df_ref: object = field(default=None, repr=False, compare=False)


def ensure_generic_fields(sig) -> None:
    """
    Retrofit `pattern_family` / `key_points` / `trendlines` /
    `confirmation_candle` / `structure_note` / `extra` onto a signal
    that doesn't already declare them — i.e. a legacy CupHandleSignal.
    New-pattern detectors set these directly, so this is a no-op for
    them. Mutates in place.
    """
    if getattr(sig, "pattern_family", None) is None:
        sig.pattern_family = "cup_handle"
    if not getattr(sig, "key_points", None) and not getattr(sig, "trendlines", None):
        kp, tl = _derive_annotations_from_cup_handle(sig)
        sig.key_points = kp
        sig.trendlines = tl
    if not hasattr(sig, "confirmation_candle"):
        sig.confirmation_candle = None
    if not hasattr(sig, "structure_note") or not sig.structure_note:
        sig.structure_note = _default_structure_note(sig)
    if not hasattr(sig, "extra"):
        sig.extra = {}


def _derive_annotations_from_cup_handle(sig) -> tuple[list[dict], list[dict]]:
    key_points: list[dict] = []
    if sig.cup_start_date:
        key_points.append({"date": str(sig.cup_start_date), "price": sig.left_rim_price,
                            "label": "Left Rim", "role": "peak"})
    if sig.cup_bottom_date:
        key_points.append({"date": str(sig.cup_bottom_date), "price": sig.cup_bottom_price,
                            "label": "Cup Bottom", "role": "trough"})
    if sig.cup_end_date:
        key_points.append({"date": str(sig.cup_end_date), "price": sig.right_rim_price,
                            "label": "Right Rim", "role": "peak"})
    if sig.has_handle and sig.handle_end_date and sig.handle_low_price is not None:
        key_points.append({"date": str(sig.handle_end_date), "price": sig.handle_low_price,
                            "label": "Handle Low", "role": "trough"})

    trendlines: list[dict] = []
    if sig.cup_start_date and sig.cup_bottom_date and sig.cup_end_date:
        trendlines.append({
            "role": "structure", "style": "solid", "color": "#bc8cff", "width": 2,
            "points": [[str(sig.cup_start_date), sig.left_rim_price],
                       [str(sig.cup_bottom_date), sig.cup_bottom_price],
                       [str(sig.cup_end_date), sig.right_rim_price]],
        })
    if sig.has_handle and sig.handle_start_date and sig.handle_end_date and sig.handle_low_price is not None:
        trendlines.append({
            "role": "base_top", "style": "dashed", "color": "#d29922", "width": 1,
            "points": [[str(sig.handle_start_date), sig.right_rim_price],
                       [str(sig.handle_end_date), sig.right_rim_price]],
        })
        trendlines.append({
            "role": "base_bottom", "style": "dashed", "color": "#d29922", "width": 1,
            "points": [[str(sig.handle_start_date), sig.handle_low_price],
                       [str(sig.handle_end_date), sig.handle_low_price]],
        })
    return key_points, trendlines


def _default_structure_note(sig) -> str:
    if sig.pattern_family == "cup_handle":
        return f"{sig.cup_shape} cup, {sig.cup_depth_pct:.1f}% deep" + (
            f", handle quality {sig.handle_quality_subscore:.0f}/20" if sig.has_handle else ", no handle yet"
        )
    return sig.cup_shape or ""


# ─── Vocabulary per pattern family (used by explain.py / chart_export.py) ──

PATTERN_VOCAB = {
    "cup_handle": {
        "name": "Cup & Handle", "structure_noun": "cup", "structure_noun_cap": "Cup",
        "rim_noun": "rim", "base_noun": "handle", "base_noun_cap": "Handle",
        "preconfirm_label": "CUP ONLY", "depth_noun": "cup depth", "shape_noun": "cup shape",
        "high_label": "Left Rim", "low_label": "Cup Bottom", "confirm_label": "Right Rim",
    },
    "double_bottom": {
        "name": "Double Bottom", "structure_noun": "double bottom", "structure_noun_cap": "Double Bottom",
        "rim_noun": "neckline", "base_noun": "confirmation base", "base_noun_cap": "Confirmation Base",
        "preconfirm_label": "BASE FORMING", "depth_noun": "pattern depth", "shape_noun": "bottom symmetry",
        "high_label": "Neckline", "low_label": "Bottom 1", "confirm_label": "Neckline",
    },
    "vcp": {
        "name": "VCP", "structure_noun": "VCP base", "structure_noun_cap": "VCP Base",
        "rim_noun": "pivot resistance", "base_noun": "final contraction", "base_noun_cap": "Final Contraction",
        "preconfirm_label": "BASE FORMING", "depth_noun": "base depth", "shape_noun": "contraction pattern",
        "high_label": "Base High", "low_label": "Deepest Contraction", "confirm_label": "Pivot Resistance",
    },
    "ascending_triangle": {
        "name": "Ascending Triangle", "structure_noun": "ascending triangle", "structure_noun_cap": "Ascending Triangle",
        "rim_noun": "resistance", "base_noun": "final approach", "base_noun_cap": "Final Approach",
        "preconfirm_label": "BASE FORMING", "depth_noun": "triangle height", "shape_noun": "triangle quality",
        "high_label": "Resistance", "low_label": "First Support Low", "confirm_label": "Resistance",
    },
}


def vocab_for(pattern_family: Optional[str]) -> dict:
    return PATTERN_VOCAB.get(pattern_family or "cup_handle", PATTERN_VOCAB["cup_handle"])


# ─── Zigzag swing detection (used by VCP + Ascending Triangle) ────────────

def zigzag_pivots(close: pd.Series, threshold_pct: float) -> list[dict]:
    """
    Percentage-reversal zigzag over a Close series. Returns pivots in
    chronological order: [{"pos": int, "date": Timestamp, "price": float,
    "kind": "high"|"low", "provisional": bool}, ...].

    The LAST pivot is always marked provisional=True — it's the
    current, still-forming extreme (hasn't been confirmed by a further
    reversal yet), which is exactly the "as of today" edge every other
    detector in this codebase needs to classify signal_type / pivot.
    """
    n = len(close)
    if n < 2:
        return []
    vals = close.values
    idx = close.index

    trend = None
    anchor_pos = 0
    anchor_price = float(vals[0])
    i = 1
    while i < n and trend is None:
        if anchor_price:
            change = (vals[i] - anchor_price) / anchor_price
            if change >= threshold_pct:
                trend = 1
            elif change <= -threshold_pct:
                trend = -1
        i += 1
    if trend is None:
        return []

    pivots = [{"pos": anchor_pos, "date": idx[anchor_pos], "price": anchor_price,
               "kind": "low" if trend == 1 else "high", "provisional": False}]

    extreme_pos = i - 1
    extreme_price = float(vals[i - 1])

    for j in range(i, n):
        price = float(vals[j])
        if trend == 1:
            if price > extreme_price:
                extreme_price, extreme_pos = price, j
            elif extreme_price and (extreme_price - price) / extreme_price >= threshold_pct:
                pivots.append({"pos": extreme_pos, "date": idx[extreme_pos], "price": extreme_price,
                               "kind": "high", "provisional": False})
                trend, extreme_price, extreme_pos = -1, price, j
        else:
            if price < extreme_price:
                extreme_price, extreme_pos = price, j
            elif extreme_price and (price - extreme_price) / extreme_price >= threshold_pct:
                pivots.append({"pos": extreme_pos, "date": idx[extreme_pos], "price": extreme_price,
                               "kind": "low", "provisional": False})
                trend, extreme_price, extreme_pos = 1, price, j

    pivots.append({"pos": extreme_pos, "date": idx[extreme_pos], "price": extreme_price,
                   "kind": "high" if trend == 1 else "low", "provisional": True})
    return pivots


# ─── Generalised signal-type classification ────────────────────────────────

def classify_signal_type(current_price: float, pivot_point: float, prev_close: float,
                          has_handle: bool) -> str:
    """Generalised version of cup_handle_detector._classify_signal_type,
    using the generic 'BASE FORMING' label instead of 'CUP ONLY'."""
    if pivot_point <= 0:
        return "EARLY STAGE"
    price_vs_pivot = (current_price - pivot_point) / pivot_point

    if current_price >= pivot_point and current_price > prev_close:
        return "BREAKOUT NOW"
    if not has_handle:
        return cfg.GENERIC_PRECONFIRM_SIGNAL_TYPE
    if -cfg.NEAR_BREAKOUT_THRESHOLD <= price_vs_pivot < 0 or current_price >= pivot_point:
        return "NEAR BREAKOUT"
    if -cfg.BASING_THRESHOLD <= price_vs_pivot < -cfg.NEAR_BREAKOUT_THRESHOLD:
        return "BASING"
    return "EARLY STAGE"


# ─── Generalised depth / trend classification (reuses cup&handle bands) ───

def classify_depth(depth_pct: float) -> tuple[str, bool]:
    band_label = "Crash Recovery"
    for label, lo, hi in cfg.GENERIC_DEPTH_BANDS:
        if lo <= depth_pct < hi:
            band_label = label
            break
    verify_flag = depth_pct < cfg.CUP_VERY_SHALLOW_THRESHOLD
    return band_label, verify_flag


def prior_trend_pct(df: pd.DataFrame, structure_start_abs: int, duration_bars: int) -> float:
    """% gain in the bars immediately before the structure started,
    lookback equal to the structure's own duration (capped)."""
    lookback = min(max(duration_bars, 1), 252)
    prior_start_pos = max(structure_start_abs - lookback, 0)
    if prior_start_pos >= structure_start_abs:
        return 0.0
    prior_start_price = float(df["Close"].iloc[prior_start_pos])
    structure_start_price = float(df["Close"].iloc[structure_start_abs])
    if prior_start_price <= 0:
        return 0.0
    return ((structure_start_price - prior_start_price) / prior_start_price) * 100.0


def prior_trend_tag(pct: float) -> str:
    if pct >= cfg.PRIOR_UPTREND_STRONG_PCT:
        return "Strong Continuation"
    if pct >= cfg.PRIOR_UPTREND_MODERATE_PCT:
        return "Moderate Continuation"
    return "Possible Bottoming Pattern"


# ─── Generalised final-base ("handle") quality scoring ────────────────────

def score_final_base_quality(df: pd.DataFrame, structure_start_abs: int, structure_end_abs: int,
                              base_start_abs: int, base_end_abs: int) -> dict:
    """
    Generalised version of cup_handle_detector's four `_handle_*_score`
    helpers. `structure_*` spans the WHOLE formation (used as the
    baseline for "did volume/range contract in the final base
    relative to the formation as a whole"); `base_*` spans just the
    final base/handle/contraction being scored.
    """
    structure_start_abs = max(structure_start_abs, 0)
    base_end_abs = min(base_end_abs, len(df))
    if base_start_abs >= base_end_abs or structure_start_abs >= structure_end_abs:
        return _empty_base_quality()

    structure_bars = df.iloc[structure_start_abs:structure_end_abs + 1]
    base_bars = df.iloc[base_start_abs:base_end_abs]
    if structure_bars.empty or base_bars.empty:
        return _empty_base_quality()

    vol_pts, vol_label = _vol_dryup_score(df, structure_bars, base_bars)
    tight_pts, tight_label = _tightness_score(structure_bars, base_bars)
    higher_lows_pts = _higher_lows_score(base_bars)
    close_high_pts = _close_near_high_score(base_bars)

    return {
        "handle_quality_subscore": vol_pts + tight_pts + higher_lows_pts + close_high_pts,
        "handle_vol_dryup_pts": vol_pts,
        "handle_tightness_pts": tight_pts,
        "handle_higher_lows_pts": higher_lows_pts,
        "handle_close_high_pts": close_high_pts,
        "volume_dryup_label": vol_label,
        "tightness_label": tight_label,
    }


def _empty_base_quality() -> dict:
    return {
        "handle_quality_subscore": 0.0, "handle_vol_dryup_pts": 0.0,
        "handle_tightness_pts": 0.0, "handle_higher_lows_pts": 0.0,
        "handle_close_high_pts": 0.0, "volume_dryup_label": "none", "tightness_label": "none",
    }


def _vol_dryup_score(df, structure_bars, base_bars) -> tuple[float, str]:
    try:
        structure_vol = clean_volume(structure_bars["Volume"])
        base_vol = clean_volume(base_bars["Volume"])
    except Exception:
        return 0.0, "none"
    structure_avg = structure_vol.mean()
    base_avg = base_vol.mean()
    if not structure_avg or pd.isna(structure_avg):
        return 0.0, "none"
    ratio = base_avg / structure_avg
    if ratio <= cfg.HQ_VOL_DRYUP_FULL_RATIO:
        return cfg.HQ_VOL_DRYUP_PTS["full"], "full"
    if ratio <= cfg.HQ_VOL_DRYUP_PARTIAL_RATIO:
        return cfg.HQ_VOL_DRYUP_PTS["partial"], "partial"
    return cfg.HQ_VOL_DRYUP_PTS["none"], "none"


def _tightness_score(structure_bars, base_bars) -> tuple[float, str]:
    structure_range_ratio = ((structure_bars["High"] - structure_bars["Low"]) /
                              structure_bars["Close"].replace(0, np.nan)).mean()
    base_range_ratio = ((base_bars["High"] - base_bars["Low"]) /
                         base_bars["Close"].replace(0, np.nan)).mean()
    if pd.isna(structure_range_ratio) or structure_range_ratio <= 0 or pd.isna(base_range_ratio):
        return 0.0, "none"
    ratio = base_range_ratio / structure_range_ratio
    if ratio <= cfg.HQ_TIGHTNESS_FULL_RATIO:
        return cfg.HQ_TIGHTNESS_PTS["full"], "full"
    if ratio <= cfg.HQ_TIGHTNESS_PARTIAL_RATIO:
        return cfg.HQ_TIGHTNESS_PTS["partial"], "partial"
    return cfg.HQ_TIGHTNESS_PTS["none"], "none"


def _higher_lows_score(base_bars: pd.DataFrame) -> float:
    lows = base_bars["Low"].values
    if len(lows) < 2:
        return cfg.HQ_HIGHER_LOWS_PTS["none"]
    increases = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
    if increases >= 2:
        return cfg.HQ_HIGHER_LOWS_PTS["sequential"]
    min_pos = int(np.argmin(lows))
    if min_pos >= len(lows) / 2:
        return cfg.HQ_HIGHER_LOWS_PTS["second_half"]
    if lows[-1] < lows[0]:
        return cfg.HQ_HIGHER_LOWS_PTS["none"]
    return cfg.HQ_HIGHER_LOWS_PTS["second_half"]


def _close_near_high_score(base_bars: pd.DataFrame) -> float:
    rng = (base_bars["High"] - base_bars["Low"]).replace(0, np.nan)
    close_pos = (base_bars["Close"] - base_bars["Low"]) / rng
    avg = close_pos.mean()
    if pd.isna(avg):
        return cfg.HQ_CLOSE_HIGH_PTS["none"]
    if avg >= cfg.HQ_CLOSE_HIGH_FULL:
        return cfg.HQ_CLOSE_HIGH_PTS["full"]
    if avg >= cfg.HQ_CLOSE_HIGH_PARTIAL:
        return cfg.HQ_CLOSE_HIGH_PTS["partial"]
    return cfg.HQ_CLOSE_HIGH_PTS["none"]


# ─── Generalised quality score (0-100, same weights as Cup & Handle) ──────

def compute_quality_score_generic(
    rim_symmetry_pct: float, depth_pct: float, prior_trend_pct_val: float,
    shape_bucket: str, rs_rating: float, handle_quality_subscore: float,
    volume_dryup_label: str, mtf_confluence: bool,
) -> float:
    """shape_bucket in {'strong', 'moderate', 'weak'} maps onto the same
    point buckets Cup & Handle uses for u_shape/v_shape/irregular, so
    scores stay on a comparable 0-100 scale across every pattern type
    shown in the same dashboard."""
    score = 0.0

    if rim_symmetry_pct >= 90:
        score += cfg.QS_RIM_SYMMETRY["high"]
    elif rim_symmetry_pct >= 75:
        score += cfg.QS_RIM_SYMMETRY["mid"]
    else:
        score += cfg.QS_RIM_SYMMETRY["low"]

    if 15 <= depth_pct <= 35:
        score += cfg.QS_CUP_DEPTH["classic"]
    elif 8 <= depth_pct < 15 or 35 < depth_pct <= 50:
        score += cfg.QS_CUP_DEPTH["near"]
    elif depth_pct > 70:
        score += cfg.QS_CUP_DEPTH["crash"]
    else:
        score += cfg.QS_CUP_DEPTH["far"]

    if prior_trend_pct_val >= cfg.PRIOR_UPTREND_STRONG_PCT:
        score += cfg.QS_PRIOR_UPTREND["strong"]
    elif prior_trend_pct_val >= cfg.PRIOR_UPTREND_MODERATE_PCT:
        score += cfg.QS_PRIOR_UPTREND["moderate"]
    else:
        score += cfg.QS_PRIOR_UPTREND["weak"]

    if volume_dryup_label == "full":
        score += cfg.QS_VOLUME_DRYUP["full"]
    elif volume_dryup_label == "partial":
        score += cfg.QS_VOLUME_DRYUP["partial"]
    else:
        score += cfg.QS_VOLUME_DRYUP["none"]

    if shape_bucket == "strong":
        score += cfg.QS_SHAPE["u_shape"]
    elif shape_bucket == "moderate":
        score += cfg.QS_SHAPE["v_shape"]
    else:
        score += cfg.QS_SHAPE["irregular"]

    if rs_rating >= cfg.RS_LEADER_THRESHOLD:
        score += cfg.QS_RS_RATING["leader"]
    elif rs_rating >= cfg.RS_RISING_THRESHOLD:
        score += cfg.QS_RS_RATING["rising"]
    else:
        score += cfg.QS_RS_RATING["lagging"]

    if mtf_confluence:
        score += cfg.QS_MTF_CONFLUENCE

    score += min(handle_quality_subscore, cfg.QS_HANDLE_MAX)

    return min(score, cfg.QUALITY_SCORE_CAP)


def apply_mtf_confluence_bonus(score: float) -> float:
    return min(score + cfg.QS_MTF_CONFLUENCE, cfg.QUALITY_SCORE_CAP)


# ─── ATR at a bar position (shared helper) ─────────────────────────────────

def atr_at_position(df: pd.DataFrame, pos_abs: Optional[int]) -> Optional[float]:
    if pos_abs is None or pos_abs < 0 or pos_abs >= len(df):
        return None
    from indicators import atr
    try:
        val = atr(df, period=cfg.ATR_PERIOD).iloc[pos_abs]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def to_date(ts) -> Optional[date]:
    if ts is None:
        return None
    if isinstance(ts, date) and not isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).date()
