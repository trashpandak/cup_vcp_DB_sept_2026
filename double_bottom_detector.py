"""
NSE Multi-Pattern Scanner - Double Bottom Detection Engine
===============================================================
Same philosophy as cup_handle_detector.py: ONE hard gate (the
recovery off the second bottom must clear 50% of the decline from the
neckline down to that bottom), everything else — bottom symmetry,
prior trend, shape, volume — is scored/tagged, never used to reject.

Geometry:
    prior high -> BOTTOM 1 -> NECKLINE (peak) -> BOTTOM 2 -> recovery
                                                              -> pivot = neckline
                                                                 (or the top of a
                                                                  final tight base,
                                                                  if one has formed)

Bottom 2 is allowed to sit anywhere from 10% below to 8% above
Bottom 1's price (a slight undercut is a common, often bullish
"shakeout" — a slightly HIGHER second low is the classic, strongest
variant). Emits `pattern_common.PatternSignal`, which readiness.py and
entry_exit.py consume exactly like a CupHandleSignal (see
pattern_common.py's module docstring for the field mapping).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
import pattern_common as pc
from indicators import clean_volume


def detect_double_bottom(df: pd.DataFrame, timeframe: str, symbol: str,
                          rs_rating: float = 50.0) -> list[pc.PatternSignal]:
    min_bars_cfg = cfg.MIN_BARS_BY_TIMEFRAME.get(timeframe)
    if min_bars_cfg is None or len(df) < min_bars_cfg:
        return []

    min_w, max_w = cfg.DB_LOOKBACKS.get(timeframe, (20, 300))
    if len(df) < min_w:
        return []

    n = len(df)
    candidates: list[dict] = []
    for window_size in range(min_w, min(max_w, n) + 1):
        window = df.iloc[n - window_size:]
        cand = _evaluate_window(window, timeframe)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return []

    best = max(candidates, key=lambda c: (c["window_size"], round(c["recovery_pct"], 4)))
    sig = _build_signal(df, best, timeframe, symbol, rs_rating)
    return [sig] if sig is not None else []


# ─── Window evaluation ──────────────────────────────────────────────────────

def _evaluate_window(window: pd.DataFrame, timeframe: str) -> Optional[dict]:
    close = window["Close"]
    n = len(window)
    if n < 6:
        return None

    # STEP 1 — Bottom 1: lowest close in the first half of the window
    zone1_end = max(2, int(n * cfg.DB_BOTTOM1_ZONE_FRACTION))
    zone1 = close.iloc[:zone1_end]
    b1_offset = int(np.argmin(zone1.values))
    bottom1_pos = b1_offset
    bottom1_price = float(zone1.iloc[b1_offset])
    bottom1_date = zone1.index[b1_offset]
    if bottom1_price <= 0:
        return None

    # STEP 2 — Neckline: first genuine local max after Bottom 1 that
    # bounces meaningfully off it (same "confirmed peak" walk cup&handle
    # uses for its right rim, applied here to find the middle peak).
    after_b1 = close.iloc[bottom1_pos + 1:]
    if len(after_b1) < 3:
        return None
    peak_pos, peak_price = _find_confirmed_peak(
        close, bottom1_pos, bottom1_price, cfg.DB_MIN_BOUNCE_FROM_BOTTOM1_PCT
    )
    if peak_pos is None:
        return None
    peak_date = close.index[peak_pos]

    # STEP 3 — Bottom 2: lowest close after the neckline peak
    after_peak = close.iloc[peak_pos + 1:]
    if len(after_peak) < 2:
        return None
    b2_offset = int(np.argmin(after_peak.values))
    bottom2_pos = peak_pos + 1 + b2_offset
    bottom2_price = float(after_peak.values[b2_offset])
    bottom2_date = after_peak.index[b2_offset]
    if bottom2_price <= 0 or bottom2_price >= peak_price:
        return None

    # Similarity gate: bottom2 within [-max_undercut, +similarity] of bottom1
    diff_pct = (bottom2_price - bottom1_price) / bottom1_price
    if diff_pct < -cfg.DB_MAX_UNDERCUT_PCT or diff_pct > cfg.DB_BOTTOM_SIMILARITY_PCT:
        return None

    decline2 = peak_price - bottom2_price
    if decline2 <= 0:
        return None

    # STEP 4 — THE ONLY HARD GATE: recovery off bottom2 >= 50% of decline2
    recovery_zone = close.iloc[bottom2_pos:]
    if recovery_zone.empty:
        return None
    best_recovery_price = float(recovery_zone.max())
    recovery_pct = (best_recovery_price - bottom2_price) / decline2
    if recovery_pct < cfg.DB_MIN_RECOVERY_PCT:
        return None

    # Recency guard — current price shouldn't have fallen far below the neckline
    current_price = float(close.iloc[-1])
    if current_price < peak_price * cfg.DB_CURRENT_PRICE_MIN_PCT_OF_NECKLINE:
        return None

    if not (bottom1_pos < peak_pos < bottom2_pos):
        return None

    return {
        "window_size": n,
        "bottom1_pos": bottom1_pos, "bottom1_date": bottom1_date, "bottom1_price": bottom1_price,
        "peak_pos": peak_pos, "peak_date": peak_date, "peak_price": peak_price,
        "bottom2_pos": bottom2_pos, "bottom2_date": bottom2_date, "bottom2_price": bottom2_price,
        "recovery_pct": recovery_pct,
        "diff_pct": diff_pct * 100.0,
        "depth_pct": (decline2 / peak_price) * 100.0,
    }


def _find_confirmed_peak(close: pd.Series, from_pos: int, ref_low_price: float,
                          min_bounce_frac: float) -> tuple[Optional[int], Optional[float]]:
    """Walk forward from `from_pos` and return the first genuine local
    max (price rises into it, then pulls back) that bounces at least
    `min_bounce_frac` off `ref_low_price`. Falls back to the running
    max (provisional, still-rallying case) if it alone clears the
    bounce requirement but never gets a confirming pullback — mirrored
    from cup_handle_detector._find_right_rim."""
    after = close.iloc[from_pos + 1:]
    if after.empty:
        return None, None
    values = after.values
    start_pos = from_pos + 1
    confirm_margin = max(ref_low_price * min_bounce_frac * 0.25, ref_low_price * 0.003)

    running_max, running_max_pos = -float("inf"), None
    candidate_pos, candidate_price = None, None

    for offset, price in enumerate(values):
        pos = start_pos + offset
        if price > running_max:
            running_max, running_max_pos = price, pos
            candidate_pos, candidate_price = pos, price
        else:
            if candidate_pos is not None and (running_max - price) >= confirm_margin:
                bounce = (candidate_price - ref_low_price) / ref_low_price
                if bounce >= min_bounce_frac:
                    return candidate_pos, float(candidate_price)
                candidate_pos = None

    if running_max_pos is not None:
        bounce = (running_max - ref_low_price) / ref_low_price
        if bounce >= min_bounce_frac:
            return running_max_pos, float(running_max)
    return None, None


# ─── Build full signal ──────────────────────────────────────────────────────

def _build_signal(df: pd.DataFrame, cand: dict, timeframe: str, symbol: str,
                   rs_rating: float) -> Optional[pc.PatternSignal]:
    n = len(df)
    window_start_pos = n - cand["window_size"]
    bottom1_abs = window_start_pos + cand["bottom1_pos"]
    peak_abs = window_start_pos + cand["peak_pos"]
    bottom2_abs = window_start_pos + cand["bottom2_pos"]

    bottom1_price, peak_price, bottom2_price = cand["bottom1_price"], cand["peak_price"], cand["bottom2_price"]
    depth_pct = cand["depth_pct"]
    duration_bars = bottom2_abs - bottom1_abs

    # ── Shape / symmetry classification ──
    diff_pct = cand["diff_pct"]
    if abs(diff_pct) <= cfg.DB_EVEN_TOLERANCE_PCT * 100:
        shape, shape_bucket = "Even Double Bottom", "strong"
    elif diff_pct > 0:
        shape, shape_bucket = "Ascending Double Bottom", "strong"
    elif diff_pct >= -5.0:
        shape, shape_bucket = "Descending Double Bottom", "moderate"
    else:
        shape, shape_bucket = "Descending Double Bottom (Shakeout)", "weak"

    rim_symmetry_pct = max(0.0, 100.0 - abs(diff_pct) * 2.0)

    # ── Prior trend into bottom 1 ──
    prior_pct = pc.prior_trend_pct(df, bottom1_abs, duration_bars)
    prior_tag = pc.prior_trend_tag(prior_pct)

    depth_class, depth_verify_flag = pc.classify_depth(depth_pct)

    # ── Final confirmation base ("handle") search: last stretch before now ──
    base_info = _detect_confirmation_base(df, peak_abs, bottom2_abs, timeframe)

    pivot_point = base_info["handle_high"] if base_info["has_handle"] else float(df["High"].iloc[peak_abs])
    current_price = float(df["Close"].iloc[-1])
    price_vs_pivot_pct = ((current_price - pivot_point) / pivot_point) * 100.0 if pivot_point else 0.0
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    signal_type = pc.classify_signal_type(current_price, pivot_point, prev_close, base_info["has_handle"])
    pattern_type = "Double Bottom" if base_info["has_handle"] else "Double Bottom (Forming)"

    atr_at_base_start = pc.atr_at_position(df, base_info.get("handle_start_pos_abs"))

    quality_score = pc.compute_quality_score_generic(
        rim_symmetry_pct=rim_symmetry_pct, depth_pct=depth_pct, prior_trend_pct_val=prior_pct,
        shape_bucket=shape_bucket, rs_rating=rs_rating,
        handle_quality_subscore=base_info["handle_quality_subscore"],
        volume_dryup_label=base_info.get("volume_dryup_label", "none"), mtf_confluence=False,
    )

    key_points = [
        {"date": str(pc.to_date(cand["bottom1_date"])), "price": bottom1_price, "label": "Bottom 1", "role": "trough"},
        {"date": str(pc.to_date(cand["peak_date"])), "price": peak_price, "label": "Neckline", "role": "peak"},
        {"date": str(pc.to_date(cand["bottom2_date"])), "price": bottom2_price, "label": "Bottom 2", "role": "trough"},
    ]
    trendlines = [{
        "role": "structure", "style": "solid", "color": "#58d6ff", "width": 2,
        "points": [[str(pc.to_date(cand["bottom1_date"])), bottom1_price],
                   [str(pc.to_date(cand["peak_date"])), peak_price],
                   [str(pc.to_date(cand["bottom2_date"])), bottom2_price]],
    }, {
        "role": "neckline", "style": "dashed", "color": "#58a6ff", "width": 1,
        "points": [[str(pc.to_date(cand["peak_date"])), peak_price],
                   [str(pc.to_date(df.index[-1])), peak_price]],
    }]
    if base_info["has_handle"]:
        key_points.append({"date": str(pc.to_date(base_info["handle_end_date"])), "price": base_info["handle_low_price"],
                            "label": "Base Low", "role": "trough"})
        trendlines.append({
            "role": "base_bottom", "style": "dashed", "color": "#d29922", "width": 1,
            "points": [[str(pc.to_date(base_info["handle_start_date"])), base_info["handle_low_price"]],
                       [str(pc.to_date(base_info["handle_end_date"])), base_info["handle_low_price"]]],
        })

    sig = pc.PatternSignal(
        symbol=symbol, timeframe=timeframe, pattern_type=pattern_type, pattern_family="double_bottom",
        cup_start_date=pc.to_date(cand["bottom1_date"]), cup_bottom_date=pc.to_date(cand["bottom1_date"]),
        cup_end_date=pc.to_date(cand["bottom2_date"]),
        left_rim_price=round(peak_price, 2), cup_bottom_price=round(min(bottom1_price, bottom2_price), 2),
        right_rim_price=round(peak_price, 2),
        cup_depth_pct=round(depth_pct, 2), cup_duration_bars=duration_bars, cup_shape=shape,
        recovery_pct=round(cand["recovery_pct"] * 100.0, 2),
        prior_uptrend_pct=round(prior_pct, 2), prior_uptrend_tag=prior_tag,
        cup_depth_class=depth_class, cup_depth_verify_flag=depth_verify_flag,
        has_handle=base_info["has_handle"],
        handle_start_date=pc.to_date(base_info.get("handle_start_date")),
        handle_end_date=pc.to_date(base_info.get("handle_end_date")),
        handle_low_price=base_info.get("handle_low_price"),
        handle_depth_pct=base_info.get("handle_depth_pct"),
        handle_duration_bars=base_info.get("handle_duration_bars"),
        handle_quality_subscore=base_info["handle_quality_subscore"],
        handle_vol_dryup_pts=base_info["handle_vol_dryup_pts"],
        handle_tightness_pts=base_info["handle_tightness_pts"],
        handle_higher_lows_pts=base_info["handle_higher_lows_pts"],
        handle_close_high_pts=base_info["handle_close_high_pts"],
        atr_at_handle_start=atr_at_base_start,
        pivot_point=round(pivot_point, 2), current_price=round(current_price, 2),
        price_vs_pivot_pct=round(price_vs_pivot_pct, 2), signal_type=signal_type,
        already_breaking_out_handle=base_info.get("already_breaking_out", False),
        quality_score=round(quality_score, 1), rim_symmetry_pct=round(rim_symmetry_pct, 2),
        key_points=key_points, trendlines=trendlines,
        structure_note=(f"{shape}: Bottom 1 ₹{bottom1_price:.2f} → Neckline ₹{peak_price:.2f} → "
                        f"Bottom 2 ₹{bottom2_price:.2f} ({diff_pct:+.1f}% vs Bottom 1)"),
        extra={"bottom1_price": bottom1_price, "bottom2_price": bottom2_price, "neckline_price": peak_price,
               "bottom_diff_pct": round(diff_pct, 2)},
    )
    sig._df_ref = df
    return sig


# ─── Final confirmation base ("handle") detection ──────────────────────────

def _detect_confirmation_base(df: pd.DataFrame, peak_abs: int, bottom2_abs: int, timeframe: str) -> dict:
    n = len(df)
    min_bars, max_bars = cfg.DB_HANDLE_SEARCH_BARS.get(timeframe, (2, 45))
    search_start = max(bottom2_abs + 1, n - max_bars)
    search_end = n

    no_base = {
        "has_handle": False, "handle_quality_subscore": 0.0, "handle_vol_dryup_pts": 0.0,
        "handle_tightness_pts": 0.0, "handle_higher_lows_pts": 0.0, "handle_close_high_pts": 0.0,
        "handle_start_pos_abs": peak_abs, "already_breaking_out": False, "volume_dryup_label": "none",
    }
    if search_start >= search_end:
        return no_base

    base_zone = df.iloc[search_start:search_end]
    if len(base_zone) < 1:
        return no_base

    peak_price = float(df["High"].iloc[peak_abs])
    bottom2_price = float(df["Low"].iloc[bottom2_abs])
    structure_height = peak_price - bottom2_price
    if structure_height <= 0:
        return no_base

    base_low = float(base_zone["Low"].min())
    base_high = float(base_zone["High"].max())

    min_allowed_low = bottom2_price + cfg.DB_HANDLE_MIN_PCT_FROM_BOTTOM * structure_height
    if base_low < min_allowed_low:
        return no_base

    base_depth_pct = ((peak_price - base_low) / peak_price) * 100.0
    cup_depth_pct_abs = (structure_height / peak_price) * 100.0
    if cup_depth_pct_abs > 0 and (base_depth_pct / cup_depth_pct_abs) > cfg.DB_HANDLE_MAX_DEPTH_RATIO:
        return no_base

    already_breaking_out = base_high > peak_price * (1 + cfg.DB_HANDLE_MAX_BREAKOUT_PCT)

    duration = len(base_zone)
    if duration < min_bars and not already_breaking_out:
        return no_base

    scored = pc.score_final_base_quality(
        df, structure_start_abs=peak_abs, structure_end_abs=bottom2_abs,
        base_start_abs=search_start, base_end_abs=search_end,
    )

    return {
        "has_handle": True,
        "handle_start_date": base_zone.index[0], "handle_end_date": base_zone.index[-1],
        "handle_low_price": round(base_low, 2), "handle_high": round(base_high, 2),
        "handle_depth_pct": round(base_depth_pct, 2), "handle_duration_bars": duration,
        "handle_quality_subscore": scored["handle_quality_subscore"],
        "handle_vol_dryup_pts": scored["handle_vol_dryup_pts"],
        "handle_tightness_pts": scored["handle_tightness_pts"],
        "handle_higher_lows_pts": scored["handle_higher_lows_pts"],
        "handle_close_high_pts": scored["handle_close_high_pts"],
        "handle_start_pos_abs": search_start,
        "already_breaking_out": already_breaking_out,
        "volume_dryup_label": scored["volume_dryup_label"],
    }
