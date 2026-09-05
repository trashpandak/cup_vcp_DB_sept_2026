"""
NSE Multi-Pattern Scanner - Ascending Triangle Detector
============================================================
A flat resistance ceiling (a cluster of highs within a tight
tolerance of each other) with a rising support line beneath it (a
sequence of higher lows). The range coils as the rising lows squeeze
up toward the flat top; breakout = a close above the flat resistance.

Detection method: percentage-reversal zigzag (pattern_common.
zigzag_pivots), then (1) cluster the trailing swing highs that sit
within `cfg.AT_RESISTANCE_TOLERANCE_PCT` of each other into the flat
top, and (2) find the trailing run of swing lows, from that cluster's
start onward, that rise monotonically (small undercut tolerance
allowed).

THE ONLY HARD GATES: >= `AT_MIN_TOUCHES_RESISTANCE` clustered highs
AND >= `AT_MIN_TOUCHES_SUPPORT` rising lows. Fit tightness, prior
trend, and volume are scored, never used to reject.

pivot_point = the flat resistance price. The final rising leg (last
support touch -> now) plays the role of "handle": its low is the stop
reference.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import config as cfg
import pattern_common as pc


def detect_ascending_triangle(df: pd.DataFrame, timeframe: str, symbol: str,
                               rs_rating: float = 50.0) -> list[pc.PatternSignal]:
    min_bars_cfg = cfg.MIN_BARS_BY_TIMEFRAME.get(timeframe)
    if min_bars_cfg is None or len(df) < min_bars_cfg:
        return []

    min_w, max_w = cfg.AT_LOOKBACKS.get(timeframe, (20, 260))
    if len(df) < min_w:
        return []

    window_size = min(max_w, len(df))
    window = df.iloc[len(df) - window_size:]
    window_start_abs = len(df) - window_size

    threshold = cfg.AT_ZIGZAG_THRESHOLD.get(timeframe, 0.04)
    pivots = pc.zigzag_pivots(window["Close"], threshold)
    if len(pivots) < 4:
        return []

    found = _find_triangle(pivots)
    if found is None:
        return []

    sig = _build_signal(df, found, window_start_abs, timeframe, symbol, rs_rating)
    return [sig] if sig is not None else []


def _find_triangle(pivots: list[dict]) -> Optional[dict]:
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]
    if len(highs) < cfg.AT_MIN_TOUCHES_RESISTANCE or len(lows) < cfg.AT_MIN_TOUCHES_SUPPORT:
        return None

    # ── Cluster the trailing highs into a flat resistance ──
    cluster = [highs[-1]]
    cluster_max = cluster_min = highs[-1]["price"]
    for h in reversed(highs[:-1]):
        new_max, new_min = max(cluster_max, h["price"]), min(cluster_min, h["price"])
        if new_max > 0 and (new_max - new_min) / new_max <= cfg.AT_RESISTANCE_TOLERANCE_PCT:
            cluster.insert(0, h)
            cluster_max, cluster_min = new_max, new_min
        else:
            break
    if len(cluster) < cfg.AT_MIN_TOUCHES_RESISTANCE:
        return None

    resistance_price = cluster_max
    fit_deviation_pct = ((cluster_max - cluster_min) / cluster_max) * 100.0

    # ── Find the trailing run of rising lows from the cluster's start onward ──
    rel_lows = [l for l in lows if l["pos"] >= cluster[0]["pos"] - 1]
    if len(rel_lows) < cfg.AT_MIN_TOUCHES_SUPPORT:
        return None

    valid_lows = [rel_lows[0]]
    for lo in rel_lows[1:]:
        prev = valid_lows[-1]
        if lo["price"] >= prev["price"] * (1 - cfg.AT_LOW_UNDERCUT_TOLERANCE):
            valid_lows.append(lo)
        else:
            valid_lows = [lo]
    if len(valid_lows) < cfg.AT_MIN_TOUCHES_SUPPORT:
        return None
    if valid_lows[-1]["price"] < valid_lows[0]["price"] * (1 + cfg.AT_MIN_LOW_RISE_PCT):
        return None

    return {"cluster": cluster, "resistance_price": resistance_price,
            "fit_deviation_pct": fit_deviation_pct, "lows": valid_lows}


def _build_signal(df: pd.DataFrame, found: dict, window_start_abs: int, timeframe: str,
                   symbol: str, rs_rating: float) -> Optional[pc.PatternSignal]:
    n = len(df)
    cluster, lows = found["cluster"], found["lows"]
    resistance_price, fit_deviation_pct = found["resistance_price"], found["fit_deviation_pct"]

    structure_start_pos = min(cluster[0]["pos"], lows[0]["pos"])
    structure_start_abs = window_start_abs + structure_start_pos
    last_low_abs = window_start_abs + lows[-1]["pos"]
    first_low_price = lows[0]["price"]

    duration_bars = (n - 1) - structure_start_abs
    if duration_bars < 1:
        return None

    depth_pct = ((resistance_price - first_low_price) / resistance_price) * 100.0 if resistance_price else 0.0

    current_price = float(df["Close"].iloc[-1])
    if current_price < resistance_price * cfg.AT_CURRENT_PRICE_MIN_PCT_OF_PIVOT:
        return None
    prev_close = float(df["Close"].iloc[-2]) if n >= 2 else current_price

    if fit_deviation_pct <= cfg.AT_TIGHT_FIT_TOLERANCE_PCT * 100.0:
        shape_bucket, fit_label, rim_symmetry_pct = "strong", "Textbook", 95.0
    else:
        shape_bucket, fit_label, rim_symmetry_pct = "moderate", "Standard", 80.0
    cup_shape = f"{fit_label} Ascending Triangle ({len(cluster)} resistance touches, {len(lows)} rising lows)"

    prior_pct = pc.prior_trend_pct(df, structure_start_abs, duration_bars)
    prior_tag = pc.prior_trend_tag(prior_pct)
    depth_class, depth_verify_flag = pc.classify_depth(depth_pct)

    base_start_abs = last_low_abs
    base_end_abs = n
    scored = pc.score_final_base_quality(
        df, structure_start_abs=structure_start_abs, structure_end_abs=last_low_abs,
        base_start_abs=base_start_abs, base_end_abs=base_end_abs,
    )

    signal_type = pc.classify_signal_type(current_price, resistance_price, prev_close, True)
    already_breaking_out = current_price > resistance_price

    quality_score = pc.compute_quality_score_generic(
        rim_symmetry_pct=rim_symmetry_pct, depth_pct=depth_pct, prior_trend_pct_val=prior_pct,
        shape_bucket=shape_bucket, rs_rating=rs_rating,
        handle_quality_subscore=scored["handle_quality_subscore"],
        volume_dryup_label=scored["volume_dryup_label"], mtf_confluence=False,
    )

    atr_at_base_start = pc.atr_at_position(df, base_start_abs)

    # ── Support trendline, linearly extended from the first to the last rising low ──
    first_low, last_low = lows[0], lows[-1]
    bar_span = max(last_low["pos"] - first_low["pos"], 1)
    slope = (last_low["price"] - first_low["price"]) / bar_span
    current_pos_in_window = n - 1 - window_start_abs
    extrapolated_support = last_low["price"] + slope * (current_pos_in_window - last_low["pos"])

    key_points = [{"date": str(pc.to_date(h["date"])), "price": h["price"],
                   "label": "Resistance Touch", "role": "peak"} for h in cluster]
    key_points += [{"date": str(pc.to_date(lo["date"])), "price": lo["price"],
                    "label": "Rising Low", "role": "trough"} for lo in lows]

    trendlines = [
        {"role": "resistance", "style": "dashed", "color": "#d29922", "width": 2,
         "points": [[str(pc.to_date(cluster[0]["date"])), resistance_price],
                    [str(pc.to_date(df.index[-1])), resistance_price]]},
        {"role": "support", "style": "dashed", "color": "#3fb950", "width": 2,
         "points": [[str(pc.to_date(first_low["date"])), first_low["price"]],
                    [str(pc.to_date(df.index[-1])), round(extrapolated_support, 2)]]},
    ]

    sig = pc.PatternSignal(
        symbol=symbol, timeframe=timeframe, pattern_type="Ascending Triangle", pattern_family="ascending_triangle",
        cup_start_date=pc.to_date(df.index[window_start_abs + structure_start_pos]),
        cup_bottom_date=pc.to_date(first_low["date"]), cup_end_date=pc.to_date(last_low["date"]),
        left_rim_price=round(resistance_price, 2), cup_bottom_price=round(first_low_price, 2),
        right_rim_price=round(cluster[-1]["price"], 2),
        cup_depth_pct=round(depth_pct, 2), cup_duration_bars=duration_bars, cup_shape=cup_shape,
        recovery_pct=round(((current_price - first_low_price) / max(resistance_price - first_low_price, 1e-9)) * 100.0, 2),
        prior_uptrend_pct=round(prior_pct, 2), prior_uptrend_tag=prior_tag,
        cup_depth_class=depth_class, cup_depth_verify_flag=depth_verify_flag,
        has_handle=True,
        handle_start_date=pc.to_date(last_low["date"]), handle_end_date=pc.to_date(df.index[-1]),
        handle_low_price=round(last_low["price"], 2),
        handle_depth_pct=round(((resistance_price - last_low["price"]) / resistance_price) * 100.0, 2),
        handle_duration_bars=base_end_abs - base_start_abs,
        handle_quality_subscore=scored["handle_quality_subscore"],
        handle_vol_dryup_pts=scored["handle_vol_dryup_pts"], handle_tightness_pts=scored["handle_tightness_pts"],
        handle_higher_lows_pts=scored["handle_higher_lows_pts"], handle_close_high_pts=scored["handle_close_high_pts"],
        atr_at_handle_start=atr_at_base_start,
        pivot_point=round(resistance_price, 2), current_price=round(current_price, 2),
        price_vs_pivot_pct=round(((current_price - resistance_price) / resistance_price) * 100.0, 2),
        signal_type=signal_type, already_breaking_out_handle=already_breaking_out,
        quality_score=round(quality_score, 1), rim_symmetry_pct=round(rim_symmetry_pct, 2),
        key_points=key_points, trendlines=trendlines,
        structure_note=(f"{fit_label} Ascending Triangle: flat resistance ₹{resistance_price:.2f} "
                        f"({len(cluster)} touches), rising support from ₹{first_low_price:.2f} "
                        f"to ₹{last_low['price']:.2f} ({len(lows)} touches)"),
        extra={"resistance_touches": len(cluster), "support_touches": len(lows),
               "fit_deviation_pct": round(fit_deviation_pct, 2)},
    )
    sig._df_ref = df
    return sig
