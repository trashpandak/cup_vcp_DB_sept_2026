"""
NSE Multi-Pattern Scanner - VCP (Volatility Contraction Pattern) Detector
=============================================================================
Minervini-style VCP: a series of pullbacks off a high, each one
SHALLOWER than the last and with a HIGHER low than the last, volume
drying up as the range tightens, culminating in a breakout above the
base's high on expanding volume.

Detection method: run a percentage-reversal zigzag (pattern_common.
zigzag_pivots) over the lookback window, then walk backward from the
most recent swing low, extending the "contraction run" for as long as
each earlier leg is a deeper pullback with a lower low than the one
after it (both with a little slack, since real data is noisy).

THE ONLY HARD GATE: at least `cfg.VCP_MIN_CONTRACTIONS` (2) qualifying
contractions. Everything else — how clean the tightening is, how far
volume dried up, prior trend — is scored, never used to reject.

pivot_point = the highest high across the qualifying contractions.
The final (most recent, tightest) contraction plays the role of
"handle" (see pattern_common.py's mapping): its low is the stop
reference, and its quality feeds the same handle_quality_subscore
bucket cup & handle uses.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import config as cfg
import pattern_common as pc


def detect_vcp(df: pd.DataFrame, timeframe: str, symbol: str,
                rs_rating: float = 50.0) -> list[pc.PatternSignal]:
    min_bars_cfg = cfg.MIN_BARS_BY_TIMEFRAME.get(timeframe)
    if min_bars_cfg is None or len(df) < min_bars_cfg:
        return []

    min_w, max_w = cfg.VCP_LOOKBACKS.get(timeframe, (25, 300))
    if len(df) < min_w:
        return []

    window_size = min(max_w, len(df))
    window = df.iloc[len(df) - window_size:]
    window_start_abs = len(df) - window_size

    threshold = cfg.VCP_ZIGZAG_THRESHOLD.get(timeframe, 0.05)
    pivots = pc.zigzag_pivots(window["Close"], threshold)
    if len(pivots) < 3:
        return []

    run = _extract_contraction_run(pivots)
    if len(run) < cfg.VCP_MIN_CONTRACTIONS:
        return []
    run = run[-cfg.VCP_MAX_CONTRACTIONS_CONSIDERED:]

    sig = _build_signal(df, run, window_start_abs, timeframe, symbol, rs_rating)
    return [sig] if sig is not None else []


def _extract_contraction_run(pivots: list[dict]) -> list[tuple[dict, dict]]:
    downlegs = [(pivots[k], pivots[k + 1]) for k in range(len(pivots) - 1)
                if pivots[k]["kind"] == "high" and pivots[k + 1]["kind"] == "low"]
    if not downlegs:
        return []

    run = [downlegs[-1]]
    for i in range(len(downlegs) - 2, -1, -1):
        prev_leg, next_leg = downlegs[i], run[0]
        prev_depth = (prev_leg[0]["price"] - prev_leg[1]["price"]) / max(prev_leg[0]["price"], 1e-9)
        next_depth = (next_leg[0]["price"] - next_leg[1]["price"]) / max(next_leg[0]["price"], 1e-9)
        lows_ok = next_leg[1]["price"] >= prev_leg[1]["price"] * (1 - cfg.VCP_LOW_UNDERCUT_TOLERANCE)
        depth_ok = next_depth <= prev_depth * cfg.VCP_CONTRACTION_SLACK
        if lows_ok and depth_ok:
            run.insert(0, prev_leg)
        else:
            break
    return run


def _build_signal(df: pd.DataFrame, run: list[tuple[dict, dict]], window_start_abs: int,
                   timeframe: str, symbol: str, rs_rating: float) -> Optional[pc.PatternSignal]:
    n = len(df)
    # Positions returned by zigzag are relative to the window — convert to absolute
    for leg in run:
        for pivot in leg:
            pivot["pos_abs"] = window_start_abs + pivot["pos"]

    depths = [(leg[0]["price"] - leg[1]["price"]) / leg[0]["price"] * 100.0 for leg in run]
    num_contractions = len(run)

    pivot_point = max(leg[0]["price"] for leg in run)
    deepest_low = min(leg[1]["price"] for leg in run)
    structure_start_abs = run[0][0]["pos_abs"]

    last_high, last_low = run[-1]
    base_start_abs = last_high["pos_abs"]
    base_end_abs = n if last_low.get("provisional") else last_low["pos_abs"] + 1
    structure_end_abs = base_end_abs - 1

    duration_bars = (n - 1) - structure_start_abs
    if duration_bars < 1:
        return None

    depth_pct = (pivot_point - deepest_low) / pivot_point * 100.0 if pivot_point else 0.0

    current_price = float(df["Close"].iloc[-1])
    if current_price < pivot_point * cfg.VCP_CURRENT_PRICE_MIN_PCT_OF_PIVOT:
        return None
    prev_close = float(df["Close"].iloc[-2]) if n >= 2 else current_price

    # ── Shape bucket ──
    if num_contractions >= cfg.VCP_TEXTBOOK_MIN_CONTRACTIONS:
        shape_bucket = "strong"
    else:
        shape_bucket = "moderate"
    depths_str = " → ".join(f"{d:.0f}%" for d in depths)
    cup_shape = f"{num_contractions} Contractions ({depths_str})"

    rim_symmetry_pct = 100.0 if num_contractions >= cfg.VCP_TEXTBOOK_MIN_CONTRACTIONS else 80.0

    prior_pct = pc.prior_trend_pct(df, structure_start_abs, duration_bars)
    prior_tag = pc.prior_trend_tag(prior_pct)
    depth_class, depth_verify_flag = pc.classify_depth(depth_pct)

    scored = pc.score_final_base_quality(
        df, structure_start_abs=structure_start_abs, structure_end_abs=structure_end_abs,
        base_start_abs=base_start_abs, base_end_abs=base_end_abs,
    )

    handle_low_price = round(last_low["price"], 2)
    handle_high = float(df["High"].iloc[base_start_abs:base_end_abs].max())
    already_breaking_out = handle_high > pivot_point and current_price >= pivot_point

    signal_type = pc.classify_signal_type(current_price, pivot_point, prev_close, True)

    quality_score = pc.compute_quality_score_generic(
        rim_symmetry_pct=rim_symmetry_pct, depth_pct=depth_pct, prior_trend_pct_val=prior_pct,
        shape_bucket=shape_bucket, rs_rating=rs_rating,
        handle_quality_subscore=scored["handle_quality_subscore"],
        volume_dryup_label=scored["volume_dryup_label"], mtf_confluence=False,
    )

    atr_at_base_start = pc.atr_at_position(df, base_start_abs)

    handle_start_date = last_high["date"]
    handle_end_date = df.index[-1] if last_low.get("provisional") else last_low["date"]

    key_points = []
    for i, leg in enumerate(run, start=1):
        key_points.append({"date": str(pc.to_date(leg[0]["date"])), "price": leg[0]["price"],
                            "label": f"Contraction {i} High", "role": "peak"})
        key_points.append({"date": str(pc.to_date(leg[1]["date"])), "price": leg[1]["price"],
                            "label": f"Contraction {i} Low", "role": "trough"})

    envelope_high = [[str(pc.to_date(leg[0]["date"])), leg[0]["price"]] for leg in run]
    envelope_low = [[str(pc.to_date(leg[1]["date"])), leg[1]["price"]] for leg in run]
    trendlines = [
        {"role": "envelope_high", "style": "solid", "color": "#f778ba", "width": 2, "points": envelope_high},
        {"role": "envelope_low", "style": "solid", "color": "#f778ba", "width": 2, "points": envelope_low},
        {"role": "pivot_line", "style": "dashed", "color": "#3fb950", "width": 1,
         "points": [[str(pc.to_date(run[0][0]["date"])), pivot_point], [str(pc.to_date(df.index[-1])), pivot_point]]},
    ]

    cup_start_date = pc.to_date(run[0][0]["date"])
    cup_bottom_date = pc.to_date(min(run, key=lambda leg: leg[1]["price"])[1]["date"])
    cup_end_date = pc.to_date(handle_end_date)

    sig = pc.PatternSignal(
        symbol=symbol, timeframe=timeframe,
        pattern_type=f"VCP ({num_contractions} Contractions)", pattern_family="vcp",
        cup_start_date=cup_start_date, cup_bottom_date=cup_bottom_date, cup_end_date=cup_end_date,
        left_rim_price=round(pivot_point, 2), cup_bottom_price=round(deepest_low, 2),
        right_rim_price=round(last_high["price"], 2),
        cup_depth_pct=round(depth_pct, 2), cup_duration_bars=duration_bars, cup_shape=cup_shape,
        recovery_pct=round(((current_price - deepest_low) / max(pivot_point - deepest_low, 1e-9)) * 100.0, 2),
        prior_uptrend_pct=round(prior_pct, 2), prior_uptrend_tag=prior_tag,
        cup_depth_class=depth_class, cup_depth_verify_flag=depth_verify_flag,
        has_handle=True,
        handle_start_date=pc.to_date(handle_start_date), handle_end_date=pc.to_date(handle_end_date),
        handle_low_price=handle_low_price, handle_depth_pct=round(depths[-1], 2),
        handle_duration_bars=base_end_abs - base_start_abs,
        handle_quality_subscore=scored["handle_quality_subscore"],
        handle_vol_dryup_pts=scored["handle_vol_dryup_pts"], handle_tightness_pts=scored["handle_tightness_pts"],
        handle_higher_lows_pts=scored["handle_higher_lows_pts"], handle_close_high_pts=scored["handle_close_high_pts"],
        atr_at_handle_start=atr_at_base_start,
        pivot_point=round(pivot_point, 2), current_price=round(current_price, 2),
        price_vs_pivot_pct=round(((current_price - pivot_point) / pivot_point) * 100.0, 2),
        signal_type=signal_type, already_breaking_out_handle=already_breaking_out,
        quality_score=round(quality_score, 1), rim_symmetry_pct=round(rim_symmetry_pct, 2),
        key_points=key_points, trendlines=trendlines,
        structure_note=f"{num_contractions} contractions tightening {depths_str}, "
                        f"pivot ₹{pivot_point:.2f}, last low ₹{handle_low_price:.2f}",
        extra={"num_contractions": num_contractions, "contraction_depths_pct": [round(d, 2) for d in depths]},
    )
    sig._df_ref = df
    return sig
