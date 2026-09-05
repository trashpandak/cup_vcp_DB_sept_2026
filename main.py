"""
NSE Cup & Handle Scanner - Main Orchestrator
================================================
Implements spec v3 Part 9. Wires together:
  universe -> downloader -> indicators (RS Rating) -> detector ->
  readiness -> entry_exit -> database -> report

Run modes:
  python main.py                       # incremental daily scan
  python main.py --full-refresh        # wipe and redownload all data
  python main.py --refresh-universe    # force-refresh NSE symbol list
  python main.py --debug-symbol TCS.NS # verbose single-symbol diagnosis
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

import config as cfg
import database as db
from cup_handle_detector import (
    CupHandleSignal, apply_mtf_confluence_bonus, detect_cup_handle,
)
import pattern_common as pc
from double_bottom_detector import detect_double_bottom
from vcp_detector import detect_vcp
from ascending_triangle_detector import detect_ascending_triangle
from candlestick_patterns import detect_confirmation_candle
from downloader import load_daily, resample_monthly, resample_weekly, run_download
from entry_exit import calculate_entry_exit
from indicators import (
    adx, atr, clean_volume, cross_sectional_rs_rating, raw_weighted_return,
    rs_rating_trend, rsi, sma,
)
from logger_utils import get_logger
from readiness import compute_breakout_readiness, READY_SIGNAL_TYPES
from report import generate_excel_report
from chart_export import export_html_dashboard
from universe import fetch_nse_symbols, get_symbol_meta

log = get_logger("scanner")

TIMEFRAMES = ["daily", "weekly", "monthly"]

# All pattern detectors run for every symbol/timeframe. Each returns a
# list of signals exposing the same attribute surface (see
# pattern_common.py), so everything downstream — readiness, entry/exit,
# persistence, reporting — treats them uniformly.
PATTERN_DETECTORS = [
    detect_cup_handle,
    detect_double_bottom,
    detect_vcp,
    detect_ascending_triangle,
]


def main() -> None:
    args = _parse_args()
    started_at = datetime.now()
    log.info("=" * 70)
    log.info("NSE CUP & HANDLE SCANNER — run started %s", started_at.isoformat())
    log.info("=" * 70)

    db.init_db()

    # One-time migration: collapse duplicate rows left over from before
    # signal_id was anchored on cup_bottom_date (older signal_ids drifted
    # day to day, causing the same real pattern to accumulate as many
    # rows in the watchlist/active tracking). Safe to run every time —
    # becomes a no-op once the DB is clean.
    deduped = db.dedupe_legacy_duplicate_signals()
    if deduped:
        log.info("Migration: collapsed %d legacy duplicate signal rows", deduped)

    # One-time self-healing: reset any signals that were incorrectly
    # auto-triggered on a prior run (non-BREAKOUT-NOW states that got
    # marked Triggered when they shouldn't have been).
    cleaned = db.cleanup_bad_triggers(date.today().isoformat())
    if cleaned:
        log.info("Cleaned up %d incorrectly triggered signals → reset to Watching", cleaned)

    if args.debug_symbol:
        _run_debug_single_symbol(args.debug_symbol)
        return

    symbols = fetch_nse_symbols(force_refresh=args.refresh_universe)
    log.info("Universe: %d symbols", len(symbols))

    run_download(symbols, full_refresh=args.full_refresh)

    nifty_trend, nifty_df = _load_benchmark()

    rs_ratings, rs_trends = _compute_rs_ratings(symbols)
    log.info("RS Ratings computed for %d symbols", len(rs_ratings))

    scan_date = date.today().isoformat()
    error_count = 0
    pattern_count = 0
    by_timeframe = {tf: 0 for tf in TIMEFRAMES}
    by_signal_type: dict[str, int] = {}
    by_rs_band = {"<60": 0, "60-80": 0, ">80": 0}

    for i, symbol in enumerate(symbols, 1):
        if i % 200 == 0:
            log.info("Progress: %d/%d symbols scanned", i, len(symbols))
        try:
            n_found = _scan_symbol(
                symbol, rs_ratings, rs_trends, nifty_trend, scan_date,
                by_timeframe, by_signal_type, by_rs_band,
            )
            pattern_count += n_found
        except Exception:
            error_count += 1
            log.debug("Error scanning %s:\n%s", symbol, traceback.format_exc())

    error_rate = error_count / len(symbols) if symbols else 0
    if error_rate > 0.10:
        log.warning(
            "High error rate: %d/%d symbols (%.1f%%) failed to scan",
            error_count, len(symbols), error_rate * 100,
        )
    else:
        log.info("Scan complete: %d errors out of %d symbols", error_count, len(symbols))

    db.prune_expired_watchlist()
    _update_active_tracking()

    summary_stats = _build_summary_stats(
        scan_date, len(symbols), pattern_count, by_timeframe,
        by_signal_type, by_rs_band, nifty_trend,
    )

    _generate_report(scan_date, summary_stats)

    try:
        export_html_dashboard(scan_date, nifty_trend)
    except Exception:
        log.warning("HTML dashboard export failed (non-fatal):\n%s", traceback.format_exc())

    elapsed = (datetime.now() - started_at).total_seconds()
    log.info("=" * 70)
    log.info("RUN COMPLETE in %.1f minutes — %d patterns detected across %d symbols",
              elapsed / 60, pattern_count, len(symbols))
    log.info("=" * 70)


# ─── Per-symbol scan ────────────────────────────────────────────────────────

def _scan_symbol(
    symbol: str,
    rs_ratings: dict,
    rs_trends: dict,
    nifty_trend: str,
    scan_date: str,
    by_timeframe: dict,
    by_signal_type: dict,
    by_rs_band: dict,
) -> int:
    daily = load_daily(symbol)
    if daily is None or len(daily) < cfg.MIN_DAILY_BARS:
        return 0

    rs_val = rs_ratings.get(symbol, 50.0)
    rs_trend_label = rs_trends.get(symbol, "Unknown")
    meta = get_symbol_meta(symbol)

    timeframe_data = {"daily": daily}
    if len(daily) >= cfg.MIN_WEEKLY_BARS:
        weekly = resample_weekly(daily)
        if len(weekly) >= cfg.MIN_WEEKLY_BARS:
            timeframe_data["weekly"] = weekly
    if len(daily) >= cfg.MIN_MONTHLY_BARS:
        monthly = resample_monthly(daily)
        if len(monthly) >= cfg.MIN_MONTHLY_BARS:
            timeframe_data["monthly"] = monthly

    all_signals: list[tuple[CupHandleSignal, pd.DataFrame]] = []

    for tf, tf_df in timeframe_data.items():
        for detector in PATTERN_DETECTORS:
            try:
                sigs = detector(tf_df, tf, symbol, rs_rating=rs_val)
            except Exception:
                log.debug("Detection error %s/%s/%s:\n%s", symbol, tf, detector.__name__,
                          traceback.format_exc())
                continue
            for s in sigs:
                pc.ensure_generic_fields(s)
                if _is_stale(s, tf_df, tf):
                    continue
                all_signals.append((s, tf_df))

    if not all_signals:
        return 0

    # ── Deduplicate within-timeframe ──────────────────────────────────────
    # Two different window sizes (or two different detectors briefly
    # agreeing on the same pivot) can describe the same underlying move
    # → different signal_ids → both persist → duplicates in the report.
    # Keyed on pattern_family too, so a Double Bottom and a VCP that
    # happen to share a pivot on the same timeframe are kept as two
    # distinct signals, not collapsed into one.
    deduped: dict[tuple, tuple[CupHandleSignal, pd.DataFrame]] = {}
    for sig, tf_df in all_signals:
        key = (sig.timeframe, sig.pattern_family, round(sig.pivot_point, 0))
        existing = deduped.get(key)
        if existing is None or sig.quality_score > existing[0].quality_score:
            deduped[key] = (sig, tf_df)
    all_signals = list(deduped.values())

    # ── Multi-timeframe confluence, scoped per pattern family ──────────────
    # (2+ timeframes producing the SAME pattern type for this symbol)
    by_family: dict[str, list] = {}
    for s, tf_df in all_signals:
        by_family.setdefault(s.pattern_family, []).append((s, tf_df))
    for family_sigs in by_family.values():
        timeframes_hit = sorted({s.timeframe for s, _ in family_sigs}, key=TIMEFRAMES.index)
        if len(timeframes_hit) >= 2:
            for s, _ in family_sigs:
                s.mtf_confluence = True
                s.mtf_timeframes = ", ".join(t.capitalize() for t in timeframes_hit)
                s.quality_score = apply_mtf_confluence_bonus(s.quality_score)

    # ── Cross-pattern confluence (informational) ────────────────────────────
    # This symbol shows a bullish structure — any structure — on 2+
    # different timeframes, even if the specific pattern type differs
    # between them (e.g. a Cup & Handle forming on the daily chart while
    # a VCP is also visible weekly). Doesn't touch quality_score (that's
    # reserved for same-pattern confluence above); surfaces as a badge.
    all_timeframes_hit = sorted({s.timeframe for s, _ in all_signals}, key=TIMEFRAMES.index)
    if len(all_timeframes_hit) >= 2:
        seen, desc_parts = set(), []
        for s, _ in sorted(all_signals, key=lambda x: TIMEFRAMES.index(x[0].timeframe)):
            key = (s.pattern_type, s.timeframe)
            if key in seen:
                continue
            seen.add(key)
            desc_parts.append(f"{s.pattern_type} ({s.timeframe.capitalize()})")
        if len(desc_parts) >= 2:
            cross_desc = " + ".join(desc_parts)
            for s, _ in all_signals:
                s.extra["cross_pattern_confluence"] = True
                s.extra["cross_pattern_desc"] = cross_desc

    n_persisted = 0
    for sig, tf_df in all_signals:
        try:
            _finalise_and_persist_signal(
                sig, tf_df, symbol, meta, rs_val, rs_trend_label,
                nifty_trend, scan_date,
            )
            by_timeframe[sig.timeframe] = by_timeframe.get(sig.timeframe, 0) + 1
            by_signal_type[sig.signal_type] = by_signal_type.get(sig.signal_type, 0) + 1
            if rs_val >= cfg.RS_LEADER_THRESHOLD:
                by_rs_band[">80"] += 1
            elif rs_val >= cfg.RS_RISING_THRESHOLD:
                by_rs_band["60-80"] += 1
            else:
                by_rs_band["<60"] += 1
            n_persisted += 1
        except Exception:
            log.debug("Persist error %s/%s:\n%s", symbol, sig.timeframe, traceback.format_exc())

    return n_persisted


def _is_stale(sig: CupHandleSignal, df: pd.DataFrame, timeframe: str) -> bool:
    max_age = cfg.STALE_PATTERN_MAX_BARS.get(timeframe, 60)
    try:
        cup_end_pos = df.index.get_indexer([pd.Timestamp(sig.cup_end_date)], method="nearest")[0]
        bars_since = len(df) - 1 - cup_end_pos
        return bars_since > max_age
    except Exception:
        return False


def _finalise_and_persist_signal(
    sig: CupHandleSignal,
    df: pd.DataFrame,
    symbol: str,
    meta: dict,
    rs_val: float,
    rs_trend_label: str,
    nifty_trend: str,
    scan_date: str,
) -> None:
    readiness = compute_breakout_readiness(sig, df, rs_trend_label)
    plan = calculate_entry_exit(sig, df, nifty_trend)

    # Bullish confirmation candle at the breakout — only meaningful once
    # the signal is close enough to matter (mirrors readiness.py's own
    # gate), so a pattern still deep in EARLY STAGE doesn't get a
    # confusing "no bullish confirmation" flag for a breakout that isn't
    # actually happening yet.
    if sig.signal_type in READY_SIGNAL_TYPES:
        sig.confirmation_candle = detect_confirmation_candle(df, breakout_pos=len(df) - 1)
    else:
        sig.confirmation_candle = None

    rsi_val = _safe_last(rsi(df["Close"], period=cfg.RSI_PERIOD))
    adx_val = _safe_last(adx(df, period=cfg.ADX_PERIOD))

    ma_periods = {"daily": (50, 150, 200), "weekly": (10, 30, 40), "monthly": (3, 7, 9)}.get(
        sig.timeframe, (50, 150, 200)
    )
    ma_short = _safe_last(sma(df["Close"], ma_periods[0]))
    ma_mid = _safe_last(sma(df["Close"], ma_periods[1]))
    ma_long = _safe_last(sma(df["Close"], ma_periods[2]))
    price_vs_ma_short_pct = (
        ((sig.current_price - ma_short) / ma_short * 100.0) if ma_short else None
    )

    rs_tag = (
        "RS Leader" if rs_val >= cfg.RS_LEADER_THRESHOLD
        else "RS Rising" if rs_val >= cfg.RS_RISING_THRESHOLD
        else "RS Laggard" if rs_val < cfg.RS_LAGGARD_THRESHOLD
        else "-"
    )

    remarks = _build_remarks(sig, plan, rs_tag, readiness)

    signal_id = db.make_signal_id(
        symbol, sig.timeframe, sig.cup_bottom_date
    )

    cc = sig.confirmation_candle or {}

    row = {
        "signal_id": signal_id,
        "symbol": symbol,
        "company_name": meta.get("name", ""),
        "sector": meta.get("sector", ""),
        "scan_date": scan_date,
        "timeframe": sig.timeframe,
        "pattern_type": sig.pattern_type,
        "pattern_family": sig.pattern_family,
        "signal_type": sig.signal_type,
        "quality_score": sig.quality_score,
        "mtf_confluence": int(sig.mtf_confluence),
        "mtf_timeframes": sig.mtf_timeframes,
        "cross_pattern_confluence": int(bool(sig.extra.get("cross_pattern_confluence"))),
        "cross_pattern_desc": sig.extra.get("cross_pattern_desc", ""),

        "cup_start_date": str(sig.cup_start_date),
        "cup_bottom_date": str(sig.cup_bottom_date),
        "cup_end_date": str(sig.cup_end_date),
        "left_rim_price": sig.left_rim_price,
        "cup_bottom_price": sig.cup_bottom_price,
        "right_rim_price": sig.right_rim_price,
        "cup_depth_pct": sig.cup_depth_pct,
        "cup_depth_class": sig.cup_depth_class,
        "cup_depth_verify_flag": int(sig.cup_depth_verify_flag),
        "cup_duration_bars": sig.cup_duration_bars,
        "cup_shape": sig.cup_shape,
        "recovery_pct": sig.recovery_pct,
        "prior_uptrend_pct": sig.prior_uptrend_pct,
        "prior_uptrend_tag": sig.prior_uptrend_tag,

        "has_handle": int(sig.has_handle),
        "handle_start_date": str(sig.handle_start_date) if sig.handle_start_date else None,
        "handle_end_date": str(sig.handle_end_date) if sig.handle_end_date else None,
        "handle_low_price": sig.handle_low_price,
        "handle_depth_pct": sig.handle_depth_pct,
        "handle_duration_bars": sig.handle_duration_bars,
        "handle_quality_subscore": sig.handle_quality_subscore,
        "handle_vol_dryup_pts": sig.handle_vol_dryup_pts,
        "handle_tightness_pts": sig.handle_tightness_pts,
        "handle_higher_lows_pts": sig.handle_higher_lows_pts,
        "handle_close_high_pts": sig.handle_close_high_pts,
        "atr_at_handle_start": sig.atr_at_handle_start,

        "pivot_point": sig.pivot_point,
        "current_price": sig.current_price,
        "price_vs_pivot_pct": sig.price_vs_pivot_pct,

        "breakout_readiness_pct": readiness["readiness_pct"],
        "readiness_near_pivot": int(readiness["near_pivot"]),
        "readiness_tight_handle": int(readiness["tight_handle"]),
        "readiness_rising_rs": int(readiness["rising_rs"]),
        "readiness_atr_contract": int(readiness["atr_contracting"]),
        "readiness_above_50ma": int(readiness["above_50ma"]),
        "readiness_reasons": readiness["reasons_str"],

        "entry_price": plan.entry_price,
        "entry_zone_high": plan.entry_zone_high,
        "entry_type": plan.entry_type,
        "extended_warning": plan.extended_warning,

        "stop_loss_price": plan.stop_loss_price,
        "stop_loss_pct": plan.stop_loss_pct,
        "stop_loss_type": plan.stop_loss_type,
        "atr_14": plan.atr_14,
        "risk_per_share": plan.risk_per_share,

        "target1": plan.target1,
        "target2": plan.target2,
        "target3": plan.target3,
        "rr_t1": plan.rr_t1,
        "rr_t2": plan.rr_t2,
        "rr_t3": plan.rr_t3,
        "rr_t2_warning": plan.rr_t2_warning,
        "eight_week_hold_candidate": int(plan.eight_week_hold_candidate),

        "position_size_shares": plan.position_size_shares,
        "capital_required": plan.capital_required,
        "risk_amount": plan.risk_amount,
        "portfolio_risk_pct": plan.portfolio_risk_pct,

        "volume_ratio": plan.volume_ratio,
        "volume_confirmed": int(plan.volume_confirmed),
        "volume_confirmed_label": plan.volume_confirmed_label,

        "rs_rating": rs_val,
        "rs_trend": rs_trend_label,
        "rs_tag": rs_tag,
        "rsi_val": rsi_val,
        "adx_val": adx_val,
        "ma_short": ma_short,
        "ma_mid": ma_mid,
        "ma_long": ma_long,
        "price_vs_ma_short_pct": price_vs_ma_short_pct,

        "nifty_trend": nifty_trend,
        "market_note": plan.market_note,
        "weak_momentum_note": plan.weak_momentum_note,
        "below_50ma_note": plan.below_50ma_note,

        "liquidity_ok": int(plan.liquidity_ok),
        "liquidity_warning": plan.liquidity_warning,

        "sell_notes": plan.sell_notes,
        "remarks": remarks,

        "key_points_json": json.dumps(sig.key_points) if sig.key_points else None,
        "trendlines_json": json.dumps(sig.trendlines) if sig.trendlines else None,
        "structure_note": sig.structure_note,
        "confirmation_candle_found": int(bool(cc.get("found"))),
        "confirmation_candle_pattern": cc.get("pattern"),
        "confirmation_candle_strength": cc.get("strength"),
        "confirmation_candle_date": cc.get("date"),
        "confirmation_candle_detail": cc.get("detail"),

        "status": "Watching",
        "entry_triggered": 0,
        "expiry_date": (date.today() + timedelta(days=cfg.WATCHLIST_EXPIRY_DAYS)).isoformat(),
    }

    db.upsert_cup_handle_signal(row)


def _build_remarks(sig: CupHandleSignal, plan, rs_tag: str, readiness: dict) -> str:
    parts = []
    if rs_tag == "RS Leader":
        parts.append("RS Leader")
    if sig.mtf_confluence:
        parts.append(f"MTF Bullish ({sig.mtf_timeframes})")
    if sig.extra.get("cross_pattern_confluence"):
        parts.append(f"Multi-Pattern Confluence: {sig.extra.get('cross_pattern_desc', '')}")
    if plan.volume_ratio and plan.volume_ratio >= 2.0:
        parts.append("Vol Surge")
    if plan.stop_loss_type == "8pct_cap":
        parts.append("Stop Capped")
    if plan.extended_warning:
        parts.append(plan.extended_warning)
    if sig.cup_depth_verify_flag:
        parts.append("Verify manually (very shallow cup)")
    if sig.already_breaking_out_handle:
        parts.append("Already Breaking Out")
    if not plan.liquidity_ok:
        parts.append("LOW LIQUIDITY")
    if readiness.get("readiness_pct") is not None and readiness["readiness_pct"] >= cfg.READINESS_BAND_HIGH:
        parts.append("High Readiness")
    cc = sig.confirmation_candle
    if cc and cc.get("found") and cc.get("strength") in ("Strong", "Moderate"):
        parts.append(f"Bullish Confirmation: {cc.get('pattern')}")
    elif cc and cc.get("found") is False and sig.signal_type == "BREAKOUT NOW":
        parts.append("No Confirmation Candle Yet")
    return ", ".join(parts)


def _safe_last(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else None


# ─── Benchmark / market condition ──────────────────────────────────────────

def _load_benchmark() -> tuple[str, pd.DataFrame | None]:
    nifty = load_daily(cfg.NIFTY50_SYMBOL)
    if nifty is None or len(nifty) < 50:
        log.warning("Nifty 50 data unavailable — defaulting market condition to Unknown")
        return "Unknown", None

    ma50 = sma(nifty["Close"], 50).iloc[-1]
    current = nifty["Close"].iloc[-1]
    if pd.isna(ma50):
        return "Unknown", nifty

    trend = "Uptrend" if current > ma50 else "Correction"
    log.info("Nifty 50: current=%.2f  50MA=%.2f  trend=%s", current, ma50, trend)
    return trend, nifty


# ─── RS Rating computation ──────────────────────────────────────────────────

def _compute_rs_ratings(symbols: list[str]) -> tuple[dict, dict]:
    raw_returns = {}
    raw_returns_4w_ago = {}

    for sym in symbols:
        daily = load_daily(sym)
        if daily is None or len(daily) < cfg.MIN_DAILY_BARS:
            continue
        close = daily["Close"]
        raw_returns[sym] = raw_weighted_return(close)

        # 4 weeks ago snapshot (~20 trading days), for RS trend
        lookback_bars = cfg.RS_TREND_LOOKBACK_WEEKS * 5
        if len(close) > lookback_bars:
            close_4w_ago = close.iloc[:-lookback_bars]
            raw_returns_4w_ago[sym] = raw_weighted_return(close_4w_ago)

    rs_series = cross_sectional_rs_rating(pd.Series(raw_returns))
    rs_series_4w_ago = cross_sectional_rs_rating(pd.Series(raw_returns_4w_ago))

    rs_ratings = rs_series.to_dict()
    rs_trends = {}
    for sym, rs_now in rs_ratings.items():
        rs_then = rs_series_4w_ago.get(sym)
        rs_trends[sym] = rs_rating_trend(rs_now, rs_then)

    return rs_ratings, rs_trends


# ─── Active tracking updates ────────────────────────────────────────────────

def _update_active_tracking() -> None:
    """
    For signals already marked entry_triggered=1, check the latest
    price against stop/targets and update status. For signals still
    "Watching", check if today's breakout makes them newly triggered.
    """
    open_signals = db.get_open_signals()
    today_str = date.today().isoformat()

    for row in open_signals:
        symbol = row["symbol"]
        timeframe = row["timeframe"]
        daily = load_daily(symbol)
        if daily is None or daily.empty:
            continue

        tf_df = daily
        if timeframe == "weekly":
            tf_df = resample_weekly(daily)
        elif timeframe == "monthly":
            tf_df = resample_monthly(daily)
        if tf_df.empty:
            continue

        current_price = float(tf_df["Close"].iloc[-1])
        updates = {"current_price": current_price}

        stop = row.get("stop_loss_price")
        t1, t2, t3 = row.get("target1"), row.get("target2"), row.get("target3")

        if stop is not None and current_price <= stop and not row.get("stopped_out"):
            updates.update({
                "stopped_out": 1, "status": "Stopped Out",
                "exit_date": today_str, "exit_price": current_price,
                "exit_type": "Stop",
            })
        else:
            if t3 is not None and current_price >= t3 and not row.get("t3_achieved"):
                updates.update({"t3_achieved": 1, "status": "Target 3 Achieved"})
            elif t2 is not None and current_price >= t2 and not row.get("t2_achieved"):
                updates.update({"t2_achieved": 1, "status": "Target 2 Achieved"})
            elif t1 is not None and current_price >= t1 and not row.get("t1_achieved"):
                updates.update({"t1_achieved": 1, "status": "Target 1 Achieved"})

        db.update_signal_status(row["signal_id"], **updates)

    # ── Auto-trigger BREAKOUT NOW signals from today's scan ──────────────
    # Only signals classified as "BREAKOUT NOW" on this scan date are
    # marked entry_triggered immediately — they broke out TODAY per the
    # scanner's detection. Signals detected in earlier states (BASING,
    # NEAR BREAKOUT, etc.) that have price >= pivot are NOT auto-triggered
    # here; they need a fresh daily run to confirm breakout with volume.
    watching = db.get_watching_signals()
    for row in watching:
        symbol = row["symbol"]
        timeframe = row["timeframe"]
        signal_type = row.get("signal_type", "")
        scan_date_db = row.get("scan_date", "")

        # Only auto-trigger signals from today's scan that are already
        # classified BREAKOUT NOW — they satisfied volume+price conditions
        # at detection time. Do NOT auto-trigger historical signals whose
        # pivot was crossed before we started tracking them.
        if signal_type != "BREAKOUT NOW" or scan_date_db != today_str:
            continue

        pivot = row.get("pivot_point")
        entry_price = row.get("entry_price")
        if pivot is None or entry_price is None:
            continue

        daily = load_daily(symbol)
        if daily is None or daily.empty:
            continue
        tf_df = daily
        if timeframe == "weekly":
            tf_df = resample_weekly(daily)
        elif timeframe == "monthly":
            tf_df = resample_monthly(daily)
        if tf_df.empty:
            continue

        current_price = float(tf_df["Close"].iloc[-1])
        if current_price >= pivot:
            db.update_signal_status(
                row["signal_id"],
                entry_triggered=1,
                entry_date=today_str,
                status="Triggered",
                current_price=round(current_price, 2),
            )


# ─── Summary stats / report ─────────────────────────────────────────────────

def _build_summary_stats(
    scan_date: str, total_symbols: int, total_patterns: int,
    by_timeframe: dict, by_signal_type: dict, by_rs_band: dict,
    nifty_trend: str,
) -> dict:
    confirmed_df   = db.get_confirmed_breakouts_df(scan_date)
    verge_df       = db.get_on_the_verge_df(scan_date)
    todays_df      = db.get_todays_signals_df(scan_date)
    early_watch    = db.get_todays_early_watch_df(scan_date)

    n_confirmed    = len(confirmed_df)  if not confirmed_df.empty  else 0
    n_on_the_verge = len(verge_df)      if not verge_df.empty      else 0
    n_actionable   = len(todays_df)     if not todays_df.empty     else 0
    n_early_watch  = len(early_watch)   if not early_watch.empty   else 0

    top5: list[dict] = []
    if not todays_df.empty and "breakout_readiness_pct" in todays_df.columns:
        ranked = (
            todays_df
            .dropna(subset=["breakout_readiness_pct"])
            .sort_values("breakout_readiness_pct", ascending=False)
            .head(5)
        )
        top5 = ranked[
            ["symbol", "breakout_readiness_pct", "timeframe", "signal_type"]
        ].to_dict("records")

    return {
        "scan_date":        scan_date,
        "total_symbols":    total_symbols,
        "total_patterns":   total_patterns,
        "n_confirmed":      n_confirmed,
        "n_on_the_verge":   n_on_the_verge,
        "n_actionable":     n_actionable,
        "n_early_watch":    n_early_watch,
        "by_timeframe":     by_timeframe,
        "by_signal_type":   by_signal_type,
        "by_rs_band":       by_rs_band,
        "top5_readiness":   top5,
        "nifty_trend":      nifty_trend,
    }


def _generate_report(scan_date: str, summary_stats: dict) -> None:
    confirmed_breakouts = db.get_confirmed_breakouts_df(scan_date)
    todays_signals      = db.get_todays_signals_df(scan_date)
    watchlist           = db.get_near_breakout_watchlist_df(scan_date)
    on_the_verge        = db.get_on_the_verge_df(scan_date)
    early_watch         = db.get_todays_early_watch_df(scan_date)
    active_tracking     = db.get_active_tracking_df()
    historical          = db.get_historical_signals_df()

    n_confirmed = len(confirmed_breakouts) if not confirmed_breakouts.empty else 0
    n_verge = len(on_the_verge) if not on_the_verge.empty else 0
    log.info("Confirmed Breakouts today: %d  |  On The Verge: %d", n_confirmed, n_verge)

    output_path = cfg.REPORTS_DIR / f"cup_handle_report_{scan_date}.xlsx"
    generate_excel_report(
        confirmed_breakouts, on_the_verge, todays_signals, watchlist, early_watch,
        active_tracking, historical,
        summary_stats, output_path,
    )


# ─── Debug single symbol ───────────────────────────────────────────────────

def _run_debug_single_symbol(symbol: str) -> None:
    log.info("DEBUG MODE: %s", symbol)
    if not symbol.endswith(".NS") and not symbol.startswith("^"):
        symbol = f"{symbol}.NS"

    run_download([symbol], full_refresh=False)
    daily = load_daily(symbol)
    if daily is None:
        log.error("No data available for %s", symbol)
        return

    log.info("%s: %d daily bars, %s to %s", symbol, len(daily),
              daily.index[0].date(), daily.index[-1].date())

    nifty_trend, _ = _load_benchmark()
    rs_ratings, rs_trends = _compute_rs_ratings([symbol])
    rs_val = rs_ratings.get(symbol, 50.0)
    rs_trend_label = rs_trends.get(symbol, "Unknown")
    log.info("RS Rating: %.0f (%s)", rs_val, rs_trend_label)

    for tf, tf_df in [
        ("daily", daily),
        ("weekly", resample_weekly(daily)),
        ("monthly", resample_monthly(daily)),
    ]:
        log.info("-" * 50)
        log.info("Timeframe: %s (%d bars)", tf, len(tf_df))
        any_found = False
        for detector in PATTERN_DETECTORS:
            try:
                sigs = detector(tf_df, tf, symbol, rs_rating=rs_val)
            except Exception:
                log.info("  [%s] detector error:\n%s", detector.__name__, traceback.format_exc())
                continue
            for s in sigs:
                pc.ensure_generic_fields(s)
                any_found = True
                log.info("  Pattern: %s | Signal: %s | Quality: %.1f",
                          s.pattern_type, s.signal_type, s.quality_score)
                log.info("  Structure: %s -> %s -> %s | Depth: %.1f%% (%s)",
                          s.cup_start_date, s.cup_bottom_date, s.cup_end_date,
                          s.cup_depth_pct, s.cup_depth_class)
                log.info("  %s", s.structure_note)
                log.info("  Prior Trend: %.1f%% (%s)", s.prior_uptrend_pct, s.prior_uptrend_tag)
                log.info("  Has Final Base: %s | Base Quality: %.0f/20",
                          s.has_handle, s.handle_quality_subscore)
                log.info("  Pivot: %.2f | Current: %.2f", s.pivot_point, s.current_price)

                readiness = compute_breakout_readiness(s, tf_df, rs_trend_label)
                log.info("  Readiness: %s | %s", readiness["readiness_pct"], readiness["reasons_str"])

                if s.signal_type in READY_SIGNAL_TYPES:
                    cc = detect_confirmation_candle(tf_df, breakout_pos=len(tf_df) - 1)
                    log.info("  Confirmation Candle: %s (%s) — %s", cc["pattern"], cc["strength"], cc["detail"])

                plan = calculate_entry_exit(s, tf_df, nifty_trend)
                log.info("  Entry: %.2f | Stop: %.2f (%s) | T1/T2/T3: %.2f/%.2f/%.2f",
                          plan.entry_price, plan.stop_loss_price, plan.stop_loss_type,
                          plan.target1, plan.target2, plan.target3)
                log.info("  R:R T1/T2/T3: %.2f/%.2f/%.2f | Position: %d shares (₹%.0f)",
                          plan.rr_t1, plan.rr_t2, plan.rr_t3,
                          plan.position_size_shares, plan.capital_required)
        if not any_found:
            log.info("  No pattern detected (Cup & Handle / Double Bottom / VCP / Ascending Triangle).")


# ─── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE Cup & Handle Scanner")
    p.add_argument("--full-refresh", action="store_true",
                    help="Wipe and redownload all historical data")
    p.add_argument("--refresh-universe", action="store_true",
                    help="Force-refresh the NSE symbol list")
    p.add_argument("--debug-symbol", type=str, default=None,
                    help="Run verbose diagnosis for a single symbol")
    return p.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
    except Exception:
        log.error("Fatal error:\n%s", traceback.format_exc())
        sys.exit(1)
