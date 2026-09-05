"""
NSE Cup & Handle Scanner - HTML Dashboard Export
====================================================
Builds a single, self-contained, interactive HTML dashboard from the
day's scan results — a full replacement for scrolling through the
Excel report, following the same self-contained-file / vendored-JS /
template-substitution pattern used by the Flag & Pole scanner's
chart_viewer.

Called once, at the end of main(), after the Excel report has already
been generated (this reads the same database.py query functions the
Excel report used, so the two stay in sync automatically without a
second source of truth).

Design:
  - One flat list of "symbol cards", one per (symbol, timeframe)
    signal, tagged with which of the 7 DB categories it appears in
    (confirmed / verge / watchlist / today / early_watch / active /
    historical). A symbol appearing in multiple categories (e.g. an
    Active Tracking position that's also in today's Confirmed
    Breakouts) is represented once, with all its category memberships
    listed — this is what drives the "×2" multi-category badge in the
    sidebar.
  - Each card carries pre-computed AI explanation text (via
    explain.py), scanner reason PASS/FAIL/WARN rows, and OHLCV data
    for all three timeframes so the chart and timeframe switcher work
    entirely client-side with no further data fetching.
  - The template.html placeholders are substituted directly (no
    Jinja/templating engine dependency, matches Flag & Pole exactly).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
import database as db
import explain
import pattern_common as pc
from cup_handle_detector import quality_band
from downloader import load_daily, resample_monthly, resample_weekly
from logger_utils import get_logger
from readiness import readiness_band

log = get_logger("scanner")

TEMPLATE_PATH = Path(__file__).parent / "chart_viewer" / "template.html"
LWC_JS_PATH = Path(__file__).parent / "vendor" / "lightweight-charts.standalone.production.js"

CATEGORY_QUERIES = [
    ("confirmed",   lambda scan_date: db.get_confirmed_breakouts_df(scan_date)),
    ("verge",       lambda scan_date: db.get_on_the_verge_df(scan_date)),
    ("watchlist",   lambda scan_date: db.get_near_breakout_watchlist_df(scan_date)),
    ("today",       lambda scan_date: db.get_todays_signals_df(scan_date)),
    ("early_watch", lambda scan_date: db.get_todays_early_watch_df(scan_date)),
    ("active",      lambda scan_date: db.get_active_tracking_df()),
    ("historical",  lambda scan_date: db.get_historical_signals_df()),
]

# Fixed display order for the pattern-type tabs — always shown in this
# order (with a live count, greyed out if 0 today) regardless of which
# families actually produced signals, so the tab bar doesn't reflow
# from one day's scan to the next.
PATTERN_FAMILIES = [
    ("cup_handle",        "Cup & Handle"),
    ("double_bottom",     "Double Bottom"),
    ("vcp",               "VCP"),
    ("ascending_triangle", "Ascending Triangle"),
]


def export_html_dashboard(scan_date: str, nifty_trend: str) -> Optional[Path]:
    """
    Main entry point, called once from main.py at the end of a scan
    run. Returns the output path on success, None if there was nothing
    to export (e.g. zero signals today) or the template/JS assets are
    missing (logged as a warning, never crashes the scan).
    """
    if not TEMPLATE_PATH.exists():
        log.warning("chart_export: template.html not found at %s — skipping HTML dashboard", TEMPLATE_PATH)
        return None
    if not LWC_JS_PATH.exists():
        log.warning("chart_export: lightweight-charts JS not found at %s — skipping HTML dashboard", LWC_JS_PATH)
        return None

    rows_by_category, merged = _collect_all_signals(scan_date)
    if not merged:
        log.info("chart_export: no signals to export today — skipping HTML dashboard")
        return None

    symbol_cards = []
    ohlcv_cache: dict[str, pd.DataFrame] = {}

    for key, row in merged.items():
        try:
            card = _build_symbol_card(row, rows_by_category.get(key, []), ohlcv_cache)
            if card is not None:
                symbol_cards.append(card)
        except Exception as e:
            log.debug("chart_export: skipped %s due to error: %s", key, e)

    if not symbol_cards:
        log.info("chart_export: no symbol cards could be built (no chart data) — skipping HTML dashboard")
        return None

    pattern_counts: dict[str, int] = {fam: 0 for fam, _ in PATTERN_FAMILIES}
    for card in symbol_cards:
        fam = card.get("pattern_family", "cup_handle")
        pattern_counts[fam] = pattern_counts.get(fam, 0) + 1
    pattern_summary = [
        {"family": fam, "name": name, "count": pattern_counts.get(fam, 0)}
        for fam, name in PATTERN_FAMILIES
    ]

    data_payload = {
        "scan_date": scan_date,
        "nifty_trend": nifty_trend,
        "symbols": symbol_cards,
        "pattern_summary": pattern_summary,
        "pattern_vocab": pc.PATTERN_VOCAB,
    }

    output_path = _render_html(data_payload, scan_date, nifty_trend)
    log.info("HTML dashboard written: %s (%d symbol cards)", output_path, len(symbol_cards))
    return output_path


# ─── Collect + merge signals across all 7 categories ──────────────────────

def _collect_all_signals(scan_date: str) -> tuple[dict, dict]:
    """
    Returns (rows_by_category, merged) where:
      rows_by_category: {category_key: [row_dict, ...]}
      merged: {(symbol, timeframe, pattern_family): row_dict}  — one row
              per unique (symbol, timeframe, pattern_family) triple,
              preferring the most detail-rich category's version when
              the same triple appears more than once (categories are
              queried in priority order, first write wins, since
              'confirmed' has the richest guaranteed fields and later
              categories may have NULLs for tracking-only fields).

              pattern_family is part of the key (not just symbol+
              timeframe) because a single symbol/timeframe can now
              legitimately carry more than one pattern type at once
              (e.g. a stock showing both a Cup & Handle AND a VCP on
              its daily chart) — keying on symbol+timeframe alone would
              silently drop all but one of them.
    Each row_dict also gets a "_categories" list attached, listing
    every category key it was found under.
    """
    rows_by_category: dict[str, list[dict]] = {}
    merged: dict[tuple, dict] = {}
    categories_seen: dict[tuple, set] = {}

    for key, fn in CATEGORY_QUERIES:
        try:
            df = fn(scan_date)
        except Exception as e:
            log.debug("chart_export: query for category '%s' failed: %s", key, e)
            continue
        if df is None or df.empty:
            rows_by_category[key] = []
            continue

        records = df.to_dict("records")
        rows_by_category[key] = records

        for row in records:
            pk = (row.get("symbol"), row.get("timeframe"), row.get("pattern_family") or "cup_handle")
            if pk not in categories_seen:
                categories_seen[pk] = set()
            categories_seen[pk].add(key)
            if pk not in merged:
                merged[pk] = row

    for pk, row in merged.items():
        row["_categories"] = sorted(categories_seen.get(pk, set()))

    return rows_by_category, merged


# ─── Build one symbol card (all fields the HTML template needs) ───────────

def _build_symbol_card(row: dict, category_rows: list, ohlcv_cache: dict) -> Optional[dict]:
    symbol = row.get("symbol")
    timeframe = row.get("timeframe", "daily")
    if not symbol:
        return None

    daily = ohlcv_cache.get(symbol)
    if daily is None:
        daily = load_daily(symbol)
        ohlcv_cache[symbol] = daily
    if daily is None or daily.empty:
        return None

    timeframes_json = _build_timeframes_json(daily)
    if not any(timeframes_json.values()):
        return None

    quality_score = _num(row.get("quality_score"), 0.0)
    readiness_pct = _num(row.get("breakout_readiness_pct"))

    rating = explain.overall_rating(row)
    scanner_reasons = explain.scanner_reasons(row)
    why_buy = explain.why_buy(row)
    full_expl = explain.full_explanation(row)

    pattern_family = row.get("pattern_family") or "cup_handle"
    key_points, trendlines = _build_annotations(row)

    confirmation_candle = None
    if row.get("confirmation_candle_pattern"):
        confirmation_candle = {
            "found": bool(row.get("confirmation_candle_found", 0)),
            "pattern": row.get("confirmation_candle_pattern") or "",
            "strength": row.get("confirmation_candle_strength") or "",
            "date": row.get("confirmation_candle_date"),
            "detail": row.get("confirmation_candle_detail") or "",
        }

    return {
        "symbol": symbol,
        "company_name": row.get("company_name") or "",
        "sector": row.get("sector") or "",
        "timeframe": timeframe,
        "categories": row.get("_categories", []),

        "pattern_type": row.get("pattern_type") or "",
        "pattern_family": pattern_family,
        "pattern_vocab": pc.vocab_for(pattern_family),
        "pattern_stage": row.get("signal_type") or "",
        "signal_type": row.get("signal_type") or "",
        "quality_score": quality_score,
        "quality_band": quality_band(quality_score),
        "has_handle": bool(row.get("has_handle", 0)),
        "structure_note": row.get("structure_note") or "",

        "cup_start_date": _datestr(row.get("cup_start_date")),
        "cup_bottom_date": _datestr(row.get("cup_bottom_date")),
        "cup_end_date": _datestr(row.get("cup_end_date")),
        "left_rim_price": _num(row.get("left_rim_price")),
        "cup_bottom_price": _num(row.get("cup_bottom_price")),
        "right_rim_price": _num(row.get("right_rim_price")),
        "cup_depth_pct": _num(row.get("cup_depth_pct")),
        "cup_depth_class": row.get("cup_depth_class") or "",
        "cup_shape": row.get("cup_shape") or "",
        "prior_uptrend_pct": _num(row.get("prior_uptrend_pct")),
        "prior_uptrend_tag": row.get("prior_uptrend_tag") or "",

        "handle_start_date": _datestr(row.get("handle_start_date")),
        "handle_end_date": _datestr(row.get("handle_end_date")),
        "handle_low_price": _num(row.get("handle_low_price")),
        "handle_depth_pct": _num(row.get("handle_depth_pct")),
        "handle_quality_subscore": _num(row.get("handle_quality_subscore")),

        "pivot_point": _num(row.get("pivot_point")),
        "current_price": _num(row.get("current_price")),
        "price_vs_pivot_pct": _num(row.get("price_vs_pivot_pct")),

        "breakout_readiness_pct": readiness_pct,
        "readiness_band": readiness_band(readiness_pct),
        "readiness_reasons": row.get("readiness_reasons") or "",
        "readiness_atr_contract": bool(row.get("readiness_atr_contract", 0)),
        "readiness_above_50ma": bool(row.get("readiness_above_50ma", 0)),

        "entry_price": _num(row.get("entry_price")),
        "entry_zone_high": _num(row.get("entry_zone_high")),
        "entry_type": row.get("entry_type") or "",

        "stop_loss_price": _num(row.get("stop_loss_price")),
        "stop_loss_pct": _num(row.get("stop_loss_pct")),
        "stop_loss_type": row.get("stop_loss_type") or "",
        "atr_14": _num(row.get("atr_14")),

        "target1": _num(row.get("target1")),
        "target2": _num(row.get("target2")),
        "target3": _num(row.get("target3")),
        "rr_t1": _num(row.get("rr_t1")),
        "rr_t2": _num(row.get("rr_t2")),
        "rr_t3": _num(row.get("rr_t3")),

        "position_size_shares": _int(row.get("position_size_shares")),
        "capital_required": _num(row.get("capital_required")),
        "risk_amount": _num(row.get("risk_amount")),

        "volume_ratio": _num(row.get("volume_ratio")),
        "volume_confirmed_label": row.get("volume_confirmed_label") or "",

        "rs_rating": _num(row.get("rs_rating")),
        "rs_trend": row.get("rs_trend") or "",
        "rs_tag": row.get("rs_tag") or "",
        "rsi_val": _num(row.get("rsi_val")),
        "adx_val": _num(row.get("adx_val")),

        "mtf_confluence": bool(row.get("mtf_confluence", 0)),
        "mtf_timeframes": row.get("mtf_timeframes") or "",
        "cross_pattern_confluence": bool(row.get("cross_pattern_confluence", 0)),
        "cross_pattern_desc": row.get("cross_pattern_desc") or "",

        "liquidity_ok": bool(row.get("liquidity_ok", 1)),
        "liquidity_warning": row.get("liquidity_warning") or "",

        "sell_notes": row.get("sell_notes") or "",
        "remarks": row.get("remarks") or "",

        "status": row.get("status") or "",

        "rating": rating,
        "scanner_reasons": scanner_reasons,
        "why_buy": why_buy,
        "explanation": full_expl,

        "key_points": key_points,
        "trendlines": trendlines,
        "confirmation_candle": confirmation_candle,

        "timeframes": timeframes_json,
    }


# ─── Generic chart annotations (key_points / trendlines) ──────────────────

def _build_annotations(row: dict) -> tuple[list[dict], list[dict]]:
    """
    Prefer the rich, detector-authored key_points_json / trendlines_json
    columns (every signal detected since the multi-pattern extension
    carries these). Falls back to deriving a plain 3-point structure
    line from the legacy rim/handle fields for any signal already
    sitting in the database from before this column existed (old Active
    Tracking / Historical rows) — same fallback as
    pattern_common._derive_annotations_from_cup_handle, duplicated
    lightly here since this side only has a DB row dict, not a live
    signal object.
    """
    kp_raw, tl_raw = row.get("key_points_json"), row.get("trendlines_json")
    if kp_raw or tl_raw:
        try:
            key_points = json.loads(kp_raw) if kp_raw else []
        except Exception:
            key_points = []
        try:
            trendlines = json.loads(tl_raw) if tl_raw else []
        except Exception:
            trendlines = []
        if key_points or trendlines:
            return key_points, trendlines

    # ── Fallback for pre-migration rows ──
    key_points: list[dict] = []
    trendlines: list[dict] = []
    cup_start, cup_bottom, cup_end = row.get("cup_start_date"), row.get("cup_bottom_date"), row.get("cup_end_date")
    left_rim, bottom_price, right_rim = row.get("left_rim_price"), row.get("cup_bottom_price"), row.get("right_rim_price")
    if cup_start and cup_bottom and cup_end and left_rim is not None and bottom_price is not None and right_rim is not None:
        key_points = [
            {"date": _datestr(cup_start), "price": left_rim, "label": "Left Rim", "role": "peak"},
            {"date": _datestr(cup_bottom), "price": bottom_price, "label": "Cup Bottom", "role": "trough"},
            {"date": _datestr(cup_end), "price": right_rim, "label": "Right Rim", "role": "peak"},
        ]
        trendlines = [{
            "role": "structure", "style": "solid", "color": "#bc8cff", "width": 2,
            "points": [[_datestr(cup_start), left_rim], [_datestr(cup_bottom), bottom_price],
                       [_datestr(cup_end), right_rim]],
        }]
        if row.get("has_handle") and row.get("handle_start_date") and row.get("handle_end_date") and row.get("handle_low_price") is not None:
            hs, he, hl = row.get("handle_start_date"), row.get("handle_end_date"), row.get("handle_low_price")
            key_points.append({"date": _datestr(he), "price": hl, "label": "Handle Low", "role": "trough"})
            trendlines.append({"role": "base_top", "style": "dashed", "color": "#d29922", "width": 1,
                                "points": [[_datestr(hs), right_rim], [_datestr(he), right_rim]]})
            trendlines.append({"role": "base_bottom", "style": "dashed", "color": "#d29922", "width": 1,
                                "points": [[_datestr(hs), hl], [_datestr(he), hl]]})
    return key_points, trendlines


# ─── OHLCV -> JSON bars for all three timeframes ───────────────────────────

def _build_timeframes_json(daily: pd.DataFrame) -> dict:
    out = {}
    try:
        out["1D"] = _bars_to_json(daily.tail(cfg.CHART_LOOKBACK_DAILY_BARS))
    except Exception:
        out["1D"] = []
    try:
        weekly = resample_weekly(daily)
        out["1W"] = _bars_to_json(weekly.tail(cfg.CHART_LOOKBACK_WEEKLY_BARS))
    except Exception:
        out["1W"] = []
    try:
        monthly = resample_monthly(daily)
        out["1M"] = _bars_to_json(monthly.tail(cfg.CHART_LOOKBACK_MONTHLY_BARS))
    except Exception:
        out["1M"] = []
    return out


def _bars_to_json(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for idx, row in df.iterrows():
        try:
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            v = float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0.0
            if any(math.isnan(x) for x in (o, h, l, c)):
                continue
            out.append({
                "time": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                "o": round(o, 2), "h": round(h, 2), "l": round(l, 2), "c": round(c, 2),
                "v": round(v, 0),
            })
        except Exception:
            continue
    return out


# ─── Small safe-conversion helpers (NaN-safe, JSON-safe) ──────────────────

def _num(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


def _int(val, default=None):
    n = _num(val, default)
    return int(n) if n is not None else default


def _datestr(val) -> Optional[str]:
    if val is None or val == "":
        return None
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)


# ─── Render: substitute placeholders into template.html ───────────────────

def _json_safe(obj):
    """
    Recursively walk any nested structure (dict/list/tuple) and replace
    NaN/Inf floats — including numpy scalar types, which behave like
    floats but are NOT caught by isinstance(x, float) — with None.

    This exists because individual field-level sanitizers (_num, _int
    in this module) only cover values explicitly passed through them.
    explain.py's returned dicts (rating/scanner_reasons/why_buy/
    explanation) are embedded as-is and can carry a stray NaN from a
    DB row that was never routed through _num — for example if a
    computed ratio embedded a numpy.float64 NaN into an f-string
    default or a dict value directly. A single unsanitized NaN
    anywhere in the nested payload makes json.dumps(..., allow_nan=False)
    raise ValueError and abort the entire HTML export (see: the
    "Out of range float values are not JSON compliant" crash). Walking
    the whole structure once, right before serialization, is the only
    approach that's actually robust against every code path that could
    introduce a bad float, present and future.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    # numpy scalar types (np.float64, np.int64, etc.) implement
    # __float__ but are not `isinstance(x, float)` — convert explicitly.
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _render_html(data_payload: dict, scan_date: str, nifty_trend: str) -> Path:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    lwc_js = LWC_JS_PATH.read_text(encoding="utf-8")

    safe_payload = _json_safe(data_payload)
    data_json = json.dumps(safe_payload, ensure_ascii=False, allow_nan=False)

    html = template
    html = html.replace("__SCAN_DATE__", scan_date)
    html = html.replace("__NIFTY_TREND__", nifty_trend or "Unknown")
    html = html.replace("__SYMBOL_COUNT__", str(len(data_payload["symbols"])))
    html = html.replace("/*__LIGHTWEIGHT_CHARTS_JS__*/", lwc_js)
    html = html.replace("/*__CHART_DATA_JSON__*/", data_json)

    cfg.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = cfg.CHARTS_DIR / f"cup_handle_dashboard_{scan_date}.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
