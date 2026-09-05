"""
NSE Cup & Handle Scanner - Excel Report Generator
====================================================
Implements spec v3 Part 5: a 5-sheet Excel workbook with full pattern
geometry, entry/exit, readiness, and tracking detail, colour-coded for
fast visual triage.

Uses xlsxwriter (via pandas ExcelWriter) for richer formatting control
than openpyxl offers (conditional formats, frozen panes, autofilter).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

import config as cfg

import config as cfg
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Column layout for Sheet 1 ─────────────────────────────────────────────

SHEET1_COLUMNS = [
    # (db_column, display_header, width, format_key)
    ("symbol", "Symbol", 14, None),
    ("company_name", "Company Name", 24, None),
    ("sector", "Sector", 16, None),
    ("timeframe", "Timeframe", 10, None),
    ("mtf_confluence", "MTF Confluence", 12, "bool"),
    ("mtf_timeframes", "MTF Timeframes", 16, None),
    ("pattern_type", "Pattern Type", 14, None),
    ("signal_type", "Signal Type", 14, None),
    ("quality_score", "Quality Score", 12, "num1"),

    ("cup_start_date", "Cup Start", 12, "date"),
    ("cup_bottom_date", "Structure Low Date", 14, "date"),
    ("cup_end_date", "Cup End (Right Rim)", 16, "date"),
    ("left_rim_price", "Key High", 10, "money"),
    ("cup_bottom_price", "Key Low", 10, "money"),
    ("right_rim_price", "Confirm High", 10, "money"),
    ("cup_depth_pct", "Structure Depth %", 11, "pct1"),
    ("cup_depth_class", "Depth Class", 16, None),
    ("cup_depth_verify_flag", "Verify Manually", 12, "bool"),
    ("cup_duration_bars", "Cup Duration (bars)", 14, "int"),
    ("cup_shape", "Structure Shape", 11, None),
    ("recovery_pct", "Recovery %", 10, "pct1"),
    ("prior_uptrend_pct", "Prior Uptrend %", 13, "pct1"),
    ("prior_uptrend_tag", "Prior Uptrend Tag", 20, None),

    ("handle_start_date", "Handle Start", 12, "date"),
    ("handle_end_date", "Handle End", 12, "date"),
    ("handle_low_price", "Base Low", 10, "money"),
    ("handle_depth_pct", "Handle Depth %", 13, "pct1"),
    ("handle_duration_bars", "Handle Duration (bars)", 16, "int"),
    ("handle_quality_subscore", "Base Quality (0-20)", 16, "num1"),

    ("pivot_point", "Pivot Point", 11, "money"),
    ("current_price", "Current Price", 12, "money"),
    ("price_vs_pivot_pct", "Price vs Pivot %", 14, "pct1"),

    ("breakout_readiness_pct", "Breakout Readiness %", 16, "num0"),
    ("readiness_reasons", "Readiness Reasons", 40, None),

    ("entry_price", "Entry Price", 11, "money"),
    ("entry_zone_high", "Entry Zone High", 13, "money"),
    ("entry_type", "Entry Type", 14, None),
    ("extended_warning", "Extended Warning", 22, None),
    ("volume_ratio", "Volume Ratio", 11, "num2"),
    ("volume_confirmed_label", "Volume Confirmed", 14, None),

    ("stop_loss_price", "Stop Loss Price", 13, "money"),
    ("stop_loss_pct", "Stop Loss %", 11, "pct1"),
    ("stop_loss_type", "Stop Loss Type", 22, None),
    ("atr_14", "ATR (14)", 9, "money"),
    ("risk_per_share", "Risk/Share", 10, "money"),
    ("position_size_shares", "Position Size (shares)", 16, "int"),
    ("capital_required", "Capital Required (INR)", 16, "money0"),
    ("risk_amount", "Risk Amount (INR)", 14, "money0"),
    ("portfolio_risk_pct", "Portfolio Risk %", 13, "pct2"),

    ("target1", "Target 1 (T1)", 12, "money"),
    ("target2", "Target 2 (T2)", 12, "money"),
    ("target3", "Target 3 (T3)", 12, "money"),
    ("rr_t1", "R:R at T1", 10, "num2"),
    ("rr_t2", "R:R at T2", 10, "num2"),
    ("rr_t3", "R:R at T3", 10, "num2"),
    ("eight_week_hold_candidate", "8-Week Hold Candidate", 16, "bool"),

    ("rs_rating", "RS Rating", 10, "num0"),
    ("rs_trend", "RS Trend", 11, None),
    ("rs_tag", "RS Tag", 12, None),
    ("rsi_val", "RSI (14)", 9, "num1"),
    ("adx_val", "ADX (14)", 9, "num1"),
    ("ma_short", "MA Short", 10, "money"),
    ("ma_mid", "MA Mid", 10, "money"),
    ("ma_long", "MA Long", 10, "money"),
    ("price_vs_ma_short_pct", "Price vs MA Short %", 16, "pct1"),

    ("nifty_trend", "Nifty Trend", 11, None),
    ("market_note", "Market Note", 22, None),
    ("weak_momentum_note", "Momentum Note", 16, None),
    ("below_50ma_note", "MA Note", 16, None),
    ("liquidity_warning", "Liquidity Warning", 26, None),

    ("sell_notes", "Sell Notes (Checklist)", 50, None),
    ("remarks", "Remarks", 30, None),
]

WATCHLIST_COLUMNS = [
    ("symbol", "Symbol", 14, None),
    ("timeframe", "Timeframe", 10, None),
    ("quality_score", "Quality Score", 12, "num1"),
    ("breakout_readiness_pct", "Breakout Readiness %", 16, "num0"),
    ("readiness_reasons", "Readiness Reasons", 40, None),
    ("pivot_point", "Pivot Point", 11, "money"),
    ("current_price", "Current Price", 12, "money"),
    ("price_vs_pivot_pct", "Price vs Pivot %", 14, "pct1"),
    ("entry_price", "Entry Price", 11, "money"),
    ("stop_loss_price", "Stop Loss", 11, "money"),
    ("target1", "T1", 10, "money"),
    ("target2", "T2", 10, "money"),
    ("rr_t2", "R:R T2", 9, "num2"),
    ("rs_rating", "RS Rating", 10, "num0"),
    ("volume_confirmed_label", "Volume Confirmed", 14, None),
    ("remarks", "Remarks", 30, None),
]

CONFIRMED_BREAKOUT_COLUMNS = [
    # Every field a trader needs to act immediately — complete decision
    # package in a single row. No need to cross-reference other sheets.

    # Identity & classification
    ("symbol",                  "Symbol",               14, None),
    ("company_name",            "Company",              22, None),
    ("sector",                  "Sector",               14, None),
    ("timeframe",               "Timeframe",            10, None),
    ("mtf_confluence",          "MTF",                  7,  "bool"),
    ("mtf_timeframes",          "MTF Timeframes",       16, None),
    ("signal_type",             "Signal",               14, None),

    # Conviction scores
    ("breakout_readiness_pct",  "Readiness %",          12, "num0"),
    ("readiness_reasons",       "Readiness Factors",    40, None),
    ("quality_score",           "Quality Score",        12, "num1"),
    ("handle_quality_subscore", "Base Quality (/20)", 15, "num1"),

    # Cup & handle structure
    ("cup_depth_pct",           "Structure Depth %",          11, "num1"),
    ("cup_depth_class",         "Cup Class",            14, None),
    ("cup_shape",               "Structure Shape",            11, None),
    ("prior_uptrend_tag",       "Prior Trend",          22, None),
    ("handle_depth_pct",        "Handle Depth %",       13, "num1"),

    # Pivot & current price
    ("pivot_point",             "Pivot",                10, "money"),
    ("current_price",           "Current Price",        12, "money"),
    ("price_vs_pivot_pct",      "vs Pivot %",           10, "num1"),
    ("volume_ratio",            "Vol Ratio",            10, "num2"),
    ("volume_confirmed_label",  "Vol Confirmed",        13, None),

    # Entry
    ("entry_price",             "Entry Price",          11, "money"),
    ("entry_zone_high",         "Entry Zone High",      14, "money"),

    # Risk — stop capped warning first so it's impossible to miss
    ("stop_loss_type",          "⚠ Stop Type",          20, None),
    ("stop_loss_price",         "Stop Loss",            10, "money"),
    ("stop_loss_pct",           "Stop %",               9,  "num1"),
    ("atr_14",                  "ATR",                  9,  "money"),
    ("risk_per_share",          "Risk/Share ₹",         12, "money"),

    # Position sizing
    ("position_size_shares",    "Shares",               9,  "int"),
    ("capital_required",        "Capital ₹",            12, "money0"),
    ("risk_amount",             "Risk ₹",               10, "money0"),

    # Targets & R:R
    ("target1",                 "T1 (+20%)",            11, "money"),
    ("target2",                 "T2 (Projected)",       14, "money"),
    ("target3",                 "T3 (Fib 1.618x)",      15, "money"),
    ("rr_t1",                   "R:R T1",               9,  "num2"),
    ("rr_t2",                   "R:R T2",               9,  "num2"),
    ("rr_t3",                   "R:R T3",               9,  "num2"),
    ("eight_week_hold_candidate", "8-Wk Hold?",         11, "bool"),

    # Technicals
    ("rs_rating",               "RS Rating",            10, "num0"),
    ("rs_trend",                "RS Trend",             11, None),
    ("rs_tag",                  "RS Tag",               12, None),
    ("rsi_val",                 "RSI",                  8,  "num1"),
    ("adx_val",                 "ADX",                  8,  "num1"),
    ("price_vs_ma_short_pct",   "vs 50MA %",            10, "num1"),

    # Market
    ("nifty_trend",             "Nifty Trend",          11, None),
    ("market_note",             "Market Note",          22, None),

    # Action notes
    ("sell_notes",              "Sell Rules",           50, None),
    ("remarks",                 "Remarks",              30, None),
]

ON_THE_VERGE_COLUMNS = [
    # Second-tier confirmed list — more rows than Confirmed Breakouts,
    # still every row has passed real gates (handle required, minimum
    # readiness/quality/volume/handle-quality all enforced in SQL).
    # Kept leaner than Confirmed Breakouts since this sheet is meant to
    # be scanned quickly, not decided from in isolation.
    ("symbol",                  "Symbol",               14, None),
    ("company_name",            "Company",              22, None),
    ("sector",                  "Sector",               14, None),
    ("timeframe",               "Timeframe",            10, None),
    ("mtf_confluence",          "MTF",                  7,  "bool"),

    ("breakout_readiness_pct",  "Readiness %",          12, "num0"),
    ("readiness_reasons",       "Readiness Factors",    40, None),
    ("quality_score",           "Quality Score",        12, "num1"),
    ("handle_quality_subscore", "Base Quality (/20)", 15, "num1"),

    ("cup_depth_class",         "Cup Class",            14, None),
    ("prior_uptrend_tag",       "Prior Trend",          22, None),

    ("pivot_point",             "Pivot",                10, "money"),
    ("current_price",           "Current Price",        12, "money"),
    ("price_vs_pivot_pct",      "vs Pivot %",           10, "num1"),
    ("volume_ratio",            "Vol Ratio",            10, "num2"),

    ("entry_price",             "Entry Price",          11, "money"),
    ("stop_loss_price",         "Stop Loss",            10, "money"),
    ("stop_loss_type",          "Stop Type",            20, None),
    ("target1",                 "T1 (+20%)",            11, "money"),
    ("target2",                 "T2 (Projected)",       14, "money"),
    ("rr_t2",                   "R:R T2",               9,  "num2"),

    ("rs_rating",               "RS Rating",            10, "num0"),
    ("rs_trend",                "RS Trend",             11, None),
    ("rsi_val",                 "RSI",                  8,  "num1"),

    ("remarks",                 "Remarks",              30, None),
]


ACTIVE_TRACKING_COLUMNS = [
    ("symbol", "Symbol", 14, None),
    ("timeframe", "Timeframe", 10, None),
    ("entry_date", "Entry Date", 12, "date"),
    ("entry_price", "Entry Price", 11, "money"),
    ("current_price", "Current Price", 12, "money"),
    ("stop_loss_price", "Stop Loss (Original)", 16, "money"),
    ("target1", "T1", 10, "money"),
    ("target2", "T2", 10, "money"),
    ("target3", "T3", 10, "money"),
    ("t1_achieved", "T1 Hit", 9, "bool"),
    ("t2_achieved", "T2 Hit", 9, "bool"),
    ("t3_achieved", "T3 Hit", 9, "bool"),
    ("eight_week_hold_candidate", "8-Week Rule", 12, "bool"),
    ("status", "Status", 16, None),
    ("sell_notes", "Sell Notes", 50, None),
]

EARLY_WATCH_COLUMNS = [
    # Compact column set — these are early-stage, no entry/exit urgency
    ("symbol", "Symbol", 14, None),
    ("company_name", "Company Name", 22, None),
    ("sector", "Sector", 14, None),
    ("timeframe", "Timeframe", 10, None),
    ("mtf_confluence", "MTF Confluence", 12, "bool"),
    ("quality_score", "Quality Score", 12, "num1"),
    ("cup_start_date", "Cup Start", 12, "date"),
    ("cup_bottom_date", "Key Low", 12, "date"),
    ("cup_end_date", "Confirm High Date", 14, "date"),
    ("left_rim_price", "Key High", 10, "money"),
    ("cup_bottom_price", "Key Low Price", 14, "money"),
    ("right_rim_price", "Confirm High Price", 14, "money"),
    ("cup_depth_pct", "Structure Depth %", 11, "num1"),
    ("cup_depth_class", "Depth Class", 16, None),
    ("cup_shape", "Structure Shape", 11, None),
    ("prior_uptrend_pct", "Prior Uptrend %", 13, "num1"),
    ("prior_uptrend_tag", "Prior Uptrend Tag", 22, None),
    ("recovery_pct", "Recovery %", 10, "num1"),
    ("pivot_point", "Pivot Point", 11, "money"),
    ("current_price", "Current Price", 12, "money"),
    ("price_vs_pivot_pct", "Price vs Pivot %", 14, "num1"),
    ("entry_price", "Entry (if handle forms)", 18, "money"),
    ("stop_loss_price", "Stop Loss", 11, "money"),
    ("target1", "T1 (+20%)", 10, "money"),
    ("target2", "T2 (Measured)", 13, "money"),
    ("rr_t2", "R:R T2", 9, "num2"),
    ("rs_rating", "RS Rating", 10, "num0"),
    ("rs_trend", "RS Trend", 11, None),
    ("rsi_val", "RSI (14)", 9, "num1"),
    ("adx_val", "ADX (14)", 9, "num1"),
    ("remarks", "Remarks", 30, None),
]

HISTORICAL_COLUMNS = [
    ("symbol", "Symbol", 14, None),
    ("timeframe", "Timeframe", 10, None),
    ("scan_date", "Detection Date", 13, "date"),
    ("entry_date", "Entry Date", 12, "date"),
    ("entry_price", "Entry Price", 11, "money"),
    ("exit_date", "Exit Date", 12, "date"),
    ("exit_price", "Exit Price", 11, "money"),
    ("exit_type", "Exit Type", 14, None),
    ("realised_rr", "R:R Realised", 12, "num2"),
    ("hold_days", "Hold Days", 10, "int"),
    ("quality_score", "Quality Score", 12, "num1"),
    ("rs_rating", "RS at Entry", 10, "num0"),
    ("pattern_type", "Pattern Type", 14, None),
    ("status", "Status", 16, None),
]


def generate_excel_report(
    confirmed_breakouts: pd.DataFrame,
    on_the_verge: pd.DataFrame,
    todays_signals: pd.DataFrame,
    watchlist: pd.DataFrame,
    early_watch: pd.DataFrame,
    active_tracking: pd.DataFrame,
    historical: pd.DataFrame,
    summary_stats: dict,
    output_path: Path,
) -> Path:
    """Build the full 8-sheet Excel workbook and save to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confirmed_breakouts = _round_floats(confirmed_breakouts)
    on_the_verge         = _round_floats(on_the_verge)
    todays_signals       = _round_floats(todays_signals)
    watchlist            = _round_floats(watchlist)
    early_watch          = _round_floats(early_watch)
    active_tracking      = _round_floats(active_tracking)
    historical           = _round_floats(historical)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        fmts = _build_formats(workbook)

        # ── Sheet 1: Confirmed Breakouts — the most important tab ──────────
        _write_sheet(
            writer, workbook, fmts, confirmed_breakouts,
            CONFIRMED_BREAKOUT_COLUMNS,
            "🔥 Confirmed Breakouts",
            sort_cols=["mtf_confluence", "breakout_readiness_pct", "quality_score"],
            sort_asc=[False, False, False],
            extra_conditional=_apply_confirmed_breakout_formats,
            tab_color="#FF0000",   # red — highest urgency, act today
        )

        # ── Sheet 2: On The Verge — wider second-tier confirmed list ───────
        _write_sheet(
            writer, workbook, fmts, on_the_verge,
            ON_THE_VERGE_COLUMNS,
            "🎯 On The Verge",
            sort_cols=["breakout_readiness_pct", "quality_score"],
            sort_asc=[False, False],
            extra_conditional=_apply_on_the_verge_formats,
            tab_color="#FFC000",   # gold — one tier down from red
        )

        _write_sheet(
            writer, workbook, fmts, watchlist, WATCHLIST_COLUMNS,
            "⚡ Near Breakout Watchlist",
            sort_cols=["breakout_readiness_pct"], sort_asc=[False],
            extra_conditional=_apply_watchlist_conditional_formats,
            tab_color="#FF8C00",
        )

        _write_sheet(
            writer, workbook, fmts, todays_signals, SHEET1_COLUMNS,
            "📋 Today's Signals",
            sort_cols=["mtf_confluence", "quality_score"], sort_asc=[False, False],
            extra_conditional=_apply_sheet1_conditional_formats,
            tab_color="#375623",
        )

        _write_sheet(
            writer, workbook, fmts, early_watch, EARLY_WATCH_COLUMNS,
            "👁 Early Watch (Cup Only)",
            sort_cols=["quality_score"], sort_asc=[False],
            tab_color="#4472C4",
        )

        _write_sheet(
            writer, workbook, fmts, active_tracking, ACTIVE_TRACKING_COLUMNS,
            "📈 Active Tracking",
            sort_cols=["entry_date"], sort_asc=[False],
            tab_color="#7030A0",
        )

        _write_sheet(
            writer, workbook, fmts, historical, HISTORICAL_COLUMNS,
            "📚 Historical Signals",
            sort_cols=["scan_date"], sort_asc=[False],
            extra_summary=_historical_summary_block(historical),
            tab_color="#7F7F7F",
        )

        _write_strategy_summary(writer, workbook, fmts, summary_stats)

    log.info("Excel report written: %s", output_path)
    return output_path


def _round_floats(df: pd.DataFrame, dp: int = 2) -> pd.DataFrame:
    """
    Round all float columns to `dp` decimal places before writing to
    Excel. This prevents yfinance's raw float64 values (e.g.
    1112.599975585938) from appearing in cells instead of 1112.60.
    Non-numeric columns are untouched.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include="float64").columns:
        out[col] = out[col].round(dp)
    return out


# ─── Formats ────────────────────────────────────────────────────────────────

def _build_formats(workbook) -> dict:
    return {
        "header": workbook.add_format({
            "bold": True, "bg_color": "#1F4E78", "font_color": "white",
            "border": 1, "valign": "vcenter", "text_wrap": True,
        }),
        "money": workbook.add_format({"num_format": "₹#,##0.00"}),
        "money0": workbook.add_format({"num_format": "₹#,##0"}),
        "pct1": workbook.add_format({"num_format": "0.0%"}),
        "pct2": workbook.add_format({"num_format": "0.00%"}),
        "num0": workbook.add_format({"num_format": "0"}),
        "num1": workbook.add_format({"num_format": "0.0"}),
        "num2": workbook.add_format({"num_format": "0.00"}),
        "int": workbook.add_format({"num_format": "0"}),
        "date": workbook.add_format({"num_format": "dd-mmm-yyyy"}),
        "bool": workbook.add_format({"align": "center"}),
        "default": workbook.add_format({}),
        "green_bg": workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"}),
        "yellow_bg": workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"}),
        "orange_bold": workbook.add_format({"bold": True, "font_color": "#E36C09"}),
        "blue_bold": workbook.add_format({"bold": True, "font_color": "#1F4E78"}),
        "red_text": workbook.add_format({"font_color": "#C00000"}),
        "green_bold": workbook.add_format({"bold": True, "font_color": "#006100"}),
        "stop_capped_row": workbook.add_format({
            "bg_color": "#FFE0E0",   # light red background — noticeable but not aggressive
            "font_color": "#C00000",
        }),
        "title": workbook.add_format({"bold": True, "font_size": 16}),
        "subtitle": workbook.add_format({"bold": True, "font_size": 12, "font_color": "#1F4E78"}),
        "wrap": workbook.add_format({"text_wrap": True, "valign": "top"}),
    }


# ─── Generic sheet writer ──────────────────────────────────────────────────

def _write_sheet(
    writer, workbook, fmts, df: pd.DataFrame, column_spec: list,
    sheet_name: str,
    sort_cols: list[str] | None = None,
    sort_asc: list[bool] | None = None,
    extra_conditional=None,
    extra_summary: list[str] | None = None,
    tab_color: str | None = None,
) -> None:
    if df is None or df.empty:
        empty_df = pd.DataFrame(columns=[c[0] for c in column_spec])
        empty_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        ws = writer.sheets[sheet_name]
        if tab_color:
            ws.set_tab_color(tab_color)
        for i, (_, header, width, _) in enumerate(column_spec):
            ws.write(0, i, header, fmts["header"])
            ws.set_column(i, i, width)
        ws.write(2, 0, "No signals found for this category in today's scan.")
        return

    cols_present = [c for c in column_spec if c[0] in df.columns]
    db_cols = [c[0] for c in cols_present]
    out = df[db_cols].copy()

    if sort_cols:
        valid_sort = [c for c in sort_cols if c in out.columns]
        if valid_sort:
            asc = sort_asc[: len(valid_sort)] if sort_asc else [True] * len(valid_sort)
            out = out.sort_values(by=valid_sort, ascending=asc, na_position="last")

    for date_col in ("cup_start_date", "cup_bottom_date", "cup_end_date",
                      "handle_start_date", "handle_end_date", "scan_date",
                      "entry_date", "exit_date"):
        if date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    out.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
    ws = writer.sheets[sheet_name]
    if tab_color:
        ws.set_tab_color(tab_color)

    for i, (_, header, width, _) in enumerate(cols_present):
        ws.write(0, i, header, fmts["header"])
        ws.set_column(i, i, width)

    n_rows = len(out)
    for i, (db_col, _, _, fmt_key) in enumerate(cols_present):
        if fmt_key and fmt_key in fmts:
            ws.set_column(i, i, None, fmts[fmt_key])

    ws.freeze_panes(1, 1)
    if n_rows > 0:
        ws.autofilter(0, 0, n_rows, len(cols_present) - 1)

    if extra_conditional:
        extra_conditional(ws, fmts, cols_present, n_rows)

    if extra_summary:
        start_row = n_rows + 3
        for j, line in enumerate(extra_summary):
            ws.write(start_row + j, 0, line)


# ─── Sheet 1 conditional formatting ────────────────────────────────────────

def _apply_sheet1_conditional_formats(ws, fmts, cols_present, n_rows) -> None:
    if n_rows == 0:
        return
    col_index = {c[0]: i for i, c in enumerate(cols_present)}

    def col_range(name):
        if name not in col_index:
            return None
        idx = col_index[name]
        return idx, f"{_xl_col(idx)}2:{_xl_col(idx)}{n_rows + 1}"

    # Quality score banding
    if "quality_score" in col_index:
        idx, rng = col_range("quality_score")
        ws.conditional_format(rng, {
            "type": "cell", "criteria": ">=", "value": cfg.QUALITY_BAND_HIGH,
            "format": fmts["green_bg"],
        })
        ws.conditional_format(rng, {
            "type": "cell", "criteria": "between",
            "minimum": cfg.QUALITY_BAND_MEDIUM, "maximum": cfg.QUALITY_BAND_HIGH - 0.01,
            "format": fmts["yellow_bg"],
        })

    # Breakout readiness banding
    if "breakout_readiness_pct" in col_index:
        idx, rng = col_range("breakout_readiness_pct")
        ws.conditional_format(rng, {
            "type": "cell", "criteria": ">=", "value": cfg.READINESS_BAND_HIGH,
            "format": fmts["green_bold"],
        })
        ws.conditional_format(rng, {
            "type": "cell", "criteria": "between",
            "minimum": cfg.READINESS_BAND_MEDIUM, "maximum": cfg.READINESS_BAND_HIGH - 0.01,
            "format": fmts["yellow_bg"],
        })

    # MTF confluence -> blue bold symbol
    if "mtf_confluence" in col_index and "symbol" in col_index:
        sym_idx, sym_rng = col_range("symbol")
        mtf_idx = col_index["mtf_confluence"]
        ws.conditional_format(sym_rng, {
            "type": "formula",
            "criteria": f"=${_xl_col(mtf_idx)}2=TRUE",
            "format": fmts["blue_bold"],
        })

    # Signal Type=BREAKOUT NOW + Volume Confirmed=Yes -> orange bold symbol
    if all(k in col_index for k in ("signal_type", "volume_confirmed_label", "symbol")):
        sym_idx, sym_rng = col_range("symbol")
        sig_idx = col_index["signal_type"]
        vol_idx = col_index["volume_confirmed_label"]
        ws.conditional_format(sym_rng, {
            "type": "formula",
            "criteria": (
                f'=AND(${_xl_col(sig_idx)}2="BREAKOUT NOW",'
                f'${_xl_col(vol_idx)}2="Yes")'
            ),
            "format": fmts["orange_bold"],
        })

    # Stop Loss Type = 8pct_cap -> red text
    if "stop_loss_type" in col_index:
        idx, rng = col_range("stop_loss_type")
        ws.conditional_format(rng, {
            "type": "cell", "criteria": "==", "value": '"8pct_cap"',
            "format": fmts["red_text"],
        })

    # R:R at T2 < 2.0 -> orange text
    if "rr_t2" in col_index:
        idx, rng = col_range("rr_t2")
        ws.conditional_format(rng, {
            "type": "cell", "criteria": "<", "value": cfg.MIN_RR_T2,
            "format": fmts["orange_bold"],
        })


def _apply_watchlist_conditional_formats(ws, fmts, cols_present, n_rows) -> None:
    if n_rows == 0:
        return
    col_index = {c[0]: i for i, c in enumerate(cols_present)}
    if "breakout_readiness_pct" in col_index:
        idx = col_index["breakout_readiness_pct"]
        rng = f"{_xl_col(idx)}2:{_xl_col(idx)}{n_rows + 1}"
        ws.conditional_format(rng, {
            "type": "cell", "criteria": ">=",
            "value": cfg.READINESS_BAND_HIGH,
            "format": fmts["green_bold"],
        })
        ws.conditional_format(rng, {
            "type": "cell", "criteria": "between",
            "minimum": cfg.READINESS_BAND_MEDIUM,
            "maximum": cfg.READINESS_BAND_HIGH - 0.01,
            "format": fmts["yellow_bg"],
        })


def _apply_confirmed_breakout_formats(ws, fmts, cols_present, n_rows) -> None:
    """
    Every row on the Confirmed Breakouts tab has already passed a strict
    multi-condition filter, so the formatting focuses on highlighting
    DEGREE of urgency rather than filtering — all rows are already worth
    looking at.
    """
    if n_rows == 0:
        return
    col_index = {c[0]: i for i, c in enumerate(cols_present)}

    def rng(name):
        if name not in col_index:
            return None
        return f"{_xl_col(col_index[name])}2:{_xl_col(col_index[name])}{n_rows + 1}"

    # BREAKOUT NOW → entire symbol cell bold orange (breakout happening now)
    if all(k in col_index for k in ("signal_type", "symbol")):
        sym_rng = rng("symbol")
        sig_idx = col_index["signal_type"]
        ws.conditional_format(sym_rng, {
            "type": "formula",
            "criteria": f'=${_xl_col(sig_idx)}2="BREAKOUT NOW"',
            "format": fmts["orange_bold"],
        })

    # NEAR BREAKOUT → symbol bold blue (act today or tomorrow)
    if all(k in col_index for k in ("signal_type", "symbol")):
        sym_rng = rng("symbol")
        sig_idx = col_index["signal_type"]
        ws.conditional_format(sym_rng, {
            "type": "formula",
            "criteria": f'=${_xl_col(sig_idx)}2="NEAR BREAKOUT"',
            "format": fmts["blue_bold"],
        })

    # Readiness % — green gradient: 100% deepest green, 80% lighter
    if r := rng("breakout_readiness_pct"):
        ws.conditional_format(r, {
            "type": "3_color_scale",
            "min_value": cfg.HCB_MIN_READINESS,
            "mid_value": 90,
            "max_value": 100,
            "min_color": "#FFEB9C",
            "mid_color": "#92D050",
            "max_color": "#006100",
            "min_type": "num",
            "mid_type": "num",
            "max_type": "num",
        })

    # Quality score banding
    if r := rng("quality_score"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": ">=", "value": 80,
            "format": fmts["green_bg"],
        })
        ws.conditional_format(r, {
            "type": "cell", "criteria": "between",
            "minimum": 60, "maximum": 79.99,
            "format": fmts["yellow_bg"],
        })

    # Volume confirmed → green, not confirmed → orange
    if r := rng("volume_confirmed_label"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": "==", "value": '"Yes"',
            "format": fmts["green_bg"],
        })
        ws.conditional_format(r, {
            "type": "cell", "criteria": "==", "value": '"No"',
            "format": fmts["orange_bold"],
        })

    # MTF confluence → blue bold
    if all(k in col_index for k in ("mtf_confluence", "symbol")):
        sym_rng = rng("symbol")
        mtf_idx = col_index["mtf_confluence"]
        # Only add if not already coloured by signal_type (MTF takes second priority)
        ws.conditional_format(rng("mtf_confluence"), {
            "type": "cell", "criteria": "==", "value": 1,
            "format": fmts["blue_bold"],
        })

    # Stop type = 8pct_cap → red on the stop type cell AND the whole row
    if r := rng("stop_loss_type"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": "==", "value": '"8pct_cap"',
            "format": fmts["red_text"],
        })
    # Highlight the entire row red when stop is capped so it's
    # impossible to miss while scanning down the list
    if "stop_loss_type" in col_index:
        stop_col_letter = _xl_col(col_index["stop_loss_type"])
        full_row_range = f"A2:{_xl_col(len(cols_present) - 1)}{n_rows + 1}"
        ws.conditional_format(full_row_range, {
            "type": "formula",
            "criteria": f'=${stop_col_letter}2="8pct_cap"',
            "format": fmts["stop_capped_row"],
        })

    # R:R T2 < 2 → orange warning
    if r := rng("rr_t2"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": "<", "value": cfg.MIN_RR_T2,
            "format": fmts["orange_bold"],
        })

    # Handle quality heat: 20 = best, 10 = minimum threshold
    if r := rng("handle_quality_subscore"):
        ws.conditional_format(r, {
            "type": "3_color_scale",
            "min_value": 10, "mid_value": 15, "max_value": 20,
            "min_color": "#FFEB9C",
            "mid_color": "#92D050",
            "max_color": "#006100",
            "min_type": "num", "mid_type": "num", "max_type": "num",
        })


def _apply_on_the_verge_formats(ws, fmts, cols_present, n_rows) -> None:
    """
    Lighter-weight formatting for the On The Verge tab — every row has
    already passed real gates (see get_on_the_verge_df), so like the
    Confirmed Breakouts tab this focuses on ranking rows by conviction
    rather than flagging rows to reject.
    """
    if n_rows == 0:
        return
    col_index = {c[0]: i for i, c in enumerate(cols_present)}

    def rng(name):
        if name not in col_index:
            return None
        return f"{_xl_col(col_index[name])}2:{_xl_col(col_index[name])}{n_rows + 1}"

    # Readiness % gradient — same treatment as Confirmed Breakouts but
    # anchored to the lower OTV floor (65%) instead of HCB's 80%
    if r := rng("breakout_readiness_pct"):
        ws.conditional_format(r, {
            "type": "3_color_scale",
            "min_value": cfg.OTV_MIN_READINESS,
            "mid_value": 82,
            "max_value": 100,
            "min_color": "#FFEB9C",
            "mid_color": "#92D050",
            "max_color": "#006100",
            "min_type": "num", "mid_type": "num", "max_type": "num",
        })

    # Quality score banding
    if r := rng("quality_score"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": ">=", "value": 70,
            "format": fmts["green_bg"],
        })
        ws.conditional_format(r, {
            "type": "cell", "criteria": "between",
            "minimum": cfg.OTV_MIN_QUALITY, "maximum": 69.99,
            "format": fmts["yellow_bg"],
        })

    # MTF confluence → blue bold symbol
    if all(k in col_index for k in ("mtf_confluence", "symbol")):
        ws.conditional_format(rng("mtf_confluence"), {
            "type": "cell", "criteria": "==", "value": 1,
            "format": fmts["blue_bold"],
        })

    # Stop type = 8pct_cap → red text + whole-row highlight, same
    # treatment as Confirmed Breakouts
    if r := rng("stop_loss_type"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": "==", "value": '"8pct_cap"',
            "format": fmts["red_text"],
        })
    if "stop_loss_type" in col_index:
        stop_col_letter = _xl_col(col_index["stop_loss_type"])
        full_row_range = f"A2:{_xl_col(len(cols_present) - 1)}{n_rows + 1}"
        ws.conditional_format(full_row_range, {
            "type": "formula",
            "criteria": f'=${stop_col_letter}2="8pct_cap"',
            "format": fmts["stop_capped_row"],
        })

    # R:R T2 < 2 → orange warning
    if r := rng("rr_t2"):
        ws.conditional_format(r, {
            "type": "cell", "criteria": "<", "value": cfg.MIN_RR_T2,
            "format": fmts["orange_bold"],
        })


def _xl_col(idx: int) -> str:
    """0-indexed column number -> Excel column letter."""
    letters = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


# ─── Historical summary block ──────────────────────────────────────────────

def _historical_summary_block(historical: pd.DataFrame) -> list[str]:
    if historical is None or historical.empty:
        return ["No historical signals yet."]

    total = len(historical)
    wins = historical[historical["status"].isin(
        ["Target 1 Achieved", "Target 2 Achieved", "Target 3 Achieved"]
    )]
    win_rate = (len(wins) / total * 100) if total else 0.0

    realised = historical["realised_rr"].dropna()
    avg_rr = realised.mean() if not realised.empty else 0.0

    gains = historical.loc[historical["realised_rr"] > 0, "realised_rr"].sum()
    losses = abs(historical.loc[historical["realised_rr"] < 0, "realised_rr"].sum())
    profit_factor = (gains / losses) if losses > 0 else float("inf")

    lines = [
        f"Total trades triggered: {total}",
        f"Win rate (hit T1+): {win_rate:.1f}%",
        f"Average R:R realised: {avg_rr:.2f}",
        f"Profit factor: {profit_factor:.2f}" if profit_factor != float("inf") else "Profit factor: N/A (no losses yet)",
    ]
    if "timeframe" in historical.columns:
        by_tf = historical["timeframe"].value_counts()
        lines.append("By timeframe: " + ", ".join(f"{k}={v}" for k, v in by_tf.items()))
    if "pattern_type" in historical.columns:
        by_pt = historical["pattern_type"].value_counts()
        lines.append("By pattern type: " + ", ".join(f"{k}={v}" for k, v in by_pt.items()))
    return lines


# ─── Strategy Summary sheet ─────────────────────────────────────────────────

def _write_strategy_summary(writer, workbook, fmts, stats: dict) -> None:
    ws = workbook.add_worksheet("Strategy Summary")
    writer.sheets["Strategy Summary"] = ws

    ws.set_column(0, 0, 36)
    ws.set_column(1, 1, 50)

    row = 0
    ws.write(row, 0, "NSE Cup & Handle Scanner — Strategy Summary", fmts["title"])
    row += 2

    ws.write(row, 0, "Scan Date", fmts["subtitle"])
    ws.write(row, 1, stats.get("scan_date", date.today().isoformat()))
    row += 1
    ws.write(row, 0, "Total Symbols Scanned")
    ws.write(row, 1, stats.get("total_symbols", 0))
    row += 1
    ws.write(row, 0, "Total Patterns Detected")
    ws.write(row, 1, stats.get("total_patterns", 0))
    row += 1
    ws.write(row, 0, "  ↳ 🔥 Confirmed Breakouts (act today)", fmts["subtitle"])
    ws.write(row, 1, stats.get("n_confirmed", 0))
    row += 1
    ws.write(row, 0, "  ↳ 🎯 On The Verge (high conviction, not triggered yet)", fmts["subtitle"])
    ws.write(row, 1, stats.get("n_on_the_verge", 0))
    row += 1
    ws.write(row, 0, "  ↳ Actionable (Today's Signals sheet)")
    ws.write(row, 1, stats.get("n_actionable", 0))
    row += 1
    ws.write(row, 0, "  ↳ Early Watch (Cup Only — no handle yet)")
    ws.write(row, 1, stats.get("n_early_watch", 0))
    row += 2

    ws.write(row, 0, "Breakdown by Timeframe", fmts["subtitle"])
    row += 1
    for tf, count in stats.get("by_timeframe", {}).items():
        ws.write(row, 0, f"  {tf.capitalize()}")
        ws.write(row, 1, count)
        row += 1
    row += 1

    ws.write(row, 0, "Breakdown by Signal Type", fmts["subtitle"])
    row += 1
    for st, count in stats.get("by_signal_type", {}).items():
        ws.write(row, 0, f"  {st}")
        ws.write(row, 1, count)
        row += 1
    row += 1

    ws.write(row, 0, "Breakdown by RS Rating Band", fmts["subtitle"])
    row += 1
    for band, count in stats.get("by_rs_band", {}).items():
        ws.write(row, 0, f"  {band}")
        ws.write(row, 1, count)
        row += 1
    row += 1

    ws.write(row, 0, "Top 5 Highest Breakout Readiness Today", fmts["subtitle"])
    row += 1
    top5 = stats.get("top5_readiness", [])
    if top5:
        ws.write(row, 0, "Symbol")
        ws.write(row, 1, "Readiness % / Timeframe / Signal Type")
        row += 1
        for item in top5:
            ws.write(row, 0, item.get("symbol", ""))
            ws.write(
                row, 1,
                f"{item.get('breakout_readiness_pct', 0):.0f}% "
                f"| {item.get('timeframe', '')} | {item.get('signal_type', '')}"
            )
            row += 1
    else:
        ws.write(row, 0, "No actionable (near-breakout) signals today.")
        row += 1
    row += 1

    ws.write(row, 0, "Market Condition", fmts["subtitle"])
    row += 1
    ws.write(row, 0, "  Nifty Trend")
    ws.write(row, 1, stats.get("nifty_trend", "Unknown"))
    row += 2

    ws.write(row, 0, "Configuration Used", fmts["subtitle"])
    row += 1
    config_lines = [
        ("Portfolio Value", f"₹{cfg.PORTFOLIO_VALUE:,.0f}"),
        ("Risk per Trade", f"{cfg.RISK_PER_TRADE_PCT}%"),
        ("Cup Min Recovery", f"{cfg.CUP_MIN_RECOVERY_PCT*100:.0f}%"),
        ("Max Stop Loss", f"{cfg.MAX_STOP_PCT*100:.0f}%"),
        ("Volume Confirm (Daily)", f"{cfg.VOLUME_CONFIRM_DAILY*100:.0f}% of 50-bar avg"),
        ("T1 Gain Target", f"{cfg.T1_GAIN_PCT*100:.0f}%"),
        ("Fibonacci Extension (T3)", f"{cfg.FIBONACCI_EXTENSION}x"),
        ("Min R:R at T2 (flag threshold)", f"{cfg.MIN_RR_T2}:1"),
    ]
    for label, val in config_lines:
        ws.write(row, 0, f"  {label}")
        ws.write(row, 1, val)
        row += 1
