"""
NSE Cup & Handle Scanner - Configuration
==========================================
All tuneable constants live here. Detection thresholds are
deliberately loose (maximum sensitivity); entry/exit thresholds are
deliberately strict (capital protection). See spec v3 Parts 3 & 4 for
the reasoning behind each value — do not tighten detection thresholds
or loosen entry/exit thresholds without re-reading that rationale.
"""

from __future__ import annotations

from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
DAILY_DIR     = DATA_DIR / "daily"
REPORTS_DIR   = Path("reports")
SIGNALS_DB    = DATA_DIR / "scanner.db"
LOGS_DIR      = Path("logs")

for _d in (DATA_DIR, DAILY_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Benchmarks ────────────────────────────────────────────────────────────
NIFTY50_SYMBOL   = "^NSEI"
NIFTY500_SYMBOL  = "^CNX500"

# ─── Portfolio / position sizing ──────────────────────────────────────────
PORTFOLIO_VALUE       = 500_000     # INR, configurable
RISK_PER_TRADE_PCT    = 1.0         # % of portfolio risked per trade

# ─── Data download ─────────────────────────────────────────────────────────
BATCH_SIZE                = 50
BATCH_DELAY_SECONDS       = 3
MAX_RETRIES                = 5
RATELIMIT_RETRY_WAIT_MIN   = 2      # minutes, base for exponential backoff
EXPONENTIAL_BASE           = 2.0
TIMEOUT_RETRY_WAIT_SEC     = 30

# ─── Minimum bars required to scan a symbol on each timeframe ────────────
MIN_DAILY_BARS    = 150
MIN_WEEKLY_BARS   = 40
MIN_MONTHLY_BARS  = 12

# ─── Cup detection — lookback window sizes (in bars) ──────────────────────
DAILY_CUP_MIN_BARS    = 30
DAILY_CUP_MAX_BARS    = 500
WEEKLY_CUP_MIN_BARS   = 8
WEEKLY_CUP_MAX_BARS   = 104
MONTHLY_CUP_MIN_BARS  = 5
MONTHLY_CUP_MAX_BARS  = 36

CUP_LOOKBACKS = {
    "daily":   (DAILY_CUP_MIN_BARS, DAILY_CUP_MAX_BARS),
    "weekly":  (WEEKLY_CUP_MIN_BARS, WEEKLY_CUP_MAX_BARS),
    "monthly": (MONTHLY_CUP_MIN_BARS, MONTHLY_CUP_MAX_BARS),
}

MIN_BARS_BY_TIMEFRAME = {
    "daily":   MIN_DAILY_BARS,
    "weekly":  MIN_WEEKLY_BARS,
    "monthly": MIN_MONTHLY_BARS,
}

# ─── Cup detection — the ONLY hard gate ───────────────────────────────────
CUP_MIN_RECOVERY_PCT  = 0.50   # right rim must recover >=50% of the decline

# Left rim finding
LEFT_RIM_WINDOW_FRACTION = 0.25   # search first 25% of window for left rim
LEFT_RIM_ROLLING_BARS    = 5      # rolling max window to find local peaks
LEFT_RIM_MIN_PCT_OF_HIGH = 0.90   # left rim must be >=90% of window's high

# Recency requirement — current price must still be reasonably close
# to the right rim (we want CURRENT setups, not historical/resolved ones)
CURRENT_PRICE_MIN_PCT_OF_RIM    = 0.80   # current price >= 80% of right rim

# ─── Handle detection (relaxed geometry) ──────────────────────────────────
HANDLE_MAX_DEPTH_RATIO       = 0.50   # handle depth <= 50% of cup depth
HANDLE_MIN_PCT_FROM_BOTTOM   = 0.30   # handle low >= 30% up from cup bottom
HANDLE_MAX_BREAKOUT_PCT      = 0.05   # handle high can't exceed right rim by >5%

HANDLE_SEARCH_BARS = {
    # (min_bars, max_bars) after the right rim to search for a handle
    "daily":   (3, 65),
    "weekly":  (1, 15),
    "monthly": (1, 6),
}

# ─── Cup depth classification bands (display only, never gates) ──────────
CUP_DEPTH_BANDS = [
    ("Shallow Cup",    0.0,  15.0),
    ("Classic Cup",    15.0, 35.0),
    ("Deep Cup",       35.0, 50.0),
    ("Very Deep Cup",  50.0, 70.0),
    ("Crash Recovery", 70.0, 9999.0),
]
CUP_VERY_SHALLOW_THRESHOLD = 5.0   # below this -> "verify manually" flag

# Same bands, pattern-neutral wording — used by pattern_common.classify_depth()
# for Double Bottom / VCP / Ascending Triangle so their depth_class never
# reads as e.g. "Classic Cup" on a triangle.
GENERIC_DEPTH_BANDS = [
    ("Shallow",         0.0,  15.0),
    ("Classic Depth",   15.0, 35.0),
    ("Deep",            35.0, 50.0),
    ("Very Deep",       50.0, 70.0),
    ("Crash Recovery",  70.0, 9999.0),
]

# ─── Prior uptrend tag bands (display only, never gates) ──────────────────
PRIOR_UPTREND_STRONG_PCT    = 30.0   # >= this -> "Strong Continuation"
PRIOR_UPTREND_MODERATE_PCT  = 10.0   # >= this -> "Moderate Continuation"
# below PRIOR_UPTREND_MODERATE_PCT -> "Possible Bottoming Pattern"

# ─── Quality score weights ─────────────────────────────────────────────────
QS_RIM_SYMMETRY      = {"high": 25, "mid": 15, "low": 5}     # right/left rim ratio
QS_CUP_DEPTH         = {"classic": 20, "near": 10, "far": 5, "crash": 2}
QS_PRIOR_UPTREND     = {"strong": 15, "moderate": 8, "weak": 3}
QS_VOLUME_DRYUP      = {"full": 10, "partial": 5, "none": 0}
QS_SHAPE             = {"u_shape": 10, "v_shape": 5, "irregular": 0}
QS_RS_RATING         = {"leader": 10, "rising": 5, "lagging": 0}
QS_MTF_CONFLUENCE    = 10
QS_HANDLE_MAX        = 20   # handle quality sub-score, added on top
QUALITY_SCORE_CAP    = 100

QUALITY_BAND_HIGH    = 70   # >= this -> "High Quality" (green)
QUALITY_BAND_MEDIUM  = 40   # >= this -> "Medium Quality" (yellow)
# below QUALITY_BAND_MEDIUM -> "Low Quality" (white)

# ─── Handle quality sub-score thresholds ───────────────────────────────────
HQ_VOL_DRYUP_FULL_RATIO     = 0.70   # handle avg vol <= 70% of cup avg -> 6pts
HQ_VOL_DRYUP_PARTIAL_RATIO  = 1.00   # <= 100% -> 3pts, else 0
HQ_VOL_DRYUP_PTS            = {"full": 6, "partial": 3, "none": 0}

HQ_TIGHTNESS_FULL_RATIO     = 0.60   # handle range ratio <= 60% of cup -> 5pts
HQ_TIGHTNESS_PARTIAL_RATIO  = 0.90   # <= 90% -> 3pts, else 0
HQ_TIGHTNESS_PTS            = {"full": 5, "partial": 3, "none": 0}

HQ_HIGHER_LOWS_PTS          = {"sequential": 5, "second_half": 2, "none": 0}

HQ_CLOSE_HIGH_FULL          = 0.65   # avg close position >= 0.65 -> 4pts
HQ_CLOSE_HIGH_PARTIAL       = 0.45   # >= 0.45 -> 2pts, else 0
HQ_CLOSE_HIGH_PTS           = {"full": 4, "partial": 2, "none": 0}

# ─── Signal type thresholds ─────────────────────────────────────────────────
NEAR_BREAKOUT_THRESHOLD = 0.03   # within 3% below pivot
BASING_THRESHOLD        = 0.10   # 3-10% below pivot
# >10% below pivot -> EARLY STAGE
EXTENDED_THRESHOLD      = 0.05   # >5% above pivot -> EXTENDED

# ─── Breakout Readiness weights (Part 3I) ──────────────────────────────────
READINESS_WEIGHTS = {
    "near_pivot":      25,
    "tight_handle":    20,
    "rising_rs":       20,
    "atr_contracting": 20,
    "above_50ma":      15,
}
READINESS_NEAR_PIVOT_PCT = 3.0    # within 3% of pivot, either side
READINESS_BAND_HIGH      = 80     # >= this -> bold green
READINESS_BAND_MEDIUM    = 50     # >= this -> yellow background

# ─── High-Conviction Breakout filter ("Confirmed Breakouts" tab) ─────────
# ALL conditions must be met simultaneously for a signal to appear on
# this tab. The idea is to produce a very short, genuinely actionable
# list — typically 5-30 stocks on any given day. Do NOT loosen these
# to inflate the list; the value of this tab is its selectivity.
HCB_SIGNAL_TYPES          = {"BREAKOUT NOW", "NEAR BREAKOUT"}
HCB_MIN_READINESS         = 80     # Breakout Readiness >= 80% (4-5 factors)
HCB_MIN_QUALITY           = 60     # pattern must be structurally solid
HCB_MIN_VOLUME_RATIO      = 1.40   # volume >= 140% of avg (same as entry gate)
HCB_MAX_PRICE_VS_PIVOT    = 5.0    # not more than 5% above pivot (not extended)
HCB_MIN_PRICE_VS_PIVOT    = -3.0   # not more than 3% below pivot (within zone)
HCB_MIN_HANDLE_QUALITY    = 10     # handle must be tight (>= 10/20)
HCB_REQUIRE_HANDLE        = True   # Cup Only excluded — no handle, no certainty

# ─── "On The Verge" filter — a wider, second-tier confirmed list ─────────
# Sits between 🔥 Confirmed Breakouts (very few, near-certain) and the
# full Near Breakout Watchlist (hundreds, unfiltered by conviction).
# Every condition below is still a REAL gate — this is not just "show
# more rows" — but each threshold is relaxed relative to HCB so a
# larger, still-credible set of names qualifies. Typically produces
# 20-80 stocks on a given day versus HCB's 2-10.
#
# Key differences from HCB:
#   - Only NEAR BREAKOUT (excludes BREAKOUT NOW, which already belongs
#     on the Confirmed tab, and excludes BASING, which is too early to
#     call "on the verge")
#   - Volume gate lowered to 1.0x (can't demand breakout-day volume
#     from a stock that hasn't broken out yet) instead of dropped
#     entirely — some volume support is still required, just not surge
#   - Readiness, quality, and handle quality floors relaxed but not
#     removed
OTV_SIGNAL_TYPES          = {"NEAR BREAKOUT"}
OTV_MIN_READINESS         = 65     # relaxed from HCB's 80
OTV_MIN_QUALITY           = 50     # relaxed from HCB's 60
OTV_MIN_VOLUME_RATIO      = 1.00   # relaxed from HCB's 1.40 — average volume, not surge
OTV_MAX_PRICE_VS_PIVOT    = 0.0    # must not have crossed pivot yet (else it's a breakout, not "on the verge")
OTV_MIN_PRICE_VS_PIVOT    = -5.0   # relaxed from HCB's -3.0
OTV_MIN_HANDLE_QUALITY    = 6      # relaxed from HCB's 10
OTV_REQUIRE_HANDLE        = True   # still require a handle — Cup Only is never "certain"


BREAKOUT_BUFFER_INR        = 0.10
BREAKOUT_BUFFER_PCT        = 0.001   # used instead of flat INR above ₹1000
BREAKOUT_BUFFER_PRICE_CUTOFF = 1000.0

VOLUME_CONFIRM_DAILY   = 1.40    # 140% of 50-bar avg volume
VOLUME_CONFIRM_WEEKLY  = 1.20    # 120% of 10-bar avg volume (weekly/monthly)
VOLUME_AVG_BARS_DAILY   = 50
VOLUME_AVG_BARS_WEEKLY  = 10

RSI_WEAK_MOMENTUM_THRESHOLD = 45.0

# ─── Stop loss (STRICT) ─────────────────────────────────────────────────────
MAX_STOP_PCT          = 0.08    # 8% hard cap, never exceeded
STOP_ATR_MULTIPLIER   = 1.0
CUP_ONLY_STOP_RECOVERY_FRACTION = 0.20   # cup_bottom + 20% of cup height

# ─── Targets (O'Neil / CANSLIM) ─────────────────────────────────────────────
T1_GAIN_PCT          = 0.20    # +20% first profit target
T1_FAST_DAYS         = 15      # if hit in < this many trading days -> hold rule
FIBONACCI_EXTENSION  = 1.618
MIN_RR_T2            = 2.0     # flag (not exclude) if below this

TRAILING_STOP_PCT    = 0.07    # 7% trailing stop from post-entry closing high
STUCK_BASE_WEEKS      = 8       # no new high in N weeks -> flag

# ─── RS Rating ───────────────────────────────────────────────────────────────
RS_LEADER_THRESHOLD   = 85
RS_RISING_THRESHOLD   = 70
RS_LAGGARD_THRESHOLD  = 50
RS_TREND_LOOKBACK_WEEKS = 4

# RS weighted-return blend: (weight, lookback_bars) computed in indicators.py
RS_WEIGHTS = {
    "3m":  (0.4, 63),    # ~63 trading days = 3 months
    "6m":  (0.2, 126),
    "9m":  (0.2, 189),
    "12m": (0.2, 252),
}

# ─── Liquidity filter (applied AFTER detection, gates entry/exit only) ────
MIN_PRICE          = 10.0
MIN_AVG_VOLUME     = 50_000
LIQUIDITY_LOOKBACK_BARS = 20

# ─── Pattern recency (skip stale detections) ──────────────────────────────
STALE_PATTERN_MAX_BARS = {
    "daily":   60,
    "weekly":  20,
    "monthly": 12,
}

# ─── Watchlist / signal expiry ─────────────────────────────────────────────
WATCHLIST_EXPIRY_DAYS = 90

# ─── Indicator periods ──────────────────────────────────────────────────────
RSI_PERIOD   = 14
ADX_PERIOD   = 14
ATR_PERIOD   = 14
MA_PERIODS_DAILY   = [50, 150, 200]
MA_PERIODS_WEEKLY  = [10, 30, 40]     # roughly equivalent to 50/150/200 daily
MA_PERIODS_MONTHLY = [3, 7, 9]        # roughly equivalent on monthly bars

# ─── HTML chart dashboard (chart_export, baked into main.py) ──────────────
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CHART_LOOKBACK_DAILY_BARS   = 252    # ~1 trading year
CHART_LOOKBACK_WEEKLY_BARS  = 260    # ~5 years (52 weeks x 5)
CHART_LOOKBACK_MONTHLY_BARS = 9999   # effectively unlimited — full available history


# ═════════════════════════════════════════════════════════════════════════
# MULTI-PATTERN EXTENSION (v4) — Double Bottom, VCP, Ascending Triangle
# ═════════════════════════════════════════════════════════════════════════
# Same philosophy as Cup & Handle above: detection is loose (one hard
# gate per pattern), everything else is scored/tagged. All three new
# detectors reuse readiness.py and entry_exit.py UNCHANGED by emitting
# a PatternSignal (see pattern_common.py) that exposes the same
# attribute surface as CupHandleSignal (has_handle / handle_low_price /
# left_rim_price / cup_bottom_price / pivot_point / etc.) — the names
# are legacy from Cup & Handle but the *concepts* generalise cleanly:
# "handle" = final tight base right before the pivot, "left_rim/cup_
# bottom" = the two reference price points used for the measured-move
# target. See pattern_common.py's module docstring for the full mapping.

# Generic label used by Double Bottom / VCP / Ascending Triangle in
# place of Cup & Handle's pattern-specific "CUP ONLY" for a structure
# that's found but has no final base/handle confirmed yet.
GENERIC_PRECONFIRM_SIGNAL_TYPE = "BASE FORMING"

# ─── Double Bottom ("W" shape) ─────────────────────────────────────────────
DB_LOOKBACKS = {                       # (min_bars, max_bars) search window
    "daily":   (20, 300),
    "weekly":  (6, 80),
    "monthly": (4, 30),
}
DB_MIN_RECOVERY_PCT              = 0.50   # HARD GATE: recovery off bottom-2 toward the neckline
DB_BOTTOM1_ZONE_FRACTION         = 0.50   # search first 50% of window for bottom-1
DB_MIN_BOUNCE_FROM_BOTTOM1_PCT   = 0.06   # the middle peak must bounce >=6% off bottom-1
DB_BOTTOM_SIMILARITY_PCT         = 0.08   # bottom-2 must be within 8% of bottom-1's price
DB_MAX_UNDERCUT_PCT              = 0.10   # bottom-2 may undercut bottom-1 by up to 10% (shakeout)
DB_EVEN_TOLERANCE_PCT            = 0.02   # within 2% -> "Even", else Ascending/Descending
DB_CURRENT_PRICE_MIN_PCT_OF_NECKLINE = 0.80  # recency guard, mirrors cup&handle's rim guard

DB_HANDLE_SEARCH_BARS = {              # final base search window after bottom-2's recovery
    "daily": (2, 45), "weekly": (1, 12), "monthly": (1, 5),
}
DB_HANDLE_MAX_DEPTH_RATIO     = 0.50
DB_HANDLE_MIN_PCT_FROM_BOTTOM = 0.30
DB_HANDLE_MAX_BREAKOUT_PCT    = 0.05

# ─── VCP (Volatility Contraction Pattern, Minervini-style) ────────────────
VCP_LOOKBACKS = {
    "daily": (25, 300), "weekly": (8, 90), "monthly": (5, 30),
}
VCP_ZIGZAG_THRESHOLD = {"daily": 0.045, "weekly": 0.06, "monthly": 0.08}
VCP_MIN_CONTRACTIONS            = 2       # HARD GATE: at least 2 progressively tighter pullbacks
VCP_MAX_CONTRACTIONS_CONSIDERED = 5
VCP_CONTRACTION_SLACK           = 0.85    # each contraction must be <= prev depth * slack
VCP_LOW_UNDERCUT_TOLERANCE      = 0.02    # allow a 2% undercut and still call it a "higher low"
VCP_CURRENT_PRICE_MIN_PCT_OF_PIVOT = 0.75
VCP_TEXTBOOK_MIN_CONTRACTIONS   = 3       # >=3 clean contractions -> "textbook" shape bucket

# ─── Ascending Triangle ─────────────────────────────────────────────────
AT_LOOKBACKS = {
    "daily": (20, 260), "weekly": (8, 80), "monthly": (5, 26),
}
AT_ZIGZAG_THRESHOLD          = {"daily": 0.035, "weekly": 0.05, "monthly": 0.07}
AT_MIN_TOUCHES_RESISTANCE    = 2          # HARD GATE: >=2 highs clustering at the flat top
AT_MIN_TOUCHES_SUPPORT       = 2          # HARD GATE: >=2 rising lows beneath it
AT_RESISTANCE_TOLERANCE_PCT  = 0.025      # highs within 2.5% of each other count as "flat"
AT_MIN_LOW_RISE_PCT          = 0.01       # each successive low must rise by >=1%
AT_LOW_UNDERCUT_TOLERANCE    = 0.015      # allow a 1.5% undercut on the rising-lows check
AT_CURRENT_PRICE_MIN_PCT_OF_PIVOT = 0.80
AT_TIGHT_FIT_TOLERANCE_PCT   = 0.012      # resistance touches this tight -> "Textbook" shape

# ─── Bullish confirmation candles (breakout confirmation) ─────────────────
# Applied to the most recent 1-3 bars of any signal at or crossing its
# pivot, to flag whether the breakout itself looks like genuine bullish
# conviction (per classic candlestick theory) rather than a thin, weak
# push through the level. Informational — never gates detection.
CANDLE_MARUBOZU_WICK_MAX_PCT       = 0.08   # each wick <= 8% of the bar's range
CANDLE_DOJI_BODY_MAX_PCT           = 0.08   # body <= 8% of range -> doji-like
CANDLE_HAMMER_LOWER_WICK_MIN_RATIO = 2.0    # lower wick >= 2x body
CANDLE_HAMMER_UPPER_WICK_MAX_RATIO = 0.35   # upper wick <= 35% of body
CANDLE_SOLID_BODY_MIN_PCT          = 0.55   # body >= 55% of range -> "solid" bullish candle
CANDLE_SOLID_CLOSE_POS_MIN         = 0.70   # close in the top 30% of the range -> "closed strong"
CANDLE_ENGULF_SIZE_MIN_RATIO       = 1.05   # engulfing body >= 1.05x the engulfed body -> "Strong"
CANDLE_LOOKBACK_FOR_CONFIRMATION   = 3      # bars examined around the breakout bar

