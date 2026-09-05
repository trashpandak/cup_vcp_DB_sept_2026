"""
NSE Cup & Handle Scanner - Database Layer
============================================
SQLite-backed persistence for:
  - cup_handle_signals : every signal ever detected, full geometry +
                          entry/exit + readiness fields (spec Part 8)
  - active_tracking is derived from cup_handle_signals.status, not a
    separate table — simpler schema, fewer places for state to drift

Uses parameterised queries throughout; never f-strings into SQL values.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import SIGNALS_DB
from logger_utils import get_logger

log = get_logger("scanner")

DB_PATH = SIGNALS_DB

DDL = """
CREATE TABLE IF NOT EXISTS cup_handle_signals (
    signal_id               TEXT PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    company_name             TEXT,
    sector                   TEXT,
    scan_date                TEXT NOT NULL,
    timeframe                TEXT NOT NULL,
    pattern_type              TEXT,
    pattern_family             TEXT DEFAULT 'cup_handle',
    signal_type               TEXT,
    quality_score              REAL,
    mtf_confluence             INTEGER DEFAULT 0,
    mtf_timeframes             TEXT,
    cross_pattern_confluence   INTEGER DEFAULT 0,
    cross_pattern_desc         TEXT,

    cup_start_date             TEXT,
    cup_bottom_date            TEXT,
    cup_end_date               TEXT,
    left_rim_price             REAL,
    cup_bottom_price           REAL,
    right_rim_price            REAL,
    cup_depth_pct              REAL,
    cup_depth_class            TEXT,
    cup_depth_verify_flag      INTEGER DEFAULT 0,
    cup_duration_bars          INTEGER,
    cup_shape                  TEXT,
    recovery_pct               REAL,
    prior_uptrend_pct          REAL,
    prior_uptrend_tag          TEXT,

    has_handle                 INTEGER DEFAULT 0,
    handle_start_date          TEXT,
    handle_end_date            TEXT,
    handle_low_price           REAL,
    handle_depth_pct           REAL,
    handle_duration_bars       INTEGER,
    handle_quality_subscore    REAL,
    handle_vol_dryup_pts       REAL,
    handle_tightness_pts       REAL,
    handle_higher_lows_pts     REAL,
    handle_close_high_pts      REAL,
    atr_at_handle_start        REAL,

    pivot_point                REAL,
    current_price               REAL,
    price_vs_pivot_pct          REAL,

    breakout_readiness_pct      REAL,
    readiness_near_pivot        INTEGER DEFAULT 0,
    readiness_tight_handle      INTEGER DEFAULT 0,
    readiness_rising_rs         INTEGER DEFAULT 0,
    readiness_atr_contract      INTEGER DEFAULT 0,
    readiness_above_50ma        INTEGER DEFAULT 0,
    readiness_reasons           TEXT,

    entry_price                 REAL,
    entry_zone_high              REAL,
    entry_type                   TEXT,
    extended_warning              TEXT,

    stop_loss_price               REAL,
    stop_loss_pct                 REAL,
    stop_loss_type                 TEXT,
    atr_14                         REAL,
    risk_per_share                  REAL,

    target1                         REAL,
    target2                         REAL,
    target3                         REAL,
    rr_t1                           REAL,
    rr_t2                           REAL,
    rr_t3                           REAL,
    rr_t2_warning                    TEXT,
    eight_week_hold_candidate        INTEGER DEFAULT 0,

    position_size_shares              INTEGER,
    capital_required                   REAL,
    risk_amount                        REAL,
    portfolio_risk_pct                 REAL,

    volume_ratio                       REAL,
    volume_confirmed                    INTEGER DEFAULT 0,
    volume_confirmed_label               TEXT,

    rs_rating                            REAL,
    rs_trend                              TEXT,
    rs_tag                                TEXT,
    rsi_val                               REAL,
    adx_val                               REAL,
    ma_short                              REAL,
    ma_mid                                REAL,
    ma_long                               REAL,
    price_vs_ma_short_pct                 REAL,

    nifty_trend                           TEXT,
    market_note                           TEXT,
    weak_momentum_note                    TEXT,
    below_50ma_note                       TEXT,

    liquidity_ok                          INTEGER DEFAULT 1,
    liquidity_warning                     TEXT,

    sell_notes                            TEXT,
    remarks                               TEXT,

    key_points_json                       TEXT,
    trendlines_json                       TEXT,
    structure_note                        TEXT,
    confirmation_candle_found             INTEGER DEFAULT 0,
    confirmation_candle_pattern           TEXT,
    confirmation_candle_strength          TEXT,
    confirmation_candle_date              TEXT,
    confirmation_candle_detail            TEXT,

    status                                TEXT DEFAULT 'Watching',
    entry_triggered                       INTEGER DEFAULT 0,
    entry_date                            TEXT,
    t1_achieved                           INTEGER DEFAULT 0,
    t2_achieved                           INTEGER DEFAULT 0,
    t3_achieved                           INTEGER DEFAULT 0,
    stopped_out                           INTEGER DEFAULT 0,
    exit_date                             TEXT,
    exit_price                            REAL,
    exit_type                             TEXT,
    realised_rr                            REAL,
    hold_days                              INTEGER,

    expiry_date                            TEXT,
    created_at                             TEXT DEFAULT (datetime('now')),
    last_checked                           TEXT
);

CREATE INDEX IF NOT EXISTS idx_ch_status ON cup_handle_signals(status);
CREATE INDEX IF NOT EXISTS idx_ch_symbol ON cup_handle_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_ch_scan_date ON cup_handle_signals(scan_date);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(DDL)
    _migrate_schema()
    log.info("Database initialised: %s", DB_PATH)


# Columns added by the multi-pattern extension (v4). Listed explicitly
# (rather than diffing against the DDL string) so an existing database
# created by an older version of this scanner gets upgraded in place —
# `CREATE TABLE IF NOT EXISTS` is a no-op once the table already
# exists, so new columns only ever reach it via ALTER TABLE here.
_V4_NEW_COLUMNS: dict[str, str] = {
    "pattern_family": "TEXT DEFAULT 'cup_handle'",
    "cross_pattern_confluence": "INTEGER DEFAULT 0",
    "cross_pattern_desc": "TEXT",
    "key_points_json": "TEXT",
    "trendlines_json": "TEXT",
    "structure_note": "TEXT",
    "confirmation_candle_found": "INTEGER DEFAULT 0",
    "confirmation_candle_pattern": "TEXT",
    "confirmation_candle_strength": "TEXT",
    "confirmation_candle_date": "TEXT",
    "confirmation_candle_detail": "TEXT",
}


def _migrate_schema() -> None:
    with _conn() as con:
        existing = {r[1] for r in con.execute("PRAGMA table_info(cup_handle_signals)").fetchall()}
        added = []
        for col, decl in _V4_NEW_COLUMNS.items():
            if col not in existing:
                con.execute(f"ALTER TABLE cup_handle_signals ADD COLUMN {col} {decl}")
                added.append(col)
        if added:
            # Backfill pattern_family for pre-existing rows (they're all
            # Cup & Handle — that's the only pattern that existed before).
            if "pattern_family" in added:
                con.execute(
                    "UPDATE cup_handle_signals SET pattern_family = 'cup_handle' "
                    "WHERE pattern_family IS NULL"
                )
            log.info("Schema migration: added columns %s", ", ".join(added))


# ─── Signal ID generation ──────────────────────────────────────────────────

def make_signal_id(symbol: str, timeframe: str, cup_bottom_date, pivot_point: float = None) -> str:
    """
    Deterministic ID for the SAME underlying cup across daily re-scans.

    Anchored on cup_bottom_date rather than cup_start_date or
    pivot_point: the cup bottom is the most stable point of the whole
    pattern — once that low has printed, it never moves. The left rim
    (cup_start_date) can shift by a day or two as new bars extend the
    detection window, and the pivot moves as the handle forms and
    evolves, so anchoring on either of those caused the same real-world
    pattern to be treated as a brand-new signal on every run, silently
    accumulating duplicate rows in the watchlist and active tracking.

    pivot_point is accepted for backward compatibility with existing
    call sites but is no longer used in the hash.
    """
    raw = f"{symbol}|{timeframe}|{cup_bottom_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def signal_exists(signal_id: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM cup_handle_signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
    return row is not None


# ─── CRUD ───────────────────────────────────────────────────────────────────

def upsert_cup_handle_signal(row: dict) -> None:
    """
    Insert a new signal, or refresh an existing one (same symbol +
    timeframe + cup_bottom_date, per make_signal_id) with today's
    freshly detected geometry, pivot, price, readiness, and quality.

    Tracking-state fields are deliberately NOT overwritten on conflict
    — a signal that already triggered, hit a target, or stopped out
    keeps that history even if it's re-detected in a later scan. Only
    detection-derived fields (everything describing the pattern itself
    and its current price/readiness) get refreshed.
    """
    # Fields that represent live trading state — never reset these on
    # a re-detection of the same underlying cup.
    preserve_on_conflict = {
        "status", "entry_triggered", "entry_date",
        "t1_achieved", "t2_achieved", "t3_achieved", "stopped_out",
        "exit_date", "exit_price", "exit_type", "realised_rr", "hold_days",
        "created_at",
    }

    update_cols = [k for k in row.keys() if k not in preserve_on_conflict and k != "signal_id"]

    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    update_clause = ", ".join(f"{k} = excluded.{k}" for k in update_cols)

    sql = (
        f"INSERT INTO cup_handle_signals ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(signal_id) DO UPDATE SET {update_clause}"
    )
    with _conn() as con:
        con.execute(sql, row)


def update_signal_status(signal_id: str, **kwargs) -> None:
    if not kwargs:
        return
    sets = ", ".join(f"{k} = :{k}" for k in kwargs)
    kwargs["signal_id"] = signal_id
    kwargs["last_checked"] = datetime.now().isoformat()
    with _conn() as con:
        con.execute(
            f"UPDATE cup_handle_signals SET {sets}, last_checked = :last_checked "
            f"WHERE signal_id = :signal_id",
            kwargs,
        )


def get_open_signals() -> list[dict]:
    """Signals that have triggered (entry_triggered=1) but not yet hit
    a final target or stop — i.e. currently being tracked."""
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM cup_handle_signals
               WHERE entry_triggered = 1
                 AND stopped_out = 0
                 AND t3_achieved = 0
                 AND status NOT IN ('Stopped Out', 'Target 3 Achieved', 'Expired')
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_watching_signals() -> list[dict]:
    """Signals not yet triggered — the watchlist."""
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM cup_handle_signals
               WHERE entry_triggered = 0
                 AND status NOT IN ('Expired', 'Stopped Out')
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_todays_signals_df(scan_date: Optional[str] = None) -> pd.DataFrame:
    """Today's signals excluding pre-confirmation-only structures (Cup
    Only for Cup & Handle, Base Forming for the other 3 patterns) —
    the main actionable sheet."""
    scan_date = scan_date or date.today().isoformat()
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE scan_date = ?
                 AND signal_type NOT IN ('CUP ONLY', 'BASE FORMING')
               ORDER BY mtf_confluence DESC, quality_score DESC""",
            con, params=(scan_date,),
        )


def get_todays_early_watch_df(scan_date: Optional[str] = None) -> pd.DataFrame:
    """Today's pre-confirmation signals (Cup Only / Base Forming) —
    earlier stage, own sheet."""
    scan_date = scan_date or date.today().isoformat()
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE scan_date = ?
                 AND signal_type IN ('CUP ONLY', 'BASE FORMING')
               ORDER BY quality_score DESC""",
            con, params=(scan_date,),
        )


def get_all_signals_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM cup_handle_signals ORDER BY scan_date DESC", con
        )


def get_active_tracking_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE entry_triggered = 1
                 AND stopped_out = 0
                 AND t3_achieved = 0
                 AND status NOT IN ('Stopped Out', 'Target 3 Achieved', 'Expired')
               ORDER BY scan_date DESC
            """, con,
        )


def get_historical_signals_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE status IN ('Stopped Out', 'Target 1 Achieved',
                                 'Target 2 Achieved', 'Target 3 Achieved', 'Expired')
               ORDER BY scan_date DESC
            """, con,
        )


def get_near_breakout_watchlist_df(scan_date: Optional[str] = None) -> pd.DataFrame:
    """
    Signals in NEAR BREAKOUT or BASING state, sorted by readiness.
    Scoped to a single scan_date (defaults to today) — without this,
    the same underlying pattern re-detected on consecutive days (with
    a slightly different pivot/cup_start_date, hence a new signal_id)
    would accumulate indefinitely instead of being treated as an
    update to the same setup.
    """
    scan_date = scan_date or date.today().isoformat()
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE scan_date = ?
                 AND signal_type IN ('NEAR BREAKOUT', 'BASING')
                 AND entry_triggered = 0
               ORDER BY breakout_readiness_pct DESC,
                        quality_score DESC
            """, con, params=(scan_date,),
        )


def get_confirmed_breakouts_df(scan_date: Optional[str] = None) -> pd.DataFrame:
    """
    High-conviction breakout signals — ALL of the following must be true:
      • Final base confirmed (has_handle = 1) — the handle for Cup &
        Handle, the confirmation base for Double Bottom, or the final
        contraction/approach for VCP / Ascending Triangle (always true
        for those two once detected at all)
      • Signal type in BREAKOUT NOW / NEAR BREAKOUT
      • Breakout Readiness >= 80%
      • Quality Score >= 60
      • Volume ratio >= 1.40  (volume confirmation)
      • Price within -3% to +5% of pivot  (not extended, not too far below)
      • Final base quality subscore >= 10/20  (tight base confirmed)
    Sorted by: MTF confluence first, then Readiness % DESC, then Quality DESC.
    Spans all 4 pattern types — pattern_family distinguishes them.
    """
    import config as cfg
    scan_date = scan_date or date.today().isoformat()
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE scan_date = ?
                 AND has_handle = 1
                 AND signal_type IN ('BREAKOUT NOW', 'NEAR BREAKOUT')
                 AND breakout_readiness_pct >= ?
                 AND quality_score >= ?
                 AND volume_ratio >= ?
                 AND price_vs_pivot_pct >= ?
                 AND price_vs_pivot_pct <= ?
                 AND handle_quality_subscore >= ?
               ORDER BY
                 mtf_confluence DESC,
                 breakout_readiness_pct DESC,
                 quality_score DESC""",
            con,
            params=(
                scan_date,
                cfg.HCB_MIN_READINESS,
                cfg.HCB_MIN_QUALITY,
                cfg.HCB_MIN_VOLUME_RATIO,
                cfg.HCB_MIN_PRICE_VS_PIVOT,
                cfg.HCB_MAX_PRICE_VS_PIVOT,
                cfg.HCB_MIN_HANDLE_QUALITY,
            ),
        )


def get_on_the_verge_df(scan_date: Optional[str] = None) -> pd.DataFrame:
    """
    "On The Verge" — a wider, second-tier confirmed list, sitting
    between 🔥 Confirmed Breakouts (very few, near-certain, already
    breaking out or crossing today) and the full Near Breakout
    Watchlist (hundreds, unfiltered by conviction).

    ALL of the following must be true:
      • Final base confirmed (has_handle = 1) — Base Forming / Cup Only
        stages never count as "on the verge" since there's no
        confirmation the supply has dried up yet
      • Signal type = NEAR BREAKOUT only (excludes BREAKOUT NOW, which
        belongs on Confirmed Breakouts; excludes BASING, too early)
      • Breakout Readiness >= 65%  (relaxed from HCB's 80%)
      • Quality Score >= 50        (relaxed from HCB's 60)
      • Volume ratio >= 1.00       (average volume support, not surge —
        relaxed from HCB's 1.40 since the breakout hasn't happened yet)
      • Price between -5% and 0% of pivot (below pivot but close — if
        price were above pivot it would already be a breakout)
      • Handle quality subscore >= 6/20   (relaxed from HCB's 10)

    Sorted by: Readiness % DESC, then Quality Score DESC.
    """
    import config as cfg
    scan_date = scan_date or date.today().isoformat()
    with _conn() as con:
        return pd.read_sql(
            """SELECT * FROM cup_handle_signals
               WHERE scan_date = ?
                 AND has_handle = 1
                 AND signal_type IN ('NEAR BREAKOUT')
                 AND breakout_readiness_pct >= ?
                 AND quality_score >= ?
                 AND volume_ratio >= ?
                 AND price_vs_pivot_pct >= ?
                 AND price_vs_pivot_pct <= ?
                 AND handle_quality_subscore >= ?
               ORDER BY
                 breakout_readiness_pct DESC,
                 quality_score DESC""",
            con,
            params=(
                scan_date,
                cfg.OTV_MIN_READINESS,
                cfg.OTV_MIN_QUALITY,
                cfg.OTV_MIN_VOLUME_RATIO,
                cfg.OTV_MIN_PRICE_VS_PIVOT,
                cfg.OTV_MAX_PRICE_VS_PIVOT,
                cfg.OTV_MIN_HANDLE_QUALITY,
            ),
        )


def dedupe_legacy_duplicate_signals() -> int:
    """
    One-time migration cleanup for databases created before signal_id
    was anchored on cup_bottom_date. Old rows for the same underlying
    pattern (same symbol + timeframe + cup_bottom_date) but with
    different legacy signal_ids are collapsed into one: keep the row
    with the most recent scan_date (freshest data), delete the rest —
    unless a duplicate is the one carrying live tracking state
    (entry_triggered=1 or any target/stop hit), in which case that one
    is kept instead so open-position history isn't lost.
    Returns the number of rows deleted.
    """
    with _conn() as con:
        groups = con.execute(
            """SELECT symbol, timeframe, cup_bottom_date, COUNT(*) as cnt
               FROM cup_handle_signals
               GROUP BY symbol, timeframe, cup_bottom_date
               HAVING cnt > 1"""
        ).fetchall()

        deleted = 0
        for g in groups:
            rows = con.execute(
                """SELECT signal_id, scan_date, entry_triggered,
                          t1_achieved, t2_achieved, t3_achieved, stopped_out
                   FROM cup_handle_signals
                   WHERE symbol = ? AND timeframe = ? AND cup_bottom_date = ?
                   ORDER BY scan_date DESC""",
                (g["symbol"], g["timeframe"], g["cup_bottom_date"]),
            ).fetchall()

            keeper = next(
                (r for r in rows if r["entry_triggered"] or r["t1_achieved"]
                 or r["t2_achieved"] or r["t3_achieved"] or r["stopped_out"]),
                rows[0],
            )
            for r in rows:
                if r["signal_id"] != keeper["signal_id"]:
                    con.execute(
                        "DELETE FROM cup_handle_signals WHERE signal_id = ?",
                        (r["signal_id"],),
                    )
                    deleted += 1

        return deleted


def cleanup_bad_triggers(valid_scan_date: str) -> int:
    """
    On the first run, signals in non-BREAKOUT-NOW states were incorrectly
    auto-triggered. Reset them back to 'Watching' so Active Tracking
    only shows genuine triggered positions.

    Keeps triggered if:
      - signal_type was 'BREAKOUT NOW' on the day it was first detected
      - OR the signal has already hit a target (t1/t2/t3_achieved=1)
      - OR it was manually confirmed (entry_date before valid_scan_date)
    """
    with _conn() as con:
        cur = con.execute(
            """UPDATE cup_handle_signals
               SET entry_triggered = 0,
                   entry_date      = NULL,
                   status          = 'Watching'
               WHERE entry_triggered = 1
                 AND t1_achieved = 0
                 AND t2_achieved = 0
                 AND t3_achieved = 0
                 AND stopped_out = 0
                 AND signal_type != 'BREAKOUT NOW'
                 AND (entry_date = ? OR entry_date IS NULL)
            """,
            (valid_scan_date,),
        )
        return cur.rowcount


def prune_expired_watchlist(today: Optional[date] = None) -> int:
    today = today or date.today()
    with _conn() as con:
        cur = con.execute(
            """UPDATE cup_handle_signals SET status = 'Expired'
               WHERE entry_triggered = 0
                 AND expiry_date IS NOT NULL
                 AND expiry_date < ?
                 AND status NOT IN ('Expired', 'Stopped Out')
            """,
            (today.isoformat(),),
        )
        return cur.rowcount
