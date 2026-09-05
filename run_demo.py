"""
Runs the ACTUAL production pipeline (detection -> readiness -> entry/
exit -> persistence -> Excel report -> HTML dashboard) against the
synthetic demo universe, with zero network access. Mirrors main.main()
exactly, minus fetch_nse_symbols()/run_download() (which need the
internet) — everything downstream is byte-for-byte the real code path.
"""
from __future__ import annotations

import json
import traceback
from datetime import date, datetime

import config as cfg
import database as db
from main import (
    TIMEFRAMES, _build_summary_stats, _generate_report, _load_benchmark,
    _compute_rs_ratings, _scan_symbol, _update_active_tracking,
)
from chart_export import export_html_dashboard
from logger_utils import get_logger

log = get_logger("scanner")


def main():
    started_at = datetime.now()
    db.init_db()

    symbols = json.load(open(cfg.DATA_DIR / "nse_symbols.json"))
    log.info("Demo universe: %d symbols", len(symbols))

    nifty_trend, nifty_df = _load_benchmark()
    log.info("Nifty trend: %s", nifty_trend)

    rs_ratings, rs_trends = _compute_rs_ratings(symbols)
    log.info("RS ratings computed for %d symbols", len(rs_ratings))

    scan_date = date.today().isoformat()
    error_count = 0
    pattern_count = 0
    by_timeframe = {tf: 0 for tf in TIMEFRAMES}
    by_signal_type: dict[str, int] = {}
    by_rs_band = {"<60": 0, "60-80": 0, ">80": 0}

    for symbol in symbols:
        try:
            n_found = _scan_symbol(
                symbol, rs_ratings, rs_trends, nifty_trend, scan_date,
                by_timeframe, by_signal_type, by_rs_band,
            )
            pattern_count += n_found
            log.info("  %-22s -> %d signal(s)", symbol, n_found)
        except Exception:
            error_count += 1
            log.warning("Error scanning %s:\n%s", symbol, traceback.format_exc())

    log.info("Scan complete: %d errors out of %d symbols, %d total signals",
              error_count, len(symbols), pattern_count)

    db.prune_expired_watchlist()
    _update_active_tracking()

    summary_stats = _build_summary_stats(
        scan_date, len(symbols), pattern_count, by_timeframe,
        by_signal_type, by_rs_band, nifty_trend,
    )
    log.info("Summary: %s", {k: v for k, v in summary_stats.items() if not isinstance(v, dict)})

    try:
        _generate_report(scan_date, summary_stats)
    except Exception:
        log.warning("Excel report generation failed (non-fatal):\n%s", traceback.format_exc())

    html_path = export_html_dashboard(scan_date, nifty_trend)

    elapsed = (datetime.now() - started_at).total_seconds()
    log.info("Demo run complete in %.1fs — %d patterns detected across %d symbols",
              elapsed, pattern_count, len(symbols))
    log.info("HTML dashboard: %s", html_path)
    return html_path


if __name__ == "__main__":
    main()
