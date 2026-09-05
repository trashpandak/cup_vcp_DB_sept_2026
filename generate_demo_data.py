"""
Generates synthetic NSE-style OHLCV data for a demo universe covering
all 4 pattern types, various maturity stages, and confirmation-candle
outcomes. Writes Parquet files into data/daily/ in the exact format
downloader.load_daily() expects, plus a synthetic NIFTY50 (^NSEI)
benchmark and a symbol-meta cache, so main.py's real detection/
readiness/entry-exit/persistence/export pipeline can run unmodified
against it with zero network access.

NOT real market data — fictional tickers, fictional company names,
prices designed purely to exercise each detector's geometry cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

RNG_SEED = 2026
END_DATE = "2026-08-31"


def bars_from_segments(segments, end_date=END_DATE, seed=0, noise=0.006):
    """
    segments: list of (n_bars, start_price, end_price, vol_lo, vol_hi)
    or (n_bars, start_price, end_price, vol_lo, vol_hi, noise_mult) —
    the optional 6th element scales the intraday noise for just that
    segment, so a "final base" segment can be given a low noise_mult
    (e.g. 0.35) to genuinely tighten its daily range, not just its net
    price swing — real volatility-contraction bases show BOTH.
    Linearly ramps Close across each segment, derives O/H/L with small
    random noise, concatenates, and anchors the LAST bar at end_date
    (business days, walking backward).
    """
    rng = np.random.default_rng(seed)
    closes, vols, noise_mults = [], [], []
    for seg in segments:
        n, p0, p1, vlo, vhi = seg[:5]
        nm = seg[5] if len(seg) > 5 else 1.0
        closes.append(np.linspace(p0, p1, n))
        vols.append(rng.uniform(vlo, vhi, n))
        noise_mults.append(np.full(n, nm))
    close = np.concatenate(closes)
    vol = np.concatenate(vols)
    seg_noise = np.concatenate(noise_mults) * noise
    n_total = len(close)

    idx = pd.bdate_range(end=end_date, periods=n_total)
    o = close * (1 + rng.uniform(-1, 1, n_total) * seg_noise)
    h = np.maximum(o, close) * (1 + rng.uniform(0.0, 1.5, n_total) * seg_noise)
    l = np.minimum(o, close) * (1 - rng.uniform(0.0, 1.5, n_total) * seg_noise)
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close, "Volume": vol}, index=idx)
    df.index.name = "Date"
    return df


def apply_candle_override(df: pd.DataFrame, bars_from_end: int, kind: str) -> pd.DataFrame:
    """Force the exact shape of a specific bar (counting back from the
    last row) so a chosen confirmation-candle pattern is guaranteed to
    land on the breakout bar, rather than leaving it to chance. Also
    boosts volume on that bar — real breakouts are volume events, and
    the scanner's "Confirmed Breakout" category specifically requires
    volume >= 1.4x average."""
    df = df.copy()
    pos = len(df) - 1 - bars_from_end
    prev_close = df["Close"].iloc[pos - 1]
    recent_avg_vol = df["Volume"].iloc[max(0, pos - 20):pos].mean()
    if kind == "bullish_engulfing":
        prev_o, prev_c = df["Open"].iloc[pos - 1], df["Close"].iloc[pos - 1]
        df.iloc[pos, df.columns.get_loc("Open")] = min(prev_o, prev_c) * 0.995
        df.iloc[pos, df.columns.get_loc("Close")] = max(prev_o, prev_c) * 1.02
        df.iloc[pos, df.columns.get_loc("High")] = max(prev_o, prev_c) * 1.025
        df.iloc[pos, df.columns.get_loc("Low")] = min(prev_o, prev_c) * 0.993
    elif kind == "marubozu":
        c = df["Close"].iloc[pos] * 1.015
        o = prev_close * 1.002
        df.iloc[pos, df.columns.get_loc("Open")] = o
        df.iloc[pos, df.columns.get_loc("Close")] = c
        df.iloc[pos, df.columns.get_loc("High")] = c * 1.003
        df.iloc[pos, df.columns.get_loc("Low")] = o * 0.998
    elif kind == "hammer":
        c = prev_close * 1.008
        o = prev_close * 1.001
        low = prev_close * 0.965
        df.iloc[pos, df.columns.get_loc("Open")] = o
        df.iloc[pos, df.columns.get_loc("Close")] = c
        df.iloc[pos, df.columns.get_loc("High")] = c * 1.004
        df.iloc[pos, df.columns.get_loc("Low")] = low
    elif kind == "three_soldiers":
        base = prev_close
        for k in range(3):
            j = pos - 2 + k
            o = base * (1 + 0.002 * k)
            c = base * (1 + 0.013 * (k + 1))
            df.iloc[j, df.columns.get_loc("Open")] = o
            df.iloc[j, df.columns.get_loc("Close")] = c
            df.iloc[j, df.columns.get_loc("High")] = c * 1.003
            df.iloc[j, df.columns.get_loc("Low")] = o * 0.997
            base = c
    elif kind == "weak_bearish":
        o = prev_close * 1.006
        c = prev_close * 0.994
        df.iloc[pos, df.columns.get_loc("Open")] = o
        df.iloc[pos, df.columns.get_loc("Close")] = c
        df.iloc[pos, df.columns.get_loc("High")] = o * 1.008
        df.iloc[pos, df.columns.get_loc("Low")] = c * 0.99

    if kind != "weak_bearish":
        vol_mult = 2.6 if kind == "three_soldiers" else 2.2
        if kind == "three_soldiers":
            for k in range(3):
                j = pos - 2 + k
                df.iloc[j, df.columns.get_loc("Volume")] = recent_avg_vol * (1.6 + 0.5 * k)
        else:
            df.iloc[pos, df.columns.get_loc("Volume")] = recent_avg_vol * vol_mult
    else:
        df.iloc[pos, df.columns.get_loc("Volume")] = recent_avg_vol * 0.85
    return df


def write_symbol(symbol: str, df: pd.DataFrame) -> None:
    safe = symbol.replace("^", "_").replace(".", "_")
    path = cfg.DAILY_DIR / f"{safe}.parquet"
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.to_parquet(path)


# ═════════════════════════════════════════════════════════════════════════
# Benchmark — synthetic NIFTY50, in a steady uptrend (Uptrend market regime)
# ═════════════════════════════════════════════════════════════════════════
def build_universe() -> dict:
    universe = {}

    nifty = bars_from_segments([
        (300, 19500, 22000, 3e8, 5e8),
        (60, 22000, 23200, 3e8, 5e8),
    ], seed=1)
    write_symbol(cfg.NIFTY50_SYMBOL, nifty)

    # ── Cup & Handle (5) ──
    universe["AXISFORGE.NS"] = dict(
        name="Axis Forge Industries", sector="Capital Goods",
        segments=[(70, 340, 520, 1.0e6, 1.4e6), (55, 520, 380, 1.6e6, 2.2e6),
                  (60, 380, 505, 1.2e6, 1.6e6), (18, 505, 495, 0.5e6, 0.7e6, 0.3),
                  (10, 495, 515, 1.8e6, 2.6e6)],
        seed=101, confirm="bullish_engulfing", confirm_bars_from_end=0,
    )
    universe["BRIGHTPOWER.NS"] = dict(
        name="Bright Power Grid Ltd", sector="Utilities",
        segments=[(65, 210, 300, 0.8e6, 1.1e6), (50, 300, 225, 1.2e6, 1.6e6),
                  (55, 225, 292, 0.9e6, 1.3e6), (16, 292, 286, 0.4e6, 0.6e6, 0.3),
                  (10, 286, 298, 0.9e6, 1.2e6)],
        seed=102,
    )
    universe["CRESTCHEM.NS"] = dict(
        name="Crest Specialty Chemicals", sector="Chemicals",
        segments=[(60, 150, 205, 0.6e6, 0.9e6), (45, 205, 158, 1.0e6, 1.3e6),
                  (55, 158, 200, 0.7e6, 1.0e6), (14, 200, 178, 0.5e6, 0.7e6, 0.3),
                  (12, 178, 188, 0.6e6, 0.8e6)],
        seed=103,
    )
    universe["DELTAAUTO.NS"] = dict(
        name="Delta Auto Components", sector="Auto Ancillaries",
        segments=[(65, 410, 590, 1.1e6, 1.5e6), (50, 590, 440, 1.6e6, 2.0e6),
                  (60, 440, 585, 1.0e6, 1.4e6), (12, 585, 592, 1.3e6, 1.7e6, 0.3)],
        seed=104,
    )
    universe["EVERESTFIN.NS"] = dict(
        name="Everest Financial Services", sector="NBFC",
        segments=[(20, 58, 62, 0.5e6, 0.7e6),
                  (55, 95, 118, 0.9e6, 1.2e6), (25, 118, 102, 1.3e6, 1.6e6),
                  (40, 102, 116, 0.8e6, 1.1e6), (15, 116, 112, 0.5e6, 0.7e6, 0.3),
                  (10, 112, 114, 0.6e6, 0.8e6)],
        seed=105,
    )

    # ── Double Bottom (4) ──
    universe["FALCONSTEEL.NS"] = dict(
        name="Falcon Steel & Alloys", sector="Metals",
        segments=[(25, 105, 112, 0.5e6, 0.7e6),
                  (60, 260, 205, 1.2e6, 1.6e6), (12, 205, 165, 1.8e6, 2.4e6),
                  (16, 165, 235, 1.4e6, 1.8e6), (12, 235, 168, 1.6e6, 2.0e6),
                  (16, 168, 225, 1.1e6, 1.5e6), (10, 225, 222, 0.5e6, 0.7e6, 0.3),
                  (8, 222, 232, 1.8e6, 2.4e6)],
        seed=201, confirm="three_soldiers", confirm_bars_from_end=0,
    )
    universe["GRANITEPHARMA.NS"] = dict(
        name="Granite Pharmaceuticals", sector="Pharma",
        segments=[(30, 175, 185, 0.5e6, 0.7e6),
                  (55, 420, 340, 1.0e6, 1.3e6), (14, 340, 275, 1.5e6, 1.9e6),
                  (18, 275, 385, 1.1e6, 1.5e6), (13, 385, 278, 1.3e6, 1.7e6),
                  (18, 278, 360, 0.9e6, 1.3e6), (12, 360, 356, 0.5e6, 0.6e6, 0.3)],
        seed=202,
    )
    universe["HAVENRETAIL.NS"] = dict(
        name="Haven Retail Ventures", sector="Retail",
        segments=[(45, 72, 78, 0.4e6, 0.6e6),
                  (55, 180, 150, 0.9e6, 1.2e6), (12, 150, 122, 1.4e6, 1.8e6),
                  (16, 122, 168, 1.0e6, 1.4e6), (12, 168, 125, 1.2e6, 1.6e6),
                  (16, 125, 150, 0.8e6, 1.1e6)],
        seed=203,
    )
    universe["IONTECH.NS"] = dict(
        name="Ion Technologies", sector="IT Services",
        segments=[(35, 225, 235, 0.4e6, 0.6e6),
                  (55, 610, 500, 0.7e6, 1.0e6), (12, 500, 405, 1.1e6, 1.5e6),
                  (16, 405, 560, 0.8e6, 1.2e6), (12, 560, 375, 1.0e6, 1.4e6),
                  (18, 375, 510, 0.7e6, 1.0e6), (10, 510, 502, 0.5e6, 0.6e6, 0.3)],
        seed=204,
    )

    # ── VCP (4) ──
    universe["JUPITERMOTORS.NS"] = dict(
        name="Jupiter Motors Ltd", sector="Automobile",
        pattern="vcp",
        segments=[(45, 320, 340, 0.9e6, 1.2e6), (48, 340, 470, 1.5e6, 1.9e6),
                  (18, 470, 366, 2.0e6, 2.6e6), (14, 366, 456, 1.2e6, 1.6e6),
                  (14, 456, 393, 1.0e6, 1.3e6), (12, 393, 442, 0.9e6, 1.1e6),
                  (12, 442, 411, 0.6e6, 0.8e6, 0.3), (16, 411, 468, 0.55e6, 0.75e6)],
        seed=301, confirm="hammer", confirm_bars_from_end=0,
    )
    universe["KRISHNATEXTILES.NS"] = dict(
        name="Krishna Textiles & Weaves", sector="Textiles",
        pattern="vcp",
        segments=[(45, 145, 152, 0.8e6, 1.0e6), (45, 152, 210, 1.3e6, 1.7e6),
                  (16, 210, 164, 1.8e6, 2.3e6), (13, 164, 202, 1.1e6, 1.4e6),
                  (13, 202, 176, 0.9e6, 1.2e6), (11, 176, 197, 0.8e6, 1.0e6),
                  (11, 197, 183, 0.55e6, 0.7e6, 0.3), (9, 183, 195, 0.5e6, 0.65e6),
                  (14, 195, 205, 0.45e6, 0.6e6)],
        seed=302,
    )
    universe["LOTUSINFRA.NS"] = dict(
        name="Lotus Infra Developers", sector="Infrastructure",
        pattern="vcp",
        segments=[(50, 88, 92, 0.7e6, 0.9e6), (42, 92, 128, 1.1e6, 1.4e6),
                  (16, 128, 100, 1.5e6, 1.9e6), (14, 100, 121, 0.9e6, 1.2e6),
                  (14, 121, 108, 0.7e6, 0.9e6, 0.3), (18, 108, 118, 0.5e6, 0.65e6)],
        seed=303,
    )
    universe["MERIDIANFOODS.NS"] = dict(
        name="Meridian Foods & Beverages", sector="FMCG",
        pattern="vcp",
        segments=[(15, 345, 355, 0.4e6, 0.55e6),
                  (48, 505, 515, 0.8e6, 1.0e6), (40, 515, 700, 1.2e6, 1.5e6),
                  (16, 700, 555, 1.6e6, 2.0e6), (13, 555, 665, 1.0e6, 1.3e6),
                  (13, 665, 590, 0.8e6, 1.0e6, 0.3), (18, 590, 635, 0.55e6, 0.7e6)],
        seed=304,
    )

    # ── Ascending Triangle (4) ──
    universe["NORTHCEMENT.NS"] = dict(
        name="North Star Cement", sector="Cement",
        pattern="triangle",
        segments=[(28, 335, 345, 0.4e6, 0.55e6),
                  (45, 480, 500, 1.0e6, 1.3e6), (18, 500, 638, 1.6e6, 2.0e6),
                  (12, 638, 512, 1.7e6, 2.1e6), (14, 512, 635, 1.1e6, 1.4e6),
                  (10, 635, 545, 0.9e6, 1.1e6), (12, 545, 633, 0.8e6, 1.0e6),
                  (9, 633, 578, 0.6e6, 0.75e6, 0.3), (12, 578, 636, 0.55e6, 0.7e6)],
        seed=401, confirm="marubozu", confirm_bars_from_end=0,
    )
    universe["OMEGALOGISTICS.NS"] = dict(
        name="Omega Logistics Corp", sector="Logistics",
        pattern="triangle",
        segments=[(30, 180, 190, 0.4e6, 0.5e6),
                  (45, 260, 270, 0.9e6, 1.1e6), (18, 270, 330, 1.4e6, 1.7e6),
                  (12, 330, 272, 1.5e6, 1.8e6), (13, 272, 328, 1.0e6, 1.2e6),
                  (10, 328, 288, 0.8e6, 1.0e6), (12, 288, 326, 0.7e6, 0.9e6),
                  (9, 326, 305, 0.5e6, 0.65e6, 0.3), (10, 305, 322, 0.5e6, 0.6e6)],
        seed=402,
    )
    universe["PIONEERSOLAR.NS"] = dict(
        name="Pioneer Solar Energy", sector="Renewable Energy",
        pattern="triangle",
        segments=[(42, 420, 435, 0.4e6, 0.55e6),
                  (45, 610, 630, 0.7e6, 0.9e6), (18, 630, 748, 1.2e6, 1.5e6),
                  (12, 748, 645, 1.3e6, 1.6e6), (13, 645, 744, 0.9e6, 1.1e6),
                  (10, 744, 680, 0.7e6, 0.85e6, 0.3), (14, 680, 740, 0.55e6, 0.7e6)],
        seed=403,
    )
    universe["QUANTUMHEALTH.NS"] = dict(
        name="Quantum Health Diagnostics", sector="Healthcare",
        pattern="triangle",
        segments=[(40, 100, 108, 0.4e6, 0.5e6),
                  (45, 155, 160, 0.8e6, 1.0e6), (18, 160, 190, 1.3e6, 1.6e6),
                  (12, 190, 162, 1.4e6, 1.7e6), (13, 162, 191, 0.9e6, 1.1e6),
                  (10, 191, 168, 0.75e6, 0.9e6, 0.3), (18, 168, 187, 0.55e6, 0.7e6)],
        seed=404,
    )

    # ── Special showcases ──
    # Long, clean, multi-year Cup & Handle designed to also read as a
    # valid (larger-scale) Cup & Handle when resampled to weekly bars —
    # a genuine same-pattern MTF confluence example.
    universe["RAPIDCONSUMER.NS"] = dict(
        name="Rapid Consumer Goods", sector="FMCG", mtf_showcase=True,
        segments=[(120, 150, 260, 0.9e6, 1.2e6), (90, 260, 175, 1.3e6, 1.7e6),
                  (100, 175, 250, 1.0e6, 1.3e6), (30, 250, 238, 0.5e6, 0.7e6, 0.3),
                  (15, 238, 252, 1.6e6, 2.1e6)],
        seed=501, confirm="bullish_engulfing", confirm_bars_from_end=0,
    )
    # No bullish confirmation on the breakout bar (weak/red candle) —
    # shows the "No Bullish Confirmation" state explicitly.
    universe["TITANMATERIALS.NS"] = dict(
        name="Titan Advanced Materials", sector="Materials",
        segments=[(65, 300, 420, 1.0e6, 1.3e6), (50, 420, 320, 1.4e6, 1.8e6),
                  (55, 320, 410, 0.9e6, 1.2e6), (16, 410, 402, 0.5e6, 0.65e6, 0.3),
                  (10, 402, 412, 1.2e6, 1.5e6)],
        seed=601, confirm="weak_bearish", confirm_bars_from_end=0,
    )

    return universe


def main():
    cfg.DAILY_DIR.mkdir(parents=True, exist_ok=True)
    universe = build_universe()

    meta = {}
    for symbol, spec in universe.items():
        df = bars_from_segments(spec["segments"], seed=spec["seed"])
        confirm = spec.get("confirm")
        if confirm:
            df = apply_candle_override(df, spec.get("confirm_bars_from_end", 0), confirm)
        write_symbol(symbol, df)
        meta[symbol] = {"name": spec["name"], "sector": spec["sector"]}
        print(f"{symbol:22s} {len(df):4d} bars  last_close={df['Close'].iloc[-1]:.2f}")

    meta_path = cfg.DATA_DIR / "nse_symbol_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    symbols_path = cfg.DATA_DIR / "nse_symbols.json"
    with open(symbols_path, "w") as f:
        json.dump(list(universe.keys()), f, indent=2)

    print(f"\nWrote {len(universe)} demo symbols + benchmark to {cfg.DAILY_DIR}")
    return list(universe.keys())


if __name__ == "__main__":
    main()
