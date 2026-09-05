"""
NSE Cup & Handle Scanner - Entry & Exit Calculator
=====================================================
Implements spec v3 Part 4: STRICT O'Neil/CANSLIM entry, stop loss,
targets, position sizing, and sell-rule checklist generation.

Unlike detection, every threshold here is a real gate that changes the
output (entry type, stop type, warnings). Detection decides WHETHER a
stock is shown; this module decides HOW to trade it if it's shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config as cfg
from cup_handle_detector import CupHandleSignal
from indicators import atr as atr_series_fn, rsi as rsi_fn, sma


@dataclass
class EntryExitPlan:
    entry_price: float
    entry_zone_high: float
    entry_type: str              # Primary / Pullback / Early-Aggressive / Wait
    extended_warning: Optional[str]

    stop_loss_price: float
    stop_loss_pct: float
    stop_loss_type: str          # handle_low_minus_1atr / cup_bottom_zone / 8pct_cap
    atr_14: float
    risk_per_share: float

    target1: float
    target2: float
    target3: float
    rr_t1: float
    rr_t2: float
    rr_t3: float
    rr_t2_warning: Optional[str]
    eight_week_hold_candidate: bool

    position_size_shares: int
    capital_required: float
    risk_amount: float
    portfolio_risk_pct: float

    volume_ratio: float
    volume_confirmed: bool
    volume_confirmed_label: str   # Yes / No / N/A

    market_note: Optional[str]
    weak_momentum_note: Optional[str]
    below_50ma_note: Optional[str]

    sell_notes: str
    liquidity_ok: bool
    liquidity_warning: Optional[str]


def calculate_entry_exit(
    sig: CupHandleSignal,
    df: pd.DataFrame,
    nifty_trend: str,             # "Uptrend" / "Correction" / "Bear"
) -> EntryExitPlan:
    """
    Compute the full strict entry/exit plan for a detected signal.
    Always returns a plan (never None) — even low-quality or early-stage
    detections get entry/exit numbers pre-computed for the report, per
    spec 4A "every detected setup gets entry/exit calculations
    pre-computed."
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    current_price = float(close.iloc[-1])

    # ── Entry price & buffer ──
    buffer = (
        cfg.BREAKOUT_BUFFER_PCT * sig.pivot_point
        if sig.pivot_point > cfg.BREAKOUT_BUFFER_PRICE_CUTOFF
        else cfg.BREAKOUT_BUFFER_INR
    )
    entry_price = sig.pivot_point + buffer
    entry_zone_high = entry_price * (1 + cfg.EXTENDED_THRESHOLD)

    entry_type, extended_warning = _classify_entry_type(sig, current_price, entry_price)

    # ── ATR ──
    atr_series = atr_series_fn(df, period=cfg.ATR_PERIOD)
    atr_14 = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    # ── Stop loss ──
    stop_loss_price, stop_loss_type = _calculate_stop_loss(sig, entry_price, atr_14)
    stop_loss_pct = ((entry_price - stop_loss_price) / entry_price) * 100.0 if entry_price else 0.0
    risk_per_share = max(entry_price - stop_loss_price, 0.01)

    # ── Targets ──
    cup_depth_points = sig.left_rim_price - sig.cup_bottom_price
    target1 = entry_price * (1 + cfg.T1_GAIN_PCT)
    target2 = entry_price + cup_depth_points
    target3 = entry_price + cup_depth_points * cfg.FIBONACCI_EXTENSION

    rr_t1 = (target1 - entry_price) / risk_per_share
    rr_t2 = (target2 - entry_price) / risk_per_share
    rr_t3 = (target3 - entry_price) / risk_per_share
    rr_t2_warning = (
        f"R:R below {cfg.MIN_RR_T2}:1 at T2 — position size carefully"
        if rr_t2 < cfg.MIN_RR_T2 else None
    )

    eight_week_hold_candidate = _check_8week_candidate(sig)

    # ── Position sizing ──
    max_risk_amount = cfg.PORTFOLIO_VALUE * (cfg.RISK_PER_TRADE_PCT / 100.0)
    position_size_shares = int(max_risk_amount // risk_per_share) if risk_per_share > 0 else 0
    capital_required = position_size_shares * entry_price
    risk_amount = position_size_shares * risk_per_share
    portfolio_risk_pct = (risk_amount / cfg.PORTFOLIO_VALUE) * 100.0 if cfg.PORTFOLIO_VALUE else 0.0

    # ── Volume confirmation ──
    volume_ratio, volume_confirmed, volume_confirmed_label = _check_volume_confirmation(
        sig, df
    )

    # ── Soft conditions ──
    market_note = (
        "Counter-trend — higher risk" if nifty_trend != "Uptrend" else None
    )
    below_50ma_note = None
    weak_momentum_note = None
    try:
        ma_period = {"daily": 50, "weekly": 10, "monthly": 3}.get(sig.timeframe, 50)
        if len(close) >= ma_period:
            ma50 = sma(close, ma_period).iloc[-1]
            if pd.notna(ma50) and current_price < float(ma50):
                below_50ma_note = "Below 50MA — caution"
    except Exception:
        pass

    try:
        rsi_val = rsi_fn(close, period=cfg.RSI_PERIOD).iloc[-1]
        if pd.notna(rsi_val) and float(rsi_val) < cfg.RSI_WEAK_MOMENTUM_THRESHOLD:
            weak_momentum_note = "Weak momentum"
    except Exception:
        pass

    # ── Liquidity gate (applied after detection, per spec 10.4) ──
    liquidity_ok, liquidity_warning = _check_liquidity(df, current_price)

    # ── Sell rules checklist ──
    sell_notes = _build_sell_notes(eight_week_hold_candidate)

    return EntryExitPlan(
        entry_price=round(entry_price, 2),
        entry_zone_high=round(entry_zone_high, 2),
        entry_type=entry_type,
        extended_warning=extended_warning,
        stop_loss_price=round(stop_loss_price, 2),
        stop_loss_pct=round(stop_loss_pct, 2),
        stop_loss_type=stop_loss_type,
        atr_14=round(atr_14, 2),
        risk_per_share=round(risk_per_share, 2),
        target1=round(target1, 2),
        target2=round(target2, 2),
        target3=round(target3, 2),
        rr_t1=round(rr_t1, 2),
        rr_t2=round(rr_t2, 2),
        rr_t3=round(rr_t3, 2),
        rr_t2_warning=rr_t2_warning,
        eight_week_hold_candidate=eight_week_hold_candidate,
        position_size_shares=position_size_shares,
        capital_required=round(capital_required, 2),
        risk_amount=round(risk_amount, 2),
        portfolio_risk_pct=round(portfolio_risk_pct, 3),
        volume_ratio=round(volume_ratio, 2) if volume_ratio is not None else 0.0,
        volume_confirmed=volume_confirmed,
        volume_confirmed_label=volume_confirmed_label,
        market_note=market_note,
        weak_momentum_note=weak_momentum_note,
        below_50ma_note=below_50ma_note,
        sell_notes=sell_notes,
        liquidity_ok=liquidity_ok,
        liquidity_warning=liquidity_warning,
    )


# ─── Entry classification ──────────────────────────────────────────────────

def _classify_entry_type(
    sig: CupHandleSignal, current_price: float, entry_price: float
) -> tuple[str, Optional[str]]:
    if not sig.has_handle:
        return "Early-Aggressive", None

    pct_above_pivot = sig.price_vs_pivot_pct

    if pct_above_pivot > 10:
        return "Extended", "DO NOT CHASE"
    if pct_above_pivot > cfg.EXTENDED_THRESHOLD * 100:
        return "Extended", "Wait for pullback to 21-day EMA"
    if sig.signal_type == "BREAKOUT NOW":
        return "Primary", None
    if -2.0 <= pct_above_pivot < 0 and sig.signal_type in ("NEAR BREAKOUT",):
        return "Pullback", None
    return "Primary", None


# ─── Stop loss ──────────────────────────────────────────────────────────────

def _calculate_stop_loss(
    sig: CupHandleSignal, entry_price: float, atr_14: float
) -> tuple[float, str]:
    if sig.has_handle and sig.handle_low_price is not None:
        raw_stop = sig.handle_low_price - (cfg.STOP_ATR_MULTIPLIER * atr_14)
        stop_type = "handle_low_minus_1atr"
    else:
        cup_height = sig.left_rim_price - sig.cup_bottom_price
        raw_stop = sig.cup_bottom_price + cfg.CUP_ONLY_STOP_RECOVERY_FRACTION * cup_height
        stop_type = "cup_bottom_zone"

    hard_cap_stop = entry_price * (1 - cfg.MAX_STOP_PCT)

    if raw_stop < hard_cap_stop:
        return hard_cap_stop, "8pct_cap"

    return raw_stop, stop_type


# ─── Targets / 8-week rule ──────────────────────────────────────────────────

def _check_8week_candidate(sig: CupHandleSignal) -> bool:
    """
    Flag as an 8-Week Hold Rule candidate if the pattern's handle/cup
    structure suggests a fast, powerful move is plausible — specifically
    if the stock is already at or very near breakout with strong rim
    symmetry and a tight handle. This is a forward-looking flag (we
    can't know actual days-to-T1 until the trade is live); it tells the
    trader to watch for the fast-20%-gain scenario described in spec 4E.
    """
    if sig.signal_type != "BREAKOUT NOW":
        return False
    if sig.rim_symmetry_pct >= 90 and sig.handle_quality_subscore >= 12:
        return True
    return False


# ─── Volume confirmation ───────────────────────────────────────────────────

def _check_volume_confirmation(
    sig: CupHandleSignal, df: pd.DataFrame
) -> tuple[Optional[float], bool, str]:
    if "Volume" not in df.columns:
        return None, False, "N/A"

    from indicators import clean_volume
    volume = clean_volume(df["Volume"])

    if sig.timeframe == "daily":
        avg_bars = cfg.VOLUME_AVG_BARS_DAILY
        threshold = cfg.VOLUME_CONFIRM_DAILY
    else:
        avg_bars = cfg.VOLUME_AVG_BARS_WEEKLY
        threshold = cfg.VOLUME_CONFIRM_WEEKLY

    if len(volume) < avg_bars + 1:
        return None, False, "N/A"

    avg_vol = volume.iloc[-(avg_bars + 1):-1].mean()
    breakout_vol = volume.iloc[-1]

    if avg_vol <= 0 or pd.isna(avg_vol):
        return None, False, "N/A"

    ratio = breakout_vol / avg_vol
    confirmed = ratio >= threshold

    if sig.signal_type != "BREAKOUT NOW":
        # Volume confirmation only strictly applies to live breakouts;
        # for watching/basing signals we still show the ratio for context
        return ratio, confirmed, "N/A"

    return ratio, confirmed, ("Yes" if confirmed else "No")


# ─── Liquidity gate ─────────────────────────────────────────────────────────

def _check_liquidity(df: pd.DataFrame, current_price: float) -> tuple[bool, Optional[str]]:
    if current_price < cfg.MIN_PRICE:
        return False, f"Price below ₹{cfg.MIN_PRICE} — too illiquid for entry/exit sizing"

    if "Volume" not in df.columns or len(df) < cfg.LIQUIDITY_LOOKBACK_BARS:
        return True, None

    from indicators import clean_volume
    volume = clean_volume(df["Volume"])
    avg_vol = volume.iloc[-cfg.LIQUIDITY_LOOKBACK_BARS:].mean()

    if pd.isna(avg_vol) or avg_vol < cfg.MIN_AVG_VOLUME:
        return False, f"Avg volume below {cfg.MIN_AVG_VOLUME:,} shares — too illiquid"

    return True, None


# ─── Sell rules checklist ───────────────────────────────────────────────────

def _build_sell_notes(eight_week_hold_candidate: bool) -> str:
    mandatory = [
        "Stop loss hit -> exit same session, no exceptions",
        f"Stock falls {int(cfg.TRAILING_STOP_PCT*100)}% from any post-entry closing high -> trailing stop",
    ]
    recommended = [
        "Cuts below 50d MA on volume >=150% avg",
        "Three distribution days in two weeks",
        f"No new high after {cfg.STUCK_BASE_WEEKS} weeks (stuck base)",
        "Nifty enters distribution phase (4+ distribution days in 4 weeks)",
    ]
    hold = []
    if eight_week_hold_candidate:
        hold.append("8-week hold rule may activate if T1 reached in <15 days -> hold minimum 8 weeks")
    hold.append("Stage 2 uptrend, RS >=80, above all key MAs -> hold")

    parts = ["MANDATORY: " + "; ".join(mandatory)]
    parts.append("RECOMMENDED: " + "; ".join(recommended))
    parts.append("HOLD IF: " + "; ".join(hold))
    return " || ".join(parts)
