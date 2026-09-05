"""
NSE Cup & Handle Scanner - Explanation Generator
====================================================
Converts a single DB row (a dict, as returned by database.py's
get_*_df().to_dict("records") or sqlite3.Row) into human-readable
prose: why the scanner flagged this stock, what's attractive about
it, what's risky, whether buying is recommended, and a rule-by-rule
PASS/FAIL/WARNING breakdown of every detection gate it passed.

Deliberately rule-based, not a live model call — every sentence here
traces back to an actual field the scanner already computed (quality
score, cup depth class, RS rating, volume ratio, etc.). This keeps
the explanations honest: nothing here is invented or guessed, and if
a field is missing the corresponding sentence is simply omitted
rather than filled with a generic placeholder.

Consumed by chart_export (baked into main.py) to build each symbol's
`explanation` payload before it's serialized into the dashboard JSON.
"""

from __future__ import annotations

from typing import Optional

import pattern_common as pc


def _vocab(row: dict) -> dict:
    return pc.vocab_for(row.get("pattern_family"))


def _is_preconfirm(signal_type: Optional[str]) -> bool:
    return signal_type in ("CUP ONLY", "BASE FORMING")


def _shape_status(pattern_family: Optional[str], shape: str) -> tuple[str, str]:
    """Pattern-aware pass/warn assessment of the structure's shape string."""
    if pattern_family == "double_bottom":
        if shape.startswith("Even") or shape.startswith("Ascending"):
            return "pass", (f"{shape} — the two lows are close in price (or the second is "
                             f"higher), the strongest variant of this pattern.")
        return "warn", (f"{shape} — the second low sits meaningfully below the first; still "
                         f"valid (often a shakeout) but a more aggressive read.")
    if pattern_family == "vcp":
        if "Textbook" in shape or shape.count("→") >= 2:
            return "pass", f"{shape} — a clean, progressively tightening base."
        return "warn", (f"{shape} — meets the minimum bar for a VCP, but the tightening "
                         f"isn't as clean as a textbook 3+ contraction base.")
    if pattern_family == "ascending_triangle":
        if shape.startswith("Textbook"):
            return "pass", f"{shape} — resistance is tightly flat and the rising lows are clean."
        return "warn", f"{shape} — a valid ascending triangle, though resistance is a bit looser than textbook."
    # cup_handle (default)
    if shape == "U-Shape":
        return "pass", "Smooth, rounded bottom — the classic, most reliable cup shape."
    if shape == "V-Shape":
        return "warn", "Sharper, faster reversal at the bottom — still valid but more aggressive than a textbook cup."
    if shape == "Irregular":
        return "warn", "Bottom shows a double-bottom or irregular structure — treat with extra caution."
    return "warn", f"Shape classified as {shape}."


def _f(row: dict, key: str, default=None):
    """Safe field getter — treats NaN/None uniformly as missing."""
    val = row.get(key, default)
    if val is None:
        return default
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return default
    except Exception:
        pass
    return val


def _money(val) -> str:
    if val is None:
        return "—"
    return f"₹{val:,.2f}"


def _pct(val, dp: int = 1) -> str:
    if val is None:
        return "—"
    return f"{val:.{dp}f}%"


# ─── Overall rating ─────────────────────────────────────────────────────────

def overall_rating(row: dict) -> dict:
    """
    Returns {"label": str, "color": str, "reason": str} — the single
    top-line verdict shown prominently in the detail panel header.
    """
    v = _vocab(row)
    quality = _f(row, "quality_score", 0) or 0
    readiness = _f(row, "breakout_readiness_pct")
    signal_type = _f(row, "signal_type", "")
    has_handle = bool(_f(row, "has_handle", 0))
    liquidity_ok = bool(_f(row, "liquidity_ok", 1))
    stop_capped = _f(row, "stop_loss_type") == "8pct_cap"

    if not liquidity_ok:
        return {
            "label": "Rejected — Illiquid",
            "color": "grey",
            "reason": "Price or average volume falls below the liquidity floor; "
                      "not safely tradeable at any conviction level.",
        }

    if signal_type == "BREAKOUT NOW" and has_handle and quality >= 70 and (readiness or 0) >= 80:
        return {
            "label": "Strong Buy",
            "color": "green",
            "reason": f"Confirmed breakout with a well-formed {v['base_noun']}, high "
                      f"quality score, and high breakout readiness — this is the "
                      f"scanner's highest-conviction category.",
        }

    if signal_type in ("BREAKOUT NOW", "NEAR BREAKOUT") and has_handle and quality >= 55:
        return {
            "label": "Good Candidate",
            "color": "blue",
            "reason": f"{v['name']} structure with a real {v['base_noun']} and a solid "
                      f"quality score, at or near its breakout pivot.",
        }

    if signal_type == "NEAR BREAKOUT" or signal_type == "BASING":
        return {
            "label": "Watch",
            "color": "yellow",
            "reason": "Pattern is still developing — worth monitoring for a cleaner "
                      "setup or a confirmed breakout before committing capital.",
        }

    if _is_preconfirm(signal_type):
        return {
            "label": f"Near Ready — No {v['base_noun_cap']} Yet",
            "color": "orange",
            "reason": f"The {v['structure_noun']} recovery is in place but no "
                      f"{v['base_noun']} has formed yet; too early for an entry, but "
                      f"worth tracking for {v['base_noun']} formation.",
        }

    if quality < 40:
        return {
            "label": "Weak",
            "color": "red",
            "reason": "Low pattern quality score — geometry is loose, shallow, or "
                      "poorly confirmed by volume and relative strength.",
        }

    return {
        "label": "Early Stage",
        "color": "orange",
        "reason": "Detected but still far from its pivot; not yet actionable.",
    }


# ─── "Why did the scanner detect this?" — rule PASS/FAIL/WARNING list ─────

def scanner_reasons(row: dict) -> list[dict]:
    """
    Returns a list of {"rule": str, "status": "pass"|"fail"|"warn", "detail": str}
    — one entry per detection/entry gate the scanner actually evaluated,
    in the order a trader would think through them.
    """
    v = _vocab(row)
    reasons = []

    def add(rule, status, detail):
        reasons.append({"rule": rule, "status": status, "detail": detail})

    # The one hard detection gate (worded per pattern family)
    recovery = _f(row, "recovery_pct")
    if recovery is not None:
        if row.get("pattern_family") == "double_bottom":
            gate_detail = (f"Price recovered {recovery:.0f}% of the decline from the "
                            f"{v['high_label'].lower()} down to Bottom 2 (scanner requires ≥ 50%).")
        elif row.get("pattern_family") in ("vcp", "ascending_triangle"):
            gate_detail = (f"Price has recovered {recovery:.0f}% of the way from the base's "
                            f"deepest point back toward the {v['rim_noun']} (scanner requires the "
                            f"final base/contraction gates below, not a single recovery %).")
        else:
            gate_detail = (f"Right rim recovered {recovery:.0f}% of the decline from the left rim "
                            f"to the cup bottom (scanner requires ≥ 50%).")
        add(f"{v['structure_noun_cap']} recovery ≥ 50%", "pass" if recovery >= 50 else "fail", gate_detail)

    # Depth classification
    depth = _f(row, "cup_depth_pct")
    depth_class = _f(row, "cup_depth_class")
    if depth is not None:
        verify_flag = bool(_f(row, "cup_depth_verify_flag", 0))
        status = "warn" if verify_flag else "pass"
        add(
            f"{v['depth_noun'].capitalize()}",
            status,
            f"{depth:.1f}% decline from {v['high_label'].lower()} to {v['low_label'].lower()} — "
            f"classified as '{depth_class}'." + (
                f" Very shallow — verify the chart manually before trusting this as a real "
                f"{v['structure_noun']}." if verify_flag else ""
            ),
        )

    # Shape / quality of the structure
    shape = _f(row, "cup_shape")
    if shape:
        shape_status, shape_detail = _shape_status(row.get("pattern_family"), shape)
        add(f"{v['shape_noun'].capitalize()}", shape_status, shape_detail)

    # Final base ("handle") presence and quality
    has_handle = bool(_f(row, "has_handle", 0))
    if has_handle:
        hq = _f(row, "handle_quality_subscore", 0) or 0
        add(
            f"{v['base_noun_cap']} formed",
            "pass",
            f"A {v['base_noun']} has formed, scoring {hq:.0f}/20 on volume dry-up, range "
            f"tightness, higher lows, and close position.",
        )
        handle_depth = _f(row, "handle_depth_pct")
        if handle_depth is not None:
            add(
                f"{v['base_noun_cap']} depth reasonable",
                "pass" if handle_depth <= 15 else "warn",
                f"{v['base_noun_cap']} pulled back {handle_depth:.1f}% from the {v['confirm_label'].lower()}.",
            )
    else:
        add(
            f"{v['base_noun_cap']} formed",
            "fail",
            f"No {v['base_noun']} detected yet — this is a {v['preconfirm_label'].title()} / "
            f"pre-confirmation setup. Entry rules are more aggressive and higher-risk until a "
            f"{v['base_noun']} forms.",
        )

    # Prior trend context
    uptrend_tag = _f(row, "prior_uptrend_tag")
    uptrend_pct = _f(row, "prior_uptrend_pct")
    if uptrend_tag:
        status = "pass" if uptrend_tag == "Strong Continuation" else "warn" if uptrend_tag == "Moderate Continuation" else "warn"
        add(
            "Prior trend",
            status,
            f"{uptrend_pct:.1f}% gain before the {v['structure_noun']} began — tagged "
            f"'{uptrend_tag}'." + (
                f" This looks more like a base-building/bottoming setup than a textbook "
                f"continuation pattern — a different risk profile, not necessarily worse."
                if uptrend_tag == "Possible Bottoming Pattern" else ""
            ),
        )

    # Volume confirmation
    vol_label = _f(row, "volume_confirmed_label")
    vol_ratio = _f(row, "volume_ratio")
    if vol_ratio is not None:
        if vol_label == "Yes":
            add("Volume confirmed", "pass",
                f"Breakout-day volume is {vol_ratio:.2f}× the average — real "
                f"buying interest behind the move, not a low-volume drift.")
        elif vol_label == "No":
            add("Volume confirmed", "fail",
                f"Volume is only {vol_ratio:.2f}× average — O'Neil's rule requires "
                f"≥1.4×. A breakout without volume is much more likely to fail.")
        else:
            add("Volume", "warn",
                f"Volume ratio is {vol_ratio:.2f}× average (not yet a live breakout, "
                f"so this isn't gated the same way).")

    # RS Rating
    rs_rating = _f(row, "rs_rating")
    rs_tag = _f(row, "rs_tag")
    if rs_rating is not None:
        status = "pass" if rs_rating >= 70 else "warn" if rs_rating >= 50 else "fail"
        add(
            "Relative Strength",
            status,
            f"RS Rating {rs_rating:.0f}/99" + (f" ({rs_tag})" if rs_tag and rs_tag != "-" else "") +
            " — measures this stock's return against the whole NSE universe, not just the index.",
        )

    rs_trend = _f(row, "rs_trend")
    if rs_trend and rs_trend != "Unknown":
        add(
            "RS trend",
            "pass" if rs_trend == "Improving" else "warn" if rs_trend == "Flat" else "fail",
            f"RS Rating has been {rs_trend.lower()} over the last 4 weeks.",
        )

    # ATR contraction (readiness component)
    if _f(row, "breakout_readiness_pct") is not None:
        atr_ok = bool(_f(row, "readiness_atr_contract", 0))
        add(
            "Volatility contracting (ATR)",
            "pass" if atr_ok else "warn",
            "Current ATR is lower than it was at the start of the handle — a "
            "genuine volatility squeeze, the kind that often precedes a real move."
            if atr_ok else
            "ATR hasn't contracted since the handle began — less of a coiled-spring setup.",
        )

        above_ma = bool(_f(row, "readiness_above_50ma", 0))
        add(
            "Above key moving average",
            "pass" if above_ma else "fail",
            "Price is trading above its 50-day (or equivalent) moving average."
            if above_ma else
            "Price is below its 50-day (or equivalent) moving average — a real "
            "concern; O'Neil's own rules treat this as a caution/avoid signal.",
        )

    # Liquidity
    liquidity_ok = bool(_f(row, "liquidity_ok", 1))
    liquidity_warning = _f(row, "liquidity_warning")
    add(
        "Liquidity",
        "pass" if liquidity_ok else "fail",
        liquidity_warning or "Price and average volume both clear the minimum "
        "liquidity floor for safe entry/exit sizing.",
    )

    # Stop loss width
    stop_type = _f(row, "stop_loss_type")
    if stop_type:
        if stop_type == "8pct_cap":
            add("Stop loss width", "warn",
                "Stop is capped at the maximum allowed 8% below entry — the "
                "natural stop (handle low or cup-bottom zone) was wider than "
                "that, so risk here is at the ceiling of what the strategy permits.")
        else:
            add("Stop loss width", "pass",
                f"Stop placed at the natural {stop_type.replace('_', ' ')} level, "
                f"comfortably inside the 8% maximum.")

    # R:R at T2
    rr_t2 = _f(row, "rr_t2")
    if rr_t2 is not None:
        add(
            "Risk:Reward at Target 2",
            "pass" if rr_t2 >= 2.0 else "warn",
            f"{rr_t2:.2f}:1 measured to the cup-depth-projected target." +
            (" Below the 2:1 minimum the strategy prefers — size position "
             "carefully if taking this trade." if rr_t2 < 2.0 else ""),
        )

    return reasons


# ─── "Why should I buy this?" ───────────────────────────────────────────────

def why_buy(row: dict) -> dict:
    """
    Returns {"recommend": bool, "paragraphs": [str, ...]} — the
    dedicated "Why Should I Buy This?" section. If the scanner
    considers the stock weak, recommend=False and the text explains
    why buying is NOT currently advisable, rather than forcing a
    generic bullish narrative onto a weak setup.
    """
    rating = overall_rating(row)
    v = _vocab(row)
    quality = _f(row, "quality_score", 0) or 0
    signal_type = _f(row, "signal_type", "")
    has_handle = bool(_f(row, "has_handle", 0))
    rs_rating = _f(row, "rs_rating")
    rs_trend = _f(row, "rs_trend")
    vol_ratio = _f(row, "volume_ratio")
    readiness = _f(row, "breakout_readiness_pct")
    rr_t2 = _f(row, "rr_t2")
    liquidity_ok = bool(_f(row, "liquidity_ok", 1))
    mtf = bool(_f(row, "mtf_confluence", 0))

    recommend = (
        liquidity_ok
        and has_handle
        and signal_type in ("BREAKOUT NOW", "NEAR BREAKOUT")
        and quality >= 55
    )

    paragraphs = []

    if not liquidity_ok:
        paragraphs.append(
            _f(row, "liquidity_warning") or
            "This stock fails the liquidity floor — price or average volume is "
            "too low to size and exit a position safely. Buying is not recommended "
            "regardless of how the pattern looks."
        )
        return {"recommend": False, "paragraphs": paragraphs}

    # Trend
    uptrend_tag = _f(row, "prior_uptrend_tag")
    if uptrend_tag == "Strong Continuation":
        paragraphs.append(
            f"Trend: this {v['structure_noun']} formed after a strong prior advance, which "
            f"is the textbook {v['name']} setup — a continuation of an existing uptrend "
            f"rather than a bet on a turnaround."
        )
    elif uptrend_tag == "Moderate Continuation":
        paragraphs.append(
            f"Trend: a moderate prior uptrend preceded this {v['structure_noun']} — a "
            f"reasonable but not textbook-strong continuation setup."
        )
    elif uptrend_tag == "Possible Bottoming Pattern":
        paragraphs.append(
            f"Trend: there wasn't much of a prior uptrend before this {v['structure_noun']} "
            f"started, which means this reads more like a stock attempting to bottom and "
            f"turn around than a clean continuation pattern. That's a different, generally "
            f"higher-risk thesis — size accordingly."
        )

    # Momentum / RS
    if rs_rating is not None:
        if rs_rating >= 85:
            paragraphs.append(
                f"Momentum: RS Rating of {rs_rating:.0f} marks this as a genuine "
                f"relative-strength leader against the rest of the NSE universe, "
                f"not just against the index."
            )
        elif rs_rating >= 70:
            paragraphs.append(
                f"Momentum: RS Rating of {rs_rating:.0f} shows building relative "
                f"strength, without yet being a clear leader."
            )
        else:
            paragraphs.append(
                f"Momentum: RS Rating of {rs_rating:.0f} is on the weaker side — "
                f"this stock hasn't been a relative-strength leader recently, "
                f"which lowers the odds of a fast, powerful move even if the "
                f"chart pattern looks clean." +
                (f" The trend is at least {rs_trend.lower()}." if rs_trend and rs_trend != "Unknown" else "")
            )

    # Volume
    if vol_ratio is not None:
        if vol_ratio >= 1.4:
            paragraphs.append(
                f"Volume: {vol_ratio:.2f}× average — real participation behind "
                f"the move, the kind of confirmation institutional buying tends "
                f"to leave behind."
            )
        elif vol_ratio >= 1.0:
            paragraphs.append(
                f"Volume: {vol_ratio:.2f}× average — present but not surging. "
                f"Worth waiting for a stronger volume day before treating this "
                f"as a confirmed institutional move."
            )
        else:
            paragraphs.append(
                f"Volume: only {vol_ratio:.2f}× average — below-average "
                f"participation is a real caution flag; moves without volume "
                f"often fail to hold."
            )

    # Pattern maturity
    if not has_handle:
        paragraphs.append(
            f"Pattern maturity: no {v['base_noun']} has formed yet. The {v['structure_noun']} "
            f"recovery alone isn't enough to trade — wait for the {v['base_noun']} to form "
            f"and tighten before considering entry."
        )
    elif signal_type == "BREAKOUT NOW":
        paragraphs.append(
            f"Pattern maturity: the {v['base_noun']} is complete and price has already "
            f"crossed the pivot — this is a live, confirmed breakout, not a forecast."
        )
    elif signal_type == "NEAR BREAKOUT":
        paragraphs.append(
            f"Pattern maturity: the {v['base_noun']} is complete and price is right at the "
            f"pivot — a buy-stop order slightly above the pivot is the standard way to "
            f"catch this without guessing the exact breakout day."
        )

    # Bullish confirmation candle at the breakout
    cc_pattern = _f(row, "confirmation_candle_pattern")
    cc_found = bool(_f(row, "confirmation_candle_found", 0))
    cc_strength = _f(row, "confirmation_candle_strength")
    if cc_pattern and signal_type in ("BREAKOUT NOW", "NEAR BREAKOUT", "BASING"):
        if cc_found and cc_strength in ("Strong", "Moderate"):
            paragraphs.append(
                f"Confirmation candle: {cc_pattern} ({cc_strength}) — "
                f"{_f(row, 'confirmation_candle_detail', '')}"
            )
        elif signal_type == "BREAKOUT NOW":
            paragraphs.append(
                "Confirmation candle: the breakout bar itself doesn't show a strong bullish "
                "candle pattern yet — the level has been crossed, but waiting for a stronger "
                "confirmation bar (or at least a solid close near the day's high) reduces the "
                "odds of buying into a same-day fakeout."
            )

    # Probability proxy = readiness
    if readiness is not None:
        paragraphs.append(
            f"Probability: Breakout Readiness sits at {readiness:.0f}%, based on "
            f"how many of the five confirmation factors (proximity to pivot, "
            f"{v['base_noun']} tightness, rising RS, volatility contraction, and price "
            f"above its key moving average) are currently true."
        )

    # Risk / Reward
    stop_pct = _f(row, "stop_loss_pct")
    if stop_pct is not None and rr_t2 is not None:
        paragraphs.append(
            f"Risk/Reward: the stop is {stop_pct:.1f}% below entry, and the "
            f"cup-depth-projected Target 2 offers a {rr_t2:.2f}:1 reward-to-risk "
            f"ratio." + (" That clears the strategy's 2:1 minimum comfortably." if rr_t2 >= 2.0 else
                          " That's below the 2:1 minimum this strategy prefers — "
                          "still tradeable, but size the position more conservatively.")
        )

    # Institutional / technical confluence
    if mtf:
        mtf_tfs = _f(row, "mtf_timeframes", "")
        paragraphs.append(
            f"Technical confluence: this same setup is confirmed across multiple "
            f"timeframes ({mtf_tfs}) — a stronger signal than any single "
            f"timeframe alone, since it means the pattern isn't just a short-term "
            f"artifact of one chart resolution."
        )

    # Overall conviction
    if recommend:
        paragraphs.append(
            f"Overall conviction: {rating['label']}. {rating['reason']}"
        )
    else:
        paragraphs.append(
            f"Overall conviction: {rating['label']}. {rating['reason']} "
            f"Buying is not currently recommended at this stage — "
            + (
                f"wait for a {v['base_noun']} to form and the setup to mature."
                if not has_handle else
                "wait for either a stronger confirmed breakout with volume, or a "
                "meaningfully higher quality score, before committing capital."
            )
        )

    return {"recommend": recommend, "paragraphs": paragraphs}


# ─── Full narrative explanation (strengths/weaknesses/conclusion) ─────────

def full_explanation(row: dict) -> dict:
    """
    Returns the complete AI-Explanation payload:
      {
        "why_detected": str,
        "strengths": [str, ...],
        "weaknesses": [str, ...],
        "institutional_note": str,
        "conclusion": str,
      }
    """
    v = _vocab(row)
    quality = _f(row, "quality_score", 0) or 0
    signal_type = _f(row, "signal_type", "")
    pattern_type = _f(row, "pattern_type", "")
    symbol = _f(row, "symbol", "This stock")
    depth_class = _f(row, "cup_depth_class")
    shape = _f(row, "cup_shape")
    has_handle = bool(_f(row, "has_handle", 0))
    rs_rating = _f(row, "rs_rating")
    rs_tag = _f(row, "rs_tag")
    vol_ratio = _f(row, "volume_ratio")
    readiness = _f(row, "breakout_readiness_pct")
    mtf = bool(_f(row, "mtf_confluence", 0))
    handle_quality = _f(row, "handle_quality_subscore")
    cross_pattern = bool(_f(row, "cross_pattern_confluence", 0))

    # Why detected
    why_detected = (
        f"{symbol} was flagged because its price history contains a qualifying "
        f"{v['structure_noun']} ({shape or 'unclassified shape'}) that satisfies the "
        f"scanner's one hard geometric requirement for a {v['name']}. "
    )
    if has_handle:
        why_detected += (
            f"A {v['base_noun']} has since formed, which is what promotes this from a bare "
            f"pre-confirmation detection to a full '{pattern_type}' setup."
        )
    else:
        why_detected += (
            f"No {v['base_noun']} has formed yet, so this is currently surfaced as a "
            f"'{v['preconfirm_label'].title()}' / early-watch setup rather than a tradeable pattern."
        )

    # Strengths
    strengths = []
    if quality >= 70:
        strengths.append(f"High overall quality score ({quality:.0f}/100) — the pattern's geometry, symmetry, and volume behaviour all score well together.")
    if rs_rating is not None and rs_rating >= 70:
        strengths.append(f"Strong relative strength (RS {rs_rating:.0f}{', ' + rs_tag if rs_tag and rs_tag != '-' else ''}) versus the broader NSE universe.")
    if vol_ratio is not None and vol_ratio >= 1.4:
        strengths.append(f"Volume confirms the move at {vol_ratio:.2f}× average — real participation, not a quiet drift.")
    if has_handle and handle_quality is not None and handle_quality >= 12:
        strengths.append(f"Tight, well-formed {v['base_noun']} (quality {handle_quality:.0f}/20) — volume dried up and the range contracted the way a textbook {v['base_noun']} should.")
    if readiness is not None and readiness >= 80:
        strengths.append(f"Breakout Readiness of {readiness:.0f}% — most of the confirmation factors a trader would want are already in place.")
    if mtf:
        strengths.append("Confirmed on more than one timeframe simultaneously, which meaningfully raises confidence in the pattern.")
    if cross_pattern:
        strengths.append(f"Multiple bullish pattern types line up across timeframes for this symbol ({_f(row, 'cross_pattern_desc', '')}) — independent confirmation, not just one detector's read.")
    cc_found = bool(_f(row, "confirmation_candle_found", 0))
    cc_strength = _f(row, "confirmation_candle_strength")
    cc_pattern = _f(row, "confirmation_candle_pattern")
    if cc_found and cc_strength in ("Strong", "Moderate") and cc_pattern:
        strengths.append(f"Breakout bar shows a {cc_pattern} ({cc_strength}) — real bullish conviction on the candle that matters most.")
    if not strengths:
        strengths.append("Detected within the scanner's loose geometric threshold, but doesn't yet show standout strength on any single factor.")

    # Weaknesses
    weaknesses = []
    if quality < 55:
        weaknesses.append(f"Quality score is on the low side ({quality:.0f}/100) — treat this as a loosely-qualifying detection, not a polished setup.")
    if rs_rating is not None and rs_rating < 50:
        weaknesses.append(f"Weak relative strength (RS {rs_rating:.0f}) — this stock hasn't been outperforming the broader market.")
    if vol_ratio is not None and vol_ratio < 1.0:
        weaknesses.append(f"Below-average volume ({vol_ratio:.2f}×) — lacks the participation that usually accompanies a genuine institutional move.")
    if not has_handle:
        weaknesses.append(f"No {v['base_noun']} has formed yet — entry here would be early and more speculative than waiting for {v['base_noun']} confirmation.")
    if _f(row, "cup_depth_verify_flag", 0):
        weaknesses.append(f"{v['depth_noun'].capitalize()} is very shallow — worth checking the actual chart, since normal day-to-day volatility can produce a false-positive '{v['structure_noun']}' at this depth.")
    if _f(row, "stop_loss_type") == "8pct_cap":
        weaknesses.append("Stop loss is capped at the maximum 8% — the natural stop level was wider, so this trade carries more risk per share than a tighter setup would.")
    rr_t2 = _f(row, "rr_t2")
    if rr_t2 is not None and rr_t2 < 2.0:
        weaknesses.append(f"Risk:Reward at Target 2 is only {rr_t2:.2f}:1, below the 2:1 this strategy generally prefers.")
    if cc_found is False and signal_type == "BREAKOUT NOW":
        weaknesses.append("The breakout bar itself isn't a strong bullish candle — worth seeing a stronger confirmation bar before adding size.")
    if not weaknesses:
        weaknesses.append("No major weaknesses flagged by the current rule set — the usual caveats (position sizing, market regime) still apply.")

    # Institutional note
    if vol_ratio is not None and vol_ratio >= 1.8 and rs_rating is not None and rs_rating >= 70:
        institutional_note = (
            "The combination of high relative strength and above-average volume "
            "on the move is the kind of footprint institutional accumulation "
            "tends to leave — not proof of it, but consistent with it."
        )
    elif vol_ratio is not None and vol_ratio < 1.0:
        institutional_note = (
            "Volume this light is not typically what institutional buying looks "
            "like — this move may be retail-driven or simply low-conviction so far."
        )
    else:
        institutional_note = (
            "Volume and relative strength here are inconclusive on their own for "
            "judging institutional participation — neither strongly present nor "
            "strongly absent."
        )

    # Conclusion
    rating = overall_rating(row)
    conclusion = f"{rating['label']}: {rating['reason']}"

    return {
        "why_detected": why_detected,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "institutional_note": institutional_note,
        "conclusion": conclusion,
    }
