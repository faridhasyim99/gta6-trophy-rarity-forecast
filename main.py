#!/usr/bin/env python3
"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's earn rate (percent of tracked players) at the
case target as five quantiles (5/25/50/75/95%).

Model, in brief: work in logit space on x = ln(1 + hours since
2026-11-19T00:00:00Z). Anchor at the last history read. If the history
is frozen (every read shows identical rates) the page is stale and a
refresh is due, so the center shifts by a calibrated dilution jump
plus gradual decline, with a per-trophy rate term; the shift is capped
at -3.0 logit. If the history moves, fit a per-trophy OLS line of
logit(rate) on x and shrink its slope toward the tier mean. The band
is a pooled residual SD plus slope uncertainty, widened when frozen,
with an asymmetric upper tail. The 100% trophy never dilutes and gets
a fixed row. Standard library only; deterministic; no clock, no
network, no files outside the case directory.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Frozen-history center shift: D = A0 + A1*dx + B_RATE*(ln r - LN_R_MED).
A0 = -0.962
A1 = -0.920
B_RATE = 0.65
LN_R_MED = -0.248
FROZEN_SHIFT_FLOOR = -3.0

# Band constants.
WIDTH = 0.7
HI_MULT = 1.4
LO_MULT = 0.75
MIN_SD = 0.15
DRIFT_SD = 0.15
FROZEN_JUMP_SD = 0.40
TIER_SHRINK_K = 3.0


def clamp_rate(p):
    """Clamp a percent rate into [0.01, 99.99] so logits stay finite."""
    return min(99.99, max(0.01, p))


def logit(p):
    c = clamp_rate(p)
    return math.log(c / (100.0 - c))


def inv_logit(y):
    return 100.0 / (1.0 + math.exp(-y))


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp (trailing Z allowed)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def x_of(ts):
    """Log-time coordinate: ln(1 + hours since the 2026-11-19 epoch)."""
    hours = (parse_time(ts) - EPOCH).total_seconds() / 3600.0
    return math.log(1.0 + hours)


def round4_half_up(v):
    """Round to 4 decimals, halves up; whole numbers come out as ints."""
    scaled = math.floor(v * 10000.0 + 0.5) / 10000.0
    if scaled == int(scaled):
        return int(scaled)
    return scaled


def ols(xs, ys):
    """Least squares of ys on xs. Returns (intercept, slope, se_slope, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0, [y - my for y in ys]
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    if n > 2:
        s2 = sum(r * r for r in residuals) / (n - 2)
        se = math.sqrt(s2 / sxx)
    else:
        se = 0.0
    return intercept, slope, se, residuals


def validate_case(case):
    """Guardrail: fail loudly on a malformed case instead of silently
    forecasting garbage. The kit guarantees the shape below, so any
    violation here means a harness or data problem, not a forecast
    decision. Kept separate from forecast_case so the forecast math
    stays a pure function of the case dict.
    """
    trophies = case["trophies"]
    history = case["history"]
    if len(trophies) != 55:
        raise ValueError(f"expected 55 trophies, got {len(trophies)}")
    if [t["id"] for t in trophies] != list(range(1, 56)):
        raise ValueError("trophy ids must be 1..55 in order")
    if not history:
        raise ValueError("history must contain at least one read")
    for h in history:
        if len(h["rates"]) != 55:
            raise ValueError("each history read must carry 55 rates")
    levels = case.get("levels", LEVELS)
    if list(levels) != LEVELS:
        raise ValueError(f"unexpected quantile levels: {levels}")


def forecast_case(case):
    validate_case(case)
    trophies = case["trophies"]
    history = case["history"]
    target = case["target"]

    n_t = len(trophies)
    tiers = [t["tier"] for t in trophies]

    # Log-time coordinate of each history read and of the target.
    xs = [x_of(h["slot"]) for h in history]
    x_last = xs[-1]
    x_target = x_of(target)
    dx = max(0.0, x_target - x_last)

    # Last non-null rate per trophy, and the full last-read rate vector.
    rates_last = []
    for ti in range(n_t):
        r = None
        for h in reversed(history):
            v = h["rates"][ti]
            if v is not None:
                r = v
                break
        rates_last.append(0.01 if r is None else r)

    frozen = len({tuple(h["rates"]) for h in history}) <= 1

    # ---- Moving branch: per-trophy OLS on log-time, slopes pooled. ----
    slopes = [0.0] * n_t
    se_b = [0.0] * n_t
    resid_sq_sum = 0.0
    resid_dof_sum = 0
    if not frozen:
        for ti in range(n_t):
            pts_x, pts_y = [], []
            for xi, h in zip(xs, history):
                v = h["rates"][ti]
                if v is not None:
                    pts_x.append(xi)
                    pts_y.append(logit(v))
            if len(set(pts_x)) >= 3 and len(pts_x) >= 3:
                _, b, se, res = ols(pts_x, pts_y)
                slopes[ti] = b
                se_b[ti] = se
                resid_sq_sum += sum(r * r for r in res)
                resid_dof_sum += len(pts_x) - 2
            else:
                slopes[ti] = 0.0
                se_b[ti] = 0.0

    # Tier mean slopes (tiers with <3 trophies fall back to the global mean).
    tier_slopes = {}
    for tier in set(tiers):
        members = [slopes[i] for i in range(n_t) if tiers[i] == tier]
        tier_slopes[tier] = sum(members) / len(members) if members else 0.0
    global_slope = sum(slopes) / n_t if n_t else 0.0
    for tier in tier_slopes:
        if sum(1 for t in tiers if t == tier) < 3:
            tier_slopes[tier] = global_slope

    # Pooled residual SD and pooled slope SE for the band.
    sd_pool = math.sqrt(resid_sq_sum / resid_dof_sum) if resid_dof_sum > 0 else 0.0
    se_pool = math.sqrt(sum(s * s for s in se_b) / n_t) if n_t else 0.0

    var_base = max(sd_pool, MIN_SD) ** 2 + (se_pool * dx) ** 2 + (DRIFT_SD * dx) ** 2
    if frozen:
        var_base += FROZEN_JUMP_SD ** 2
    sigma = WIDTH * math.sqrt(var_base)

    rows = []
    # NOTE: use enumerate explicitly here. An earlier prototype read
    # rates_last[i] with a leaked loop variable from the fits loop
    # above, which silently read trophy 55's rate for every trophy and
    # disabled the per-trophy rate term (B1 0.1602 instead of 0.1046).
    # The B1 acceptance test catches exactly this bug class.
    for ti, trophy in enumerate(trophies):
        r_last = rates_last[ti]
        y_last = logit(r_last)

        # The 100% trophy never dilutes.
        if r_last >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]})
            continue

        if frozen:
            shift = A0 + A1 * dx + B_RATE * (math.log(max(r_last, 0.01)) - LN_R_MED)
            shift = max(shift, FROZEN_SHIFT_FLOOR)
            y_hat = y_last + shift
        else:
            n_usable = sum(
                1 for h in history if h["rates"][ti] is not None
            )
            b_star = (
                n_usable * slopes[ti] + TIER_SHRINK_K * tier_slopes[tiers[ti]]
            ) / (n_usable + TIER_SHRINK_K)
            y_hat = y_last + b_star * dx

        qs = []
        for z in Z_SCORES:
            if z < 0:
                mult = LO_MULT
            elif z > 0:
                mult = HI_MULT
            else:
                mult = 1.0
            q = inv_logit(y_hat + z * sigma * mult)
            q = min(100.0, max(0.01, q))
            qs.append(round4_half_up(q))
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        text = json.dumps(result, separators=(",", ":")) + "\n"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
