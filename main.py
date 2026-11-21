"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01), v2.

Model (see README for the full write-up): work in logit space, anchor the
median at the last history read, and put an asymmetric band around it.
The centre is branch-specific in how it extrapolates, and the band is
branch-specific in its skew:

* frozen history (every read shows identical rates -- the page is stale
  and a refresh is due): the centre jumps by a calibrated refresh shift
  D = -0.962 - 0.920*dx + 0.65*(ln(r_last) - (-0.248)), capped at -3.0,
  and the band is upper-heavy (hi/lo 1.4/0.75).
* moving history (rates actually change across reads): the centre is an
  OLS logit-linear extrapolation with the slope shrunk toward the tier
  mean (pseudo-count K=3), and the band is lower-heavy (hi/lo 1.0/1.2),
  because dilution dominates the next move and the truth sits below the
  median far more often than above it.

A trophy already at >= 99.99% never dilutes and gets a fixed row.
Standard library only; deterministic; no network; no clock.

v2 change vs v1 (validated 21 Nov on two real moving backtests): the
prediction band is now branch-specific. v1 used the frozen pair
(1.4 upper / 0.75 lower) for both branches, which is exactly backwards
for the moving branch -- on B2 the shared pair scored 1.3785 and the
moving pair (1.0 upper / 1.2 lower) scored 1.2917; on B3 it went from
0.0748 to 0.0673. The frozen cases are byte-identical to v1
(B1 0.1046, B0 0.0962 / 0.0915) because the frozen branch is untouched.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZS = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Frozen-centre constants (calibrated on the 04:30Z refresh + 08:57Z live read).
A0 = -0.962
A1 = -0.920
B_RATE = 0.65
LN_R_MED = -0.248
SHIFT_FLOOR = -3.0

# Band constants.
K_TIER = 3          # tier-shrinkage pseudo-count for the moving slope
WIDTH = 0.7         # overall band width multiplier
MIN_SD = 0.15       # floor on the pooled residual SD
DRIFT = 0.15        # extra per-log-hour slope uncertainty
JUMP_SD = 0.40      # extra jump-magnitude uncertainty (frozen only)

# Branch-specific band skew. Frozen errors are upper-heavy (a few easy
# trophies barely dilute after a refresh); moving errors are lower-heavy
# (dilution dominates the next move). Do NOT merge the two pairs.
FROZEN_HI = 1.4
FROZEN_LO = 0.75
MOVING_HI = 1.0
MOVING_LO = 1.2

PERFECT_ROW = [99.9, 99.99, 100, 100, 100]


def clamp_rate(p):
    """Clamp a percent into [0.01, 99.99] so logits stay finite."""
    return min(99.99, max(0.01, p))


def logit(p):
    c = clamp_rate(p)
    return math.log(c / (100.0 - c))


def inv_logit(y):
    return 100.0 / (1.0 + math.exp(-y))


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def x_of(moment):
    """Log-time coordinate: ln(1 + hours since the 2026-11-19 epoch)."""
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 1e-9))


def round4(value):
    """Round to 4 decimals, half-up; whole numbers come back as ints."""
    scaled = math.floor(value * 10000.0 + 0.5) / 10000.0
    if scaled == int(scaled):
        return int(scaled)
    return scaled


def ols(points):
    """Least-squares line through (x, y) points.

    Returns (slope, intercept, residuals, se_b2) or None when the fit is
    not identified (fewer than 3 points, or fewer than 3 distinct x).
    se_b2 is the squared standard error of the slope.
    """
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    if len(set(xs)) < 3:
        return None
    n = len(points)
    mx = sum(xs) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [p[1] - (intercept + slope * p[0]) for p in points]
    dof = n - 2
    s2 = sum(r * r for r in residuals) / dof if dof > 0 else 0.0
    se_b2 = s2 / sxx if sxx > 0.0 else 0.0
    return slope, intercept, residuals, se_b2


def validate_rows(rows, n_trophies):
    """Guardrail: enforce the output contract before anything is written.

    Raises ValueError on any violation, so a malformed case fails loudly
    instead of shipping a bad file. Checks: exactly one row per trophy,
    ids 1..N each once, 5 finite quantiles in [0, 100], non-decreasing.
    """
    if len(rows) != n_trophies:
        raise ValueError("expected %d rows, got %d" % (n_trophies, len(rows)))
    seen = set()
    for row in rows:
        tid = row["id"]
        if tid in seen:
            raise ValueError("duplicate trophy id %r" % (tid,))
        seen.add(tid)
        q = row["q"]
        if len(q) != len(LEVELS):
            raise ValueError("trophy %r: expected %d quantiles" % (tid, len(LEVELS)))
        for value in q:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("trophy %r: non-finite quantile %r" % (tid, value))
            if value < 0.0 or value > 100.0:
                raise ValueError("trophy %r: quantile %r out of [0, 100]" % (tid, value))
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                raise ValueError("trophy %r: quantiles decrease" % (tid,))
    if seen != set(range(1, n_trophies + 1)):
        raise ValueError("trophy ids are not 1..%d" % n_trophies)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    target = parse_time(case["target"])
    x_target = x_of(target)

    # Frozen = every read shows the identical rates vector (stale page).
    frozen = True
    if history:
        first_rates = history[0].get("rates")
        for read in history[1:]:
            if read.get("rates") != first_rates:
                frozen = False
                break
    else:
        frozen = True

    if history:
        x_last = x_of(parse_time(history[-1]["slot"]))
    else:
        x_last = x_target
    dx = max(0.0, x_target - x_last)

    # Per-trophy usable points: (x, logit(rate)) for non-null rates.
    # The explicit enumerate index below is load-bearing: a leaked loop
    # variable here once silently read the wrong trophy's last rate and
    # disabled the per-trophy rate term in the frozen centre.
    points = []
    last_rate = []
    last_y = []
    for idx, trophy in enumerate(trophies):
        pts = []
        rate = None
        for read in history:
            rates = read.get("rates") or []
            r = rates[idx] if idx < len(rates) else None
            if r is not None:
                pts.append((x_of(parse_time(read["slot"])), logit(r)))
                rate = r
        points.append(pts)
        last_rate.append(rate)
        last_y.append(logit(rate) if rate is not None else logit(0.01))

    # Per-trophy OLS fits (moving branch) and tier/global mean slopes.
    fits = [ols(pts) for pts in points]
    slopes = [f[0] if f is not None else None for f in fits]
    known_slopes = [s for s in slopes if s is not None]
    global_mean = sum(known_slopes) / len(known_slopes) if known_slopes else 0.0
    tiers = {}
    for idx, trophy in enumerate(trophies):
        tiers.setdefault(trophy["tier"], []).append(idx)
    tier_mean = {}
    for tier, members in tiers.items():
        member_slopes = [slopes[i] for i in members if slopes[i] is not None]
        if len(members) >= 3 and member_slopes:
            tier_mean[tier] = sum(member_slopes) / len(member_slopes)
        else:
            tier_mean[tier] = global_mean

    # Pooled residual SD and pooled slope SE (band inputs).
    ss_res = 0.0
    dof_total = 0
    se_terms = []
    for idx in range(len(trophies)):
        fit = fits[idx]
        if fit is None:
            continue
        n_t = len(points[idx])
        if n_t <= 2:
            continue
        ss_res += sum(r * r for r in fit[2])
        dof_total += n_t - 2
        se_terms.append(fit[3])
    sd_pool = math.sqrt(ss_res / dof_total) if dof_total > 0 else 0.0
    se_pool = math.sqrt(sum(se_terms) / len(se_terms)) if se_terms else 0.0

    var = max(sd_pool, MIN_SD) ** 2 + (se_pool * dx) ** 2 + (DRIFT * dx) ** 2
    if frozen:
        var += JUMP_SD ** 2
    sigma = WIDTH * math.sqrt(var)

    if frozen:
        hi_mult, lo_mult = FROZEN_HI, FROZEN_LO
    else:
        hi_mult, lo_mult = MOVING_HI, MOVING_LO

    rows = []
    for idx, trophy in enumerate(trophies):
        rate = last_rate[idx]
        if rate is not None and rate >= 99.99:
            rows.append({"id": trophy["id"], "q": list(PERFECT_ROW)})
            continue
        y_last = last_y[idx]
        if frozen:
            r = max(rate if rate is not None else 0.01, 0.01)
            shift = A0 + A1 * dx + B_RATE * (math.log(r) - LN_R_MED)
            shift = max(shift, SHIFT_FLOOR)
            y_hat = y_last + shift
        else:
            fit = fits[idx]
            slope = fit[0] if fit is not None else 0.0
            n_t = len(points[idx])
            b_star = (n_t * slope + K_TIER * tier_mean[trophy["tier"]]) / (n_t + K_TIER)
            y_hat = y_last + b_star * dx
        qs = []
        for z in ZS:
            if z < 0:
                mult = lo_mult
            elif z > 0:
                mult = hi_mult
            else:
                mult = 1.0
            q = inv_logit(y_hat + z * sigma * mult)
            qs.append(round4(min(100.0, max(0.01, q))))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda row: row["id"])
    validate_rows(rows, len(trophies))
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
