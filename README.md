# GTA VI Trophy Rarity Forecast — v2

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/faridhasyim99/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/faridhasyim99/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — an entry for the
[make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250, 20 places). `main.py` (Python 3.13, standard library only) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `*.out` per
case to `$MAKEFUN_OUTPUT_DIR` (else `/output`): 55 trophies, five quantiles
(5/25/50/75/95%) each, in percent.

## The model

Everything happens in logit space. For each trophy the median anchors at the
last history read and extrapolates in log-time `x = ln(1 + hours since
2026-11-19T00:00Z)` to the target; the band is an asymmetric spread around
that median, rounded half-up to 4 decimals.

**Centre — two branches:**

- **Frozen history** (every read shows identical rates, so the page is stale
  and a refresh is due): the centre jumps by a calibrated refresh shift
  `D = -0.962 - 0.920·dx + 0.65·(ln(r_last) - (-0.248))`, capped at `-3.0`
  logit. A refresh dilutes rates (the tracked-player count jumps), and
  trophies more players have already earned dilute less.
- **Moving history** (rates change across reads): per-trophy OLS of
  logit(rate) on `x`, with the slope shrunk toward the tier mean
  (`b* = (n·b + 3·b_tier)/(n + 3)`); centre `= y_last + b*·dx`.
- A trophy already at ≥ 99.99% never dilutes: fixed row
  `[99.9, 99.99, 100, 100, 100]`.

**Band — branch-specific skew (the v2 change):**

The spread is `sigma = 0.7 · sqrt(max(sd_pool, 0.15)² + (se_pool·dx)² +
(0.15·dx)²)` plus `0.40²` when frozen, with the quantile multipliers
depending on the branch and the sign of `z`:

| branch | lower tail (z < 0) | upper tail (z > 0) |
|--------|-------------------|-------------------|
| frozen | 0.75 | 1.4 |
| moving | 1.2 | 1.0 |

The two branches have opposite error skew, so they need different pairs.
After a refresh most trophies dilute but a few easy ones barely do, so
frozen errors are upper-heavy (long upper tail). In the moving branch the
centre extrapolates a jump and dilution dominates the next move — in the
B2 backtest 43 of 55 trophies fell while only 9 rose — so moving errors
are lower-heavy. v1 used the frozen pair (1.4/0.75) for both branches,
which is exactly backwards for the moving branch. Giving the moving
branch its own pair (1.0/1.2) improved B2 from 1.3785 to 1.2917 and B3
from 0.0748 to 0.0673, while the frozen cases stayed byte-identical
(B1 0.1046, B0 0.0962/0.0915). Everything else is unchanged from v1:
same frozen centre, same OLS slope + tier shrinkage (K=3), same width
0.7, same 100% special case.

## Validation (exact scorer, mean pinball loss on the logit scale)

| check | loss | reference |
|-------|------|-----------|
| B1 — live 08:57Z read (frozen) | 0.1046 | ≤ 0.11 |
| B0 p00 / p01 — interpolated 04:30Z estimates | 0.0962 / 0.0915 | ≤ 0.11 each |
| B2 — moving backtest vs 19:32Z live read | 1.2917 | beats v1's 1.3785 |
| B3 — moving backtest vs 00:02Z live read | 0.0673 | beats v1's 0.0748 |
| Combined mean of the five | 0.3302 | beats v1's 0.3491 |

Format: 55 rows, ids 1–55 once, quantiles finite in [0, 100] and
non-decreasing, ~2.8 KB per file. Deterministic (two runs byte-identical).
Runtime ~0.06 s per case (cap 8 s). A `validate_rows` guardrail enforces
the output contract before anything is written.
