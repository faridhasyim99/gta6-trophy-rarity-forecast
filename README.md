# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/faridhasyim99/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/faridhasyim99/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one:
for each of the 55 trophies, five quantiles (5/25/50/75/95%) of the
earn rate (percent of tracked players) at the case target.

## How it works

All work happens in logit space on log-time
`x = ln(1 + hours since 2026-11-19T00:00:00Z)`, anchored at the last
history read:

- **Frozen history** (every read shows identical rates — the page is
  stale and a refresh is due): the center shifts by a calibrated
  dilution jump plus gradual decline,
  `D = -0.962 - 0.920*dx + 0.65*(ln r - (-0.248))`, capped at -3.0
  logit. Trophies more players have already earned dilute less.
- **Moving history** (rates change across reads): per-trophy OLS of
  logit(rate) on x, with the slope shrunk toward the tier mean
  (pseudo-count 3; tiers with fewer than 3 trophies use the global
  mean slope).
- **The 100% trophy** never dilutes: fixed row
  `[99.9, 99.99, 100, 100, 100]`.
- **Band**: pooled residual SD plus slope uncertainty, widened when
  frozen (jump-magnitude uncertainty), width 0.7, asymmetric
  (1.4 upper / 0.75 lower). Quantiles are rounded half-up to 4
  decimals and forced non-decreasing.

## Validation (exact pinball scorer, logit scale)

| Check | Result | Bar |
|---|---|---|
| B1 — real live read (p01 history → 08:57Z page) | loss **0.1046**, typical miss 0.033pp, coverage_90 0.98 | ≤ 0.11 |
| B0 p00 / p01 vs interpolated 04:30Z estimates | loss **0.0962 / 0.0915**, coverage_90 1.00 | ≤ 0.11 |
| Flat baseline (kit starter) on B1 | loss 1.4336 | — |
| Moving-history path (synthetic 20-read case) | valid, < 1 s | — |
| Long-horizon path (target 2026-11-26) | valid; -3.0 cap binds (id 3 median 0.063) | — |
| Format | 55 rows, ids 1..55, monotone, [0,100], ~2.8 KB | ≤ 65,536 B |
| Determinism | byte-identical across repeated runs | — |
| Runtime | ~0.08 s/case | cap 8 s |

## Usage

```sh
python3 -P -s main.py
```

Reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes one `*.out` per case to `$MAKEFUN_OUTPUT_DIR`
(default `/output`). Python 3.13, standard library only, no network,
no clock — output is deterministic.

## License

MIT — see `LICENSE`.
