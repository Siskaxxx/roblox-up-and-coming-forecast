# Build 5 transcript summary

Build 5 of the Roblox Up-and-Coming forecaster (bounty
roblox-up-and-coming-forecast-01), built in a recorded session
(`~/workspace/makefun/session-build-5.jsonl`).

## What was built

A kernel / k-NN regression forecaster — a genuinely different model family
from the OU-diffusion builds 2–4. For each game on the closing chart, it
finds the 50 most similar (game, anchor) pairs in the history (similarity in
log-rank and 36h log-rank velocity) and averages what those neighbours
actually did at lags 6–72h. Forecast = lag-averaged neighbour outcome
distribution. Post-processing: temperature 1.2, 0.30 unnamed reserve per
place, cutoff 0.0005, top 10 by total occupancy.

## Scores (local, practice + two independent backtests)

| window | build 5 | build 3 | build 2 (submitted) |
|---|---|---|---|
| practice | −33.29 | −33.89 | −33.81 |
| B | −43.54 | −46.77 | −48.20 |
| C | −37.66 | −39.14 | −41.49 |
| mean | −38.16 | −39.93 | −41.17 |

Build 5 beats build 3 on every window and on the mean (+1.77), meets the
practice ≤ −33.5 target, and is the submission candidate (pending the
user's tap). Five other new families were prototyped and rejected on the
evidence (trailing frequency, momentum-projected Gaussian, two-stage
logistic, multi-lag empirical transition, per-place logistic); the
higher-mean velocity-24h + age variant (−37.52 mean) was noted but not
adopted because it loses practice by 0.78.

## Validation

Deterministic (two runs byte-identical), format valid (403 rows, per-place
sums exactly 0.70, top10 10 unique ids), runtime < 0.5 s per window,
stdlib only, no network. `main.py` was re-implemented fresh; it was not
copied from builds 2–4 (their `main.py` files were never read).
