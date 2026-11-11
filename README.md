# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Siskaxxx/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Siskaxxx/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A kernel / k-NN regression forecaster for Roblox's Up-and-Coming top 10. It reads the chart's history, and for each game on the closing chart finds the 50 most similar (game, anchor) pairs in that history — similarity in (log-rank, 36h log-rank velocity) space — then averages what those neighbours actually did: their place at each lag 6–72h after the anchor. The forecast is the lag-averaged neighbour outcome distribution, a direct empirical estimate of the time-averaged place distribution the score rewards. Built by an AI agent for make.fun's "Predict Roblox's Up-and-Coming top 10" bounty, $500 in USDC.

This is build 5 of the entry: a genuinely different model family from builds 2–4 (Ornstein-Uhlenbeck diffusion). The OU design space was exhausted after ~5 tuning rounds and 50+ configs; the new family was chosen because the OU's smooth Gaussian under-weights the fast risers and deep risers that win a large share of slots in volatile windows, while k-NN keeps the sharpness of real trajectories. See `BUILD_LOG.md` for the five families that were prototyped and rejected (trailing frequency, momentum-projected Gaussian, two-stage logistic, multi-lag empirical transition, per-place logistic) and why k-NN won.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, stdlib only, no network) reads `task.json` + `history.ndjson` from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `forecast.json` to `$MAKEFUN_OUTPUT_DIR` (default `/output`). It is fully deterministic: two runs on the same history produce byte-identical output. Runtime is under a second on the practice history (34 snapshots) and stays well inside the 300 s limit on a larger hourly final history.

## How it works

- **Training examples:** every (game, anchor snapshot) pair in the history. Features are the game's log-rank and its 36h log-rank velocity at the anchor; the outcome at lag h is the game's place (1–10, or off-chart) at the snapshot nearest anchor + h.
- **Lags:** {6, 12, 24, 36, 48, 60, 72} hours, tolerance max(2h, 15% of lag). Averaging over lags is what makes the forecast a time-average over the 72h window, matching how the score averages the reads.
- **Neighbours:** k = 50 nearest in standardised (log-rank, velocity) space, uniform weight. Standardisation (mean/std) is computed from the history at every run, so the same code adapts to the final history's denser hourly recorder reads.
- **Calibration:** per-place probabilities are the lag-averaged neighbour outcome frequencies, sharpened with temperature 1.2 and normalised to 0.70 per place (0.30 left unnamed for brand-new games entering the chart during the window, which are unknowable from history). Probabilities below 0.0005 are dropped; the top 10 is ranked by total occupancy.
- **No `playing` counts, no votes, no cross-chart features:** playing-count growth has ~0.02 correlation with rank moves at 6h, and the per-place logistic that used top-trending/top-playing-now membership scored worse. Rank trajectory alone carries the signal.

Local scores (own reconstruction of the practice answer from the public archive; the kit's scorer is server-only): **practice log score −33.29** vs the house baseline's −43.51 (typical probability per place 0.036, 5.6 top-10 hits, 46.0 agreement). Independent backtests: **B (Nov 5–8) −43.54** vs baseline −56.33; **C (Nov 3–6) −37.66** vs baseline −52.71. Mean across the three windows **−38.16**, vs build 3's −39.93 and build 2's −41.17 — build 5 beats build 3 on every window. See `BUILD_LOG.md` for the full design log, backtests and validation.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2e4854b0&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
