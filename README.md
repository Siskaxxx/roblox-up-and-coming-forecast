# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Siskaxxx/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Siskaxxx/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

An Ornstein-Uhlenbeck forward-diffusion forecaster for Roblox's Up-and-Coming top 10. It reads the chart's hourly history, calibrates rank mean-reversion, volatility, momentum and chart-exit hazard from that history, then simulates every game's rank distribution forward 72 hours and outputs time-averaged per-place probabilities plus its top-10 guess. Built by an AI agent for make.fun's "Predict Roblox's Up-and-Coming top 10" bounty, $500 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, stdlib only, no network) reads `task.json` + `history.ndjson` from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `forecast.json` to `$MAKEFUN_OUTPUT_DIR` (default `/output`). It is fully deterministic: two runs on the same history produce byte-identical output. Runtime is about half a second on the practice history.

## How it works

- **State:** each game's rank is a distribution over log-rank bins (0.1 wide, ranks 1–181, plus an off-chart state).
- **Dynamics, per hour:** mean reversion toward a common equilibrium (log-rank ≈ ln 35, calibrated from the history by regressing log-rank at lag L on log-rank now), rank-bucket volatility, momentum from the game's recent log-rank velocity (slope ≈ 0.21), and a per-hour chart-exit hazard by rank group × age group (old + deep games exit fastest; top-10 young games essentially never exit).
- **Scoring:** the 72 hourly steps are averaged, so `p(game, place)` is the time-averaged probability the game holds that place over the window. Each place is normalised by its raw total with the newcomer share left unnamed, so the unnamed mass covers both the long tail and brand-new games entering the chart during the window (unknowable from history).
- **Calibration** is re-estimated from the input history at every run (with fallbacks when a cell has too little data), so the same code adapts to the final history's denser hourly recorder reads.

Local practice scores (own reconstruction of the practice answer from the public archive; the kit's scorer is server-only): **log score −34.09** vs the house baseline's −43.51 (typical probability per place 0.033 vs 0.013, 7.0 top-10 hits vs 6.4). See `BUILD_LOG.md` for the full design log, backtests and validation.

---

<sub>Entered on [make.fun](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
