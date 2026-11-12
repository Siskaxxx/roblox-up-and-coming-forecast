# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Siskaxxx/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Siskaxxx/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Build 7: a deterministic, offline forecaster for the make.fun bounty
"Predict Roblox's Up-and-Coming top 10". It reads the chart history
(`history.ndjson`) and writes a place-by-place top-10 forecast
(`forecast.json`).

Model — two k-NN regression components over historical chart
trajectories, blended 0.35 / 0.65, followed by the step that is new in
this build: **place smoothing**. For each game on the closing chart,
the components find the 50 most similar (game, anchor) pairs in the
history — similarity in standardised (log-rank, log-rank velocity,
log-age) feature space — and average what those neighbours actually
did at lags of 6–72 hours. Component A uses a 36h velocity and no age
feature; component B uses a 24h velocity plus log-age. The blended
per-place probabilities are then convolved with a Gaussian kernel
(sigma = 1.0 place), because the neighbour outcome counts are sparse
and the chart moves gradually: mass belongs on neighbouring places
too. On the three local windows (practice + two backtests) the
smoothing is worth about +0.5 of mean log score over build 6, almost
all of it in the backtests.

Post-processing: per place k, probabilities are tempered (T = 1.2)
and normalised to (1 − reserve_k), reserve_k = 0.25 · (1 − 0.5 ·
(k−1) / 9); probabilities below 0.0005 are dropped and the top 10 is
ranked by total occupancy.

Local scores (log score, higher is better): practice **−32.60**,
backtest B **−41.71**, backtest C **−35.80**, mean **−36.70** —
build 6 scored −37.26 mean, build 2 (the previous valid entry)
−41.17. Standard library only, no network, no randomness; two runs
on the same history are byte-identical, and the runtime is about
1–3 s per window. See `BUILD_LOG.md` for the full tuning record,
including everything that was tried and rejected.

## Run it

```sh
python3 main.py /path/to/input /path/to/output
# or: MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=2e4854b0&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
