# Roblox Up-and-Coming top-10 forecaster — build 6

A deterministic, offline forecaster for the make.fun bounty
**"Predict Roblox's Up-and-Coming top 10"**. It reads the chart history
(`history.ndjson`) and writes a top-10 forecast (`forecast.json`).

## Model

A blend of two **k-NN regressions** over chart trajectories — a
continuation of build 5's k-NN family, tuned on three windows (the
practice window plus two independent backtests):

- For each game on the closing chart, find the 50 most similar
  (game, anchor) pairs in the history — similarity in standardised
  feature space — and average what those neighbours actually did:
  their place at each lag 6–72h after the anchor. The forecast is the
  lag-averaged neighbour outcome distribution, a direct empirical
  estimate of the time-averaged place distribution the score rewards.
- **Component A** — features (log-rank, 36h log-rank velocity):
  best on the practice window (the most recent window, the best proxy
  for the final window).
- **Component B** — features (log-rank, 24h log-rank velocity,
  log-age): best on the two independent backtests.
- **Blend: 0.45 · A + 0.55 · B.** The blend beats either component
  alone on every window.

Post-processing: per-place probabilities are the blended neighbour
outcome frequencies, sharpened with temperature 1.2 and normalised to
(1 − reserve_k) per place, where the unnamed reserve falls with place
(reserve_k = 0.30 · (1 − 0.5 · (k−1) / 9)): newcomers win a larger
share of the top places, so less mass is left unnamed there.
Probabilities below 0.0005 are dropped; the top 10 is ranked by total
occupancy.

Rejected on evidence (see `BUILD_LOG.md`): a k-NN + OU-diffusion blend
(even a 10% OU weight hurts, with a properly calibrated OU), kernel
weighting inside the top-k, and per-place temperature slopes beyond the
reserve slope.

## Local scores (log score, higher is better)

| window | build 5 | build 6 |
| --- | --- | --- |
| practice | −33.29 | **−32.63** |
| backtest B | −43.54 | **−42.26** |
| backtest C | −37.66 | **−36.90** |
| mean | −38.16 | **−37.26** |

Build 6 beats build 5 on every window (+0.66 / +1.28 / +0.76, mean
+0.90) and meets the build targets (practice ≤ −33.0, mean ≤ −37.5).
Top-10 hits: 7.0 / 4.7 / 5.8; agreement: 58.8 / 27.5 / 38.7.

## Run

```bash
python3 main.py /path/to/input /path/to/output
# or with environment variables:
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

Stdlib only, no network, no randomness. Two runs on the same history
produce byte-identical output. Runtime ≈ 1–3 s per window on the
practice data (well under the 300 s limit).

## Files

- `main.py` — the entry (the only file the harness runs).
- `analysis/` — prototype and tuning scripts (`proto6.py`,
  `tune6.py`, `tune6b.py`) plus the scorer and backtest windows used
  for local evaluation. Not part of the entry.
- `out/` — local forecasts produced by `main.py` on the three windows.
- `BUILD_LOG.md` — stage-by-stage build record with scores and
  rejected ideas. `TRANSCRIPT.md` — session record.
- `LICENSE` — MIT.

---

Built with Muse for the make.fun Roblox bounty.
Referral: https://make.fun/r/2e4854b0
