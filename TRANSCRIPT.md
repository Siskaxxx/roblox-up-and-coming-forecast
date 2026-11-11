# TRANSCRIPT — build 2 of the Roblox forecaster (recorded session)

Bounty: `roblox-up-and-coming-forecast-01`. This file is the chronological build
record for build 2. The full session transcript (every substantive message and
tool call, with the complete file contents as written) is
`~/workspace/makefun/session-build-2.jsonl`, recorded live with `rec_session.py`.

## Timeline (2026-11-11, WIB)

- **~16:13** — Brief received: rebuild the forecaster fresh (never read build 1's
  `main.py`) with a live-recorded session, because build 1's transcript was lost
  to compaction and cannot pass the transcript gate. Read the brief, the recorder
  helper, `kit.json`, `RULES.txt`.
- **~16:14** — Read the design spec (build 1's BUILD_LOG.md §2–§5), `task.json`
  (34 snapshots, 2026-10-30 → 2026-11-07), the kit's `starter/main.ts` house
  baseline, and the analysis tools (`score.py`, `dynamics.py`, `kernel.py`,
  `explore.py`, `validate.sh`). Ran `explore.py` + `kernel.py`: chart 92–99
  games, gaps 3.6–9.8h, no `details` in the practice history; per-bucket
  diffusion σ 0.06–0.19/hr, momentum corr 0.211, OFF hazard ~0.05/step.
- **~16:16** — Wrote `main.py` fresh (my own implementation of the OU-diffusion
  design: 0.1-wide log-rank bins to rank 181 + OFF, runtime calibration of
  θ/μ/σ/β/hazard/newcomer shares, per-place normalisation with newcomer reserve,
  cutoff 0.003). Removed build 1's `main.py` first (its content is preserved in
  GitHub commit `90af81ef`).
- **~16:17** — Validation, first run: `py_compile` OK; practice **−33.965852**
  (typical p 0.033487, 7.30 hits, 61.70 agreement) vs baseline −43.512039;
  two runs byte-identical (deterministic); 212 rows, per-place sums 0.68–0.89,
  12,257 bytes; runtime ~3 s.
- **~16:18** — Backtests: B **−48.41** (baseline −56.33), C **−42.03** (baseline
  −52.71). Mean across practice+B+C −41.47, matching build 1's −41.50.
  Edge cases: single-snapshot history and two-snapshot history with no `details`
  key both produce valid forecasts (fallbacks engage).
- **~16:20** — Review fix to `velocity()`: it had measured momentum over the
  oldest observation ≥36h back instead of the last 36h. Fixed to interpolate
  log-rank at exactly (close − 36h). Re-validation improved every window:
  practice **−33.813982**, B **−48.199901**, C **−41.491403** (mean −41.17).
  Deterministic and edge cases still pass. Updated BUILD_LOG.md and this file
  to the final scores.
- **~16:19** — Wrote the remaining entry files: `README.md` (requirements
  template, referral code `2e4854b0` in the footer), `LICENSE` (copied verbatim
  from the kit starter, sha256 verified), `.makefun/owner.json`, this file, and
  `BUILD_LOG.md`.

## Design decisions (mine, from the spec)

- OU diffusion over log-rank rather than a random walk: the history shows
  variance saturating with lag and autocorrelation decaying — mean reversion, not
  a random walk. A plain diffusion spreads far too wide over 72h.
- Full per-game distributions (not point forecasts): the scorer rewards honest
  probabilities, and the chart churns hard — the spread is the forecast.
- Calibrated defaults, no practice tuning: tuning on the 10-read practice window
  overfits (build 1's tuned configs lost 1–3 nats on the independent backtests).
- Newcomer reserve left unnamed rather than spread over guessed games: brand-new
  entrants are unknowable from history; the unnamed share is the honest
  probability for them.

## Verification summary

| check | result |
|---|---|
| practice log score | −33.813982 (baseline −43.512039) |
| backtest B / C | −48.199901 / −41.491403 (baselines −56.33 / −52.71) |
| determinism | two runs byte-identical |
| shape | exactly {probabilities, top10}, 10 distinct top10, per-place sums ≤ 1, 198 rows, 11,457 bytes |
| edge cases | single snapshot + no-details history both valid |
| runtime | ~3 s (limit 300 s) |
| Python 3.13 | not runnable locally (VM has 3.12.3); code uses no 3.13-only features, compiles cleanly |
