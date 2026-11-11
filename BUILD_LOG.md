# Build 5 — k-NN regression forecaster (roblox-up-and-coming-forecast-01)

## Verdict

**Build 5 beats build 3 on the combined evidence and is the submission candidate.**
It is a genuinely different model family (kernel / k-NN regression over
(log-rank, velocity)), not an OU re-tune, temperature sweep, or ensemble.

| window | build 2 (submitted) | build 3 | build 4 | **build 5** | house baseline |
|---|---|---|---|---|---|
| practice (Nov 8–10) | −33.81 | −33.89 | −35.21 | **−33.29** | −43.51 |
| backtest B (Nov 5–8) | −48.20 | −46.77 | −45.09 | **−43.54** | −56.33 |
| backtest C (Nov 3–6) | −41.49 | −39.14 | −38.54 | **−37.66** | −52.71 |
| **mean** | −41.17 | −39.93 | −39.62 | **−38.16** | −50.85 |

Build 5 beats build 3 on **every** window (practice +0.60, B +3.23, C +1.48)
and on the mean (+1.77). It meets the brief's targets: practice −33.29 ≤ −33.5,
and B/C are far better than build 3's. Typical probability per place 0.036
(practice), 5.6 top-10 hits, 46.0 agreement. Oracle ceilings for reference:
practice −16.62, B −24.82, C −18.16 (unknowable newcomers dominate the gap).

## Why a new family

Builds 3–4 ran ~5 tuning rounds and 50+ configs in the OU-diffusion space.
The OU's smooth Gaussian is well matched to calm windows but under-weights
the fast risers and deep risers that win a large share of slots in volatile
windows (practice: close rank 29 wins place 1 twice and place 2 four times;
8/100 slots are newcomers; B: 53% of winner slots are deep risers). The
diagnosis from build 4: the loss concentrates in the unnamed mass and in
mis-calibrated per-place probabilities, not in game selection (7.3/10 hits).

## Families prototyped and rejected (all scored on practice + B + C)

| family | practice | B | C | mean | why rejected |
|---|---|---|---|---|---|
| trailing-72h empirical frequency (shrunk to closing prior) | −39.27 | −50.26 | −43.64 | −44.40 | stale: the chart moves too fast for a trailing average |
| momentum-projected Gaussian (no diffusion) | −39.14 | −46.39 | −43.12 | −42.88 | single-horizon projection lacks time-averaging |
| two-stage: P(top10) logistic × P(place\|top10) | −39.30 | −44.91 | −40.09 | −41.43 | closest prototype, still 5.4 worse on practice |
| multi-lag empirical transition (EMP-TA, rank×velocity buckets) | −36.97 | −47.26 | −44.13 | −42.78 | coarse buckets lose per-game resolution |
| per-place binary logistic (ln rank, velocity, ln age, top-trending, top-playing-now, horizon) | −39.32 | −43.49 | −40.04 | −40.95 | beats OU on B (+3.28) but spreads mass too broadly for practice (−5.43); named winners mean ln p −3.830 vs OU's −2.654 |
| **k-NN over (ln rank, velocity)** | **−33.29** | **−43.54** | **−37.66** | **−38.16** | **adopted** |

The common failure of the first five: they spread mass too broadly for the
practice window, which is concentrated (winners mostly close ranks 1–13 plus
a few deep risers). k-NN keeps the sharpness because neighbours are real
games with real trajectories.

## The k-NN model

- **Training examples:** every (game, anchor snapshot) pair in the history.
  Features: the game's log-rank and its 36h log-rank velocity at the anchor
  (velocity = slope of log-rank over the last 36h, interpolated).
  Outcome at lag h: the game's place (1–10) at the snapshot nearest
  anchor + h, or off-chart.
- **Lags:** {6, 12, 24, 36, 48, 60, 72}h, tolerance max(2h, 15% of lag).
  Averaging over lags is the time-averaging the score rewards (the score
  averages reads spread over 72h). Adding the 6h lag was a clear win
  (mean −38.53 → −38.23 at k=50); dropping the 12h lag hurt (−39.08).
- **Neighbours:** k = 50 nearest in standardised (log-rank, velocity) space,
  uniform weight. Standardisation is computed from the history per run.
  Gaussian-kernel weighting over all examples was much worse (−41.30 to
  −42.82) — the long tail of distant examples adds noise.
- **Features tried and dropped:** age (ln hours since first seen) hurt
  practice at vel36 (−35.24 vs −34.12) and only helped in combination with
  a 24h velocity (mean −37.52) at the cost of practice (−34.67) — rejected
  because the practice window is the most recent and the best proxy for the
  final; the chosen config beats build 3 on every window instead.
  `playing`-count growth is useless (corr 0.018 with rank moves at 6h).
- **Post-processing:** per-place probabilities = lag-averaged neighbour
  outcome frequencies, temperature 1.2 (sharpen), normalised to 0.70 per
  place (0.30 unnamed reserve), cutoff 0.0005, top 10 by total occupancy.

## Tuning robustness

- 319/350 configs in the (k, weighting, reserve, temperature) sweep beat
  build 3's mean. The chosen config (k=50, unweighted, res 0.30, T 1.2) is
  central, not a knife-edge: the mean is −38.14 to −38.43 across the whole
  T ∈ [1.0, 1.5] × reserve ∈ [0.25, 0.35] grid at k=50.
- The (velocity 24h + age) variant has a better mean (−37.52) but loses
  practice by 0.78 vs build 3; the chosen (velocity 36h, no age) config
  wins practice and is reported as the primary. If the parent prefers the
  higher-mean variant, it is one config change away (vel_h=24, use_age=True,
  res 0.25, T 1.1 → −34.67 / −41.44 / −36.46, mean −37.52).

## Validation

- `main.py` (228 lines, stdlib only, no network) re-implemented fresh from
  this design; it reproduces the tuned scores exactly (practice −33.29,
  B −43.54, C −37.66; the 0.05–0.06 deltas vs the numpy prototype on B/C are
  last-ulp neighbour-ordering differences between heapq and numpy argsort).
- Determinism: two runs on the practice input produce byte-identical
  `forecast.json` (verified by diff).
- Format: exactly the keys `probabilities` + `top10`; 403 probability rows;
  per-place sums exactly 0.70; top10 has 10 unique ids; all p in (0, 1].
- Runtime: 0.45 s (practice), 0.30 s (B), 0.23 s (C) — far inside the 300 s
  limit; the neighbour search is O(close games × lags × training examples)
  and scales comfortably to a larger hourly final history.
- No `playing` counts, votes, cross-chart features, or `details` are used;
  only rank trajectories from `history.ndjson`.

## Files

- `main.py` — the entry (stdlib only).
- `README.md` — entry description.
- `LICENSE` — MIT, verbatim from the kit starter.
- `BUILD_LOG.md` — this file.
- `TRANSCRIPT.md` — session summary.
- `.makefun/owner.json` — payout wallet FsUBASCTuJDbtZabdoNMj2SrvseMTtqncW699adHaJ54.
- `analysis/` — NOT part of the entry: `score.py`, `practice_answer.json`,
  backtest B/C inputs + answers, and the prototype/tuning scripts
  (`proto.py`, `proto2.py`, `proto3.py`, `proto4.py`, `tune_knn.py`,
  `tune_knn2.py`, `tune_knn3.py`).
- `out/` — local forecast outputs for practice / B / C (NOT part of the entry).

## Recommendation

Submit build 5 (replacing build 2) if the user approves: it beats build 3 on
every window and on the mean, meets the practice ≤ −33.5 target, and the
advantage is robust across the tuning grid (319/350 configs beat build 3's
mean). Build 3 remains the fallback. Do NOT submit without the user's tap.
Entries close 2026-11-13 06:00 UTC (13:00 WIB); 1 of 5 daily submissions
used (build 2).
