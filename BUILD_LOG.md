# Build 7 — build log

**Goal (from `rebuild-brief-7.md`):** continue the k-NN family from
build 6 (practice −32.63, B −42.26, C −36.90, mean −37.26 — itself
never submittable, its session failing the server's thinking_priced
gate), beating the only valid entry, build 2 (practice −33.81,
B −48.20, C −41.49, mean −41.17). Build 7 is built by the main agent
in the main chat, one assistant message per turn, so the session
carries real per-message token usage. Targets: practice ≤ −33.0 and
mean ≤ −37.5, submit only on a genuine combined-evidence win.

**Result: targets beaten. Build 7 (main.py): practice −32.60,
B −41.71, C −35.80, mean −36.70** — beats build 6 on every window
(+0.03 / +0.55 / +1.10, mean +0.56) and build 2 by a wide margin.
Hits 7.0 / 4.7 / 5.8, agreement 59.4 / 26.9 / 38.8. Deterministic
(byte-identical repeat runs), py_compile OK, runtime 0.9–2.3 s per
window.

## Stage 1 — fresh machinery + build 6's levers re-checked (`analysis/proto7.py`)

The k-NN machinery was re-implemented generically for analysis
(velocity window, age weight, an optional log-playing feature, per-lag
weights, per-component k), without reading any earlier build's
main.py. Sanity gate: build 6's adopted configuration reproduced
exactly (blend α=0.55, reserve 0.30, T 1.2, reserve slope −0.5 →
mean −37.26, P −32.63 / B −42.28 / C −36.88; B/C within 0.02 of
build 6's entry, the known tie-order difference).

Screening post-processing for stages 1–2 is build 6's adopted
(reserve 0.30, T 1.2, reserve slope −0.5, cutoff 0.0005):

- α grid: 0.45 → −37.44, 0.50 → −37.34, 0.55 → −37.26, 0.60 → −37.25,
  0.65 → −37.23, 0.70 → −37.27. A flat plateau; α=0.65 marginally best.
- Age weight inside component B: 0.75 → −37.35, 1.25 → −37.48
  (w=1.0 stays).
- Three-way blends with a third component (vel36+age, vel24 no-age,
  vel18+age, vel30+age at weight 0.30): −37.31 … −37.52, all worse
  than the two-way blend.
- Gaussian lag weighting around 36h (σ=30 → −37.36, σ=18 → −37.41):
  rejected.
- **Log-playing feature: rejected** (A+play −38.10, B+play −37.58,
  both+play −38.34). Player counts duplicate the rank signal and add
  noise on practice.
- k: 40/40 → −37.57, 60/60 → −37.65, A40/B60 → −37.72, A60/B40 →
  −37.48. k=50 stays.

## Stage 2 — four new levers (`analysis/proto7b.py`)

- **Multi-scale velocity component** [log-rank, vel12, vel24, vel36,
  log-age]: alone −37.89, but as a third component (A 0.25 / B 0.35 /
  MS 0.40) → −37.17, and the weight refinement plateau is
  −37.15 … −37.24 (best A 0.30 / B 0.35 / MS 0.35 → −37.15). A real
  but small gain; not in the final configuration (stage 3's lever
  dominates it, and the two do not stack — smoothed three-way blends
  score −36.88 … −36.93, no better than the smoothed two-way).
- Acceleration feature (vel12 − vel36): alone −38.25, three-way
  −37.50. Rejected.
- Anchor recency (drop the oldest third of training anchors):
  −38.62, practice collapses to −37.28. Rejected — the old anchors
  carry the regime information. (Dropping half the anchors also
  exposed an empty-lag-set crash on the shortest window; that
  motivated the entry's used-lags guard below.)
- **Place smoothing — the build's main lever.** Convolving each
  game's blended per-place probabilities with a Gaussian kernel over
  places, then rescaling to the game's total: σ=0.5 → −37.08,
  σ=0.75 → −36.88, **σ=1.0 → −36.72** (P −32.67 / B −41.49 /
  C −36.00) at α=0.65 under the screening post-processing. The k-NN
  outcome counts are sparse (50 neighbours over 10 places), and the
  chart moves gradually — neighbouring places deserve mass.

## Stage 3 — sigma plateau + post-processing grid (`analysis/proto7c.py`)

- σ sweep (α=0.65): 0.75 → −36.88, 1.0 → −36.72, 1.25 → −36.89,
  1.5 → −36.91, 2.0 → −36.99. A plateau peaking at σ=1.0, not a spike.
  Smoothing before vs after blending is identical (the operator is
  linear), so the entry smooths after blending.
- Full (reserve × temperature × slopes) grid on the finalists. Top:
  **α=0.65, σ=1.0, reserve 0.25, T 1.2, reserve slope −0.5, temp
  slope 0 → mean −36.70 (P −32.60 / B −41.72 / C −35.77)** in the
  prototype; the grid neighbourhood is flat (−36.70 … −36.72 for the
  top three cells).

## Adopted configuration (the entry)

- Components: A = vel36, weights [rank 1, vel 1, age 0], blend 0.35;
  B = vel24, weights [1, 1, 1], blend 0.65. k=50, lags
  {6, 12, 24, 36, 48, 60, 72}, uniform neighbours.
- Place smoothing σ=1.0 (kernel radius 3, per-game total preserved).
- Post-processing: T 1.2 flat, reserve 0.25 with slope −0.5,
  cutoff 0.0005.

## Entry (`main.py`, fresh stdlib implementation)

- py_compile OK. Runtime ≈ 2.3 s (practice), 1.4 s (B), 0.9 s (C).
- **Scores from main.py itself: practice −32.596, B −41.713,
  C −35.798, mean −36.702** — the prototype's numbers, C within 0.03
  (stable-sort vs argsort tie order, as in build 6).
- Schema: 506 / 512 / 611 probability rows, exactly 10 top10 ids,
  per-place sums 0.74–0.87 (the reserve slope by design), file sizes
  29–35 KB (limit 1,048,576).
- Determinism: repeat practice runs byte-identical (md5 verified).
- Two genuine edits after the first write, both robustness fixes,
  neither changing any window's output (byte-identical re-runs):
  (1) lag averaging divides by the number of lags that actually have
  training pairs instead of len(LAGS); (2) a guard for histories with
  no usable snapshot pairs (a single-snapshot history previously
  crashed with a division by zero; it now emits the graceful empty
  forecast). Edge cases verified: empty history, 1 snapshot,
  3 snapshots.

## Files

- `main.py` — the entry.
- `analysis/proto7.py`, `proto7b.py`, `proto7c.py` (+ run logs) —
  prototype and tuning; `analysis/score.py`, `practice_answer.json`,
  `backtest_B/`, `backtest_C/` — local evaluation support (copied
  from build 6's analysis; not part of the entry).
- `out/` — local forecasts from `main.py` on the three windows.
- `README.md`, `LICENSE` (MIT), `BUILD_LOG.md` (this file),
  `TRANSCRIPT.md`, `.makefun/owner.json`.
- Session transcript: `~/workspace/makefun/session-build-7.jsonl`.

## What did NOT make it (and why)

- The multi-scale velocity third component (+0.09 alone, nothing
  once smoothing is applied).
- OU diffusion in any form (exhausted in builds 3–6; not retried).
- Kernel-weighted neighbours, log-playing features, acceleration,
  anchor recency, lag weighting, k ≠ 50, age weight ≠ 1.0 — each
  rejected on the three-window evidence above.
- No main.py from builds 2/3/4/5/6 was read; everything was
  re-implemented fresh from the design specs.
