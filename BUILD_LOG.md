# Build 6 — build log

**Goal (from `rebuild-brief-6.md`):** continue the k-NN family from
build 5 (submitted, practice −33.29, B −43.54, C −37.66, mean −38.16).
Do NOT go back to OU diffusion. Levers in order: (a) the unadopted
24h-velocity + age variant, (b) k-NN tuning on the mean, (c) a k-NN +
OU blend only if it improves the mean without hurting practice.
Targets: practice ≤ −33.0 and mean ≤ −37.5. Submit only if build 6
beats build 5 on the combined evidence.

**Result: targets met. Build 6 (main.py): practice −32.63, B −42.26,
C −36.90, mean −37.26** — beats build 5 on every window
(+0.66 / +1.28 / +0.76, mean +0.90). Hits 7.0 / 4.7 / 5.8,
agreement 58.8 / 27.5 / 38.7. Deterministic (byte-identical repeat
runs), py_compile OK, runtime ≈ 1–3 s per window.

## Stage 1 — prototype + lever (a) (`analysis/proto6.py`)

Re-implemented the k-NN machinery fresh from the design specs
(rebuild-brief-5.md + build 5's BUILD_LOG/TRANSCRIPT): build_sets with
feature weights after standardisation, forecast6, eval, blend, plus a
fresh OU from first principles for the later blend test.

- Baseline reproduced exactly: vel36 no-age −33.29 / −43.59 / −37.60,
  mean −38.16 (B/C differ 0.05/0.06 from build 5's reported numbers —
  known heapq vs numpy argsort last-ulp differences).
- vel24 no-age: −34.84 / −42.93 / −36.60, mean −38.13.
- vel36 + age w=1: −34.42 / −43.83 / −37.36, mean −38.53.
- **vel24 + age w=1: −34.67 / −41.44 / −36.46, mean −37.52**
  (reproduced the unadopted variant). Age scaling w=0.25/0.5/0.75 does
  NOT recover practice (P stays −34.6..−34.8 at vel24; vel36+age w=0.25:
  P −33.36, mean −38.27). The trade-off is real: the age feature helps
  the backtests a lot and hurts practice a lot.
- **Blends (the resolution):** base + vel24age α=0.25: −33.15 / −43.37 /
  −37.62, mean −38.05; **α=0.5: −32.98 / −42.15 / −37.11, mean −37.41,
  hits 7.0/4.7/5.8 — beats build 5 on EVERY window and meets both
  targets;** α=0.75: −33.65 / −41.25 / −36.89, mean −37.26.
  Blend base+vel24 (no age) α=0.5: −33.68 / −43.11 / −36.77, mean
  −37.85 — age is doing the work, not just the 24h velocity.

## Stage 2 — tuning on the mean (`analysis/tune6.py`)

- Finer α: 0.35→−37.75 (P −33.23), 0.4→−37.62 (P −33.22),
  0.45→−37.51 (P −33.24), 0.5→−37.41 (P −32.98), 0.55→−37.34
  (P −33.02), 0.6→−37.29 (P −33.34), 0.65→−37.25 (P −33.40) — a
  smooth trade-off, not a spike.
- Robustness: α=0.5 top-3 (res, temp) cells −37.41/−37.42/−37.49;
  α=0.55 −37.34/−37.34/−37.42. The (res, temp) neighbourhood is flat.
- k sweep on blend α=0.5: k=30 −37.99, k=40 −37.66, k=50 −37.41
  (best), k=60 −37.72, k=75 −37.66, k=100 −37.83. **k=50 stays.**
- Kernel weighting within the top-k **REJECTED** (σ=0.5→−39.27,
  1.0→−38.39, 2.0→−37.65; uniform wins).
- lags6 (no 6h lag) −37.61 vs lags7 −37.41 — the 6h lag helps.
- OU blend **REJECTED at first pass** (β=0.1→−38.65, 0.2→−38.98,
  0.3→−39.64) — but the fresh OU was miscalibrated (OU alone −47.82;
  variance formula bug: residual 36h std used directly as diffusion σ,
  so Var blows up for small θ). The blend conclusion needed a re-test
  with a fixed OU (stage 3).
- Per-place slopes on blend α=0.5: temp slope 0.3→−37.40 (P −33.23),
  −0.3→−37.55; reserve slope 0.5→−37.72, −0.5→−37.34 (P −32.60!);
  ts=0.3+rs=0.5→−37.71.

## Stage 3 — fixed OU + α × reserve-slope grid (`analysis/tune6b.py`)

- **Fixed OU** (mean-reverted projection + empirical 36h residual
  spread as sd, no diffusion variance formula): OU alone −36.77 /
  −46.27 / −41.30, mean −41.45 — much better than the broken OU
  (−47.82) but still far below k-NN. Blends: α=0.5 + OU β=0.1 → −37.45,
  β=0.2 → −37.81; α=0.55 + OU β=0.1 → −37.41, β=0.2 → −37.81.
  **Lever (c) REJECTED conclusively, with a properly calibrated OU:
  even a 10% OU weight hurts.**
- α × reserve-slope grid (res ∈ {0.25, 0.3, 0.35}, temp ∈
  {1.0, 1.1, 1.2}): best α=0.6/rs=−0.5 → mean −37.21 (P −32.95);
  top-5 configs span −37.21..−37.28 — a flat plateau, not a spike.
  Best + temp slope 0.3: −37.19 (P −32.92) — a noise-level gain,
  not adopted.
- **Adopted config (central in the plateau, largest practice margin):
  blend α=0.55, k=50, lags {6,12,24,36,48,60,72}, uniform top-k,
  reserve slope −0.5, reserve 0.30, temperature 1.2, cutoff 0.0005.**
  Prototype scores: practice −32.63, B −42.28, C −36.88, mean −37.26.

## Entry (`main.py`, fresh implementation)

Stdlib only. Two k-NN components (vel36 no-age; vel24 + age),
blended 0.45/0.55, per-place temperature 1.2 + reserve slope −0.5.
Input/output from argv, then `MAKEFUN_INPUT_DIR` /
`MAKEFUN_OUTPUT_DIR`, then `/input` / `/output`.

- py_compile OK. Runtime ≈ 2.5 s (practice), 1.5 s (B), 1.0 s (C).
- **Scores from main.py itself: practice −32.63, B −42.26, C −36.90,
  mean −37.26** — matches the prototype within 0.02 on B/C.
- Determinism: two practice runs byte-identical.
- Schema: 481 probability rows, top10 = 10 unique ids, per-place
  sums 0.70–0.85 (as designed by the reserve slope), 41,112 bytes.
- One genuine edit after first validation: lag averaging now divides
  by the number of lags actually used instead of len(LAGS) — a
  correctness fix for edge-case histories; scores unchanged
  (all three outputs byte-identical to pre-edit runs).

## Files

- `main.py` — the entry.
- `analysis/proto6.py`, `tune6.py`, `tune6b.py` — prototype + tuning.
- `analysis/score.py`, `practice_answer.json`, `backtest_B/`,
  `backtest_C/` — local evaluation support (copied from build 5's
  analysis; not part of the entry).
- `out/` — local forecasts from `main.py` on the three windows.
- `README.md`, `LICENSE` (MIT), `BUILD_LOG.md` (this file),
  `TRANSCRIPT.md`, `.makefun/owner.json`.
- Session transcript: `~/workspace/makefun/session-build-6.jsonl`.

## What did NOT make it (and why)

- OU diffusion in any form (builds 3–4 exhausted it; stage 3 re-test
  with a fixed OU still hurts).
- Kernel-weighted k-NN (uniform wins).
- The 24h-velocity + age variant alone (mean −37.52 but practice
  −34.67 — the blend is strictly better).
- Temp-slope on top of the adopted config (+0.02 — noise, not adopted).
- No main.py from builds 2/3/4/5 was read; everything was
  re-implemented fresh from the design specs.
