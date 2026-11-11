# TRANSCRIPT — build session for roblox-up-and-coming-forecast-01

Agent: Muse subagent (session 2f5133b7-320c-4467-8663-dd99a3c6e88c), working for the main agent / user cyberdrop (@retrodropid on make.fun, GitHub Siskaxxx).
Date: 2026-11-11, ~15:46–16:05 WIB. Working directory: ~/workspace/makefun/roblox-forecast/.

This is a human-readable record of the build. The authoritative transcripts for the make.fun submission are the agent session files the parent bundles with the kit CLI; this file summarises what was done, in order, so the build is auditable.

## 1. Setup and requirements

- Read `https://make.fun/skill.md` (v1.5.93, steps 1–7) and fetched the requirements endpoint `GET /api/competitions/roblox-up-and-coming-forecast-01/requirements` with the account API key (key read from `~/workspace/makefun/credentials.json`, never printed). Connection already done: handle `retrodropid`, key valid to 2026-12-11, agent wallet `FsUBASCTuJDbtZabdoNMj2SrvseMTtqncW699adHaJ54`.
- Recorded: $500 USDC escrow, entries close 2026-11-13T06:00:00Z, quota 5/day / 20 total, payout curve (1st $100 … 20th $14), checks incl. `repo_created_after_open` (opens 2026-11-11T06:00:00Z), `thinking_reproduces` ≥0.3, `run_deterministic` (2 runs), `dev_profitable`. Noted the discrepancy: escrow deadlines say entries_deadline 2026-11-16, requirements say entries_close 2026-11-13 — requirements win.
- Downloaded the kit (`curl https://make.fun/c/roblox-up-and-coming-forecast-01/kit | tar xz`) to `~/workspace/makefun/roblox-up-and-coming-forecast-01-kit/`. Read `RULES.txt` (full), `competition.yaml` (scoring/baseline sections), `input/task.json`, `input/NOTICE.txt`, `starter/main.ts` (the house baseline).

## 2. Local practice scorer (validated against the published baseline)

- `grade-local` refuses this competition ("its scorer and reference data are not published in the kit, so it is graded on the server only"), so I built a local scorer from RULES.txt (`analysis/score.py`).
- Reconstructed the practice answer from the public archive repo `Kenny-Dhub/roblox-snapshots` (cloned to `archive-repo/`; `data/discover/latest.json` history, `explore:up-and-coming` ranks). Conversion validated: 97/97 exact rank matches vs the kit history's last snapshot. Practice answer = the 10 reads 2026-11-08T00:37 → 2026-11-10T17:51Z (`analysis/practice_answer.json`).
- Validation: the kit baseline scores **−43.512039, typical p 0.012891, 6.4 hits, 53.5 agreement** — exactly the RULES.txt numbers. Scorer confirmed correct.

## 3. Data analysis (all in `analysis/`)

- `explore.py`: 34 snapshots, gaps 3.6–9.8h, chart 92–99 games, archive-only (details null), top-10 overlap 6–10/10 between consecutive reads.
- `dynamics.py`: close-rank of practice occupants; 8/100 slots were games not on the close chart; big risers (Sweep The Moon #29→#1, Impact Lab #64→#7, Ditch Hotel #84→#6); 42/97 close games left the chart; rank-transition table by bucket; playing growth vs rank move corr 0.018 at 6h.
- `kernel.py`: trajectories of key games; log-rank drift/volatility by bucket; momentum corr 0.211, slope β=0.213; OFF hazard by first-seen age; ~3.5 brand-new games per snapshot.
- Roblox games API (analysis only): fetched `created` for 201/207 distinct universes. Found a soft age gradient (hazard 2.8% → 8.9%/step), NOT a 28-day cliff (games aged up to 324d on the chart); gone games older on average (33.5d vs 21.9d).
- Variance-saturation test: Var(Δlog-rank) for top-20 games plateaus (0.094 @6h → 0.87 @100h); autocorrelation 0.97 → 0.50. Conclusion: mean reversion (OU), not a random walk.
- Feature tests (all rejected): playing growth 24h (no independent signal beyond rank velocity in joint OLS, coefficient −0.019); Top Trending presence (confounded by rank); recent-peak (ex-top-10 games keep falling); tenure at #1 (long tenure stays MORE, 87% vs 56% — opposite of the hypothesis).

## 4. Model iterations (main.py)

- **v1** — independent log-rank diffusion (bucket drift/volatility + momentum). Caught by verification: per-place sums > 1 (place 1: 1.83). Invalid, no score.
- **v2** — added per-place normalisation `p = q·(1−newcomer)/Σq` with newcomer shares calibrated at runtime from 48–96h pairs. Practice **−37.448**.
- **v3** — replaced the random-walk kernel with an Ornstein-Uhlenbeck step (θ, μ_eq re-estimated at runtime from lag-bucketed log-rank regressions; practice θ=0.0080, μ_eq≈rank 35). Practice **−34.087**.
- Tuning grid on practice: cutoff flat; σ 0.85 best (−33.88); θ 0.012–0.016 best (−33.64); β 0.1 best (−33.58); velocity window flat.
- **Validation on two independent backtests (B: train ≤Nov 5, test Nov 5–8; C: train ≤Nov 3, test Nov 3–6; built with `analysis/build_backtest.py`) rejected the tuning:** tuned configs lose 1–3 nats on B/C vs calibrated defaults. Submitted model = v3 with fully calibrated defaults.
- Final code review pass: removed unused imports, added a never-vanish guard to kernel rows.

## 5. Final verification (2026-11-11)

- `py_compile` OK (VM Python is 3.12.3; grader uses 3.13 — code uses no 3.13-only features, but 3.13 was not runnable locally).
- Practice: two runs byte-identical (deterministic); shape asserted (exactly {probabilities, top10}, 189 rows, 10,948 bytes, per-place sums 0.69–0.81); score **−34.087135**, typical p 0.033084, 7.0 hits, 56.2 agreement.
- Backtest B: **−48.191684**; backtest C: **−42.219333**.
- Edge cases: single-snapshot history and two-snapshot history without `details` both produce valid forecasts (fallbacks engage).
- `grade-local --driver process`: server-only grading, no local score by design.
- Runtime ≈ 0.5 s (limit 300 s). Stdlib only, no network, no randomness, no model calls.

## 6. Submission files written

- `main.py`, `LICENSE` (MIT, verbatim from kit starter), `README.md` (from requirements template), `.makefun/owner.json` (agent wallet), `BUILD_LOG.md` (this log's detail), `TRANSCRIPT.md` (this file).
- Not for the repo: `analysis/` (scorer, practice answer, backtests, archive clone), `out/` (local runs).
- Parent's next steps: push the five files to `Siskaxxx/roblox-up-and-coming-forecast`, bundle transcripts, `check`, `submit` before 2026-11-13 06:00 UTC. Do not submit from this session (no push/submission attempted).
