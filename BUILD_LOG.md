# BUILD_LOG — Roblox Up-and-Coming top-10 forecaster

Bounty: `roblox-up-and-coming-forecast-01` — "Predict Roblox's Up-and-Coming top 10"
Account: @retrodropid (agent wallet `FsUBASCTuJDbtZabdoNMj2SrvseMTtqncW699adHaJ54`)
Repo: `Siskaxxx/roblox-up-and-coming-forecast` (public, created by the user 2026-11-11)
Entry: `main.py` (Python 3.13, stdlib only) at repo root + `LICENSE` (MIT) + `README.md` + `.makefun/owner.json`
Deadline: entries close **2026-11-13 06:00 UTC** (13:00 WIB). Final window 2026-11-13 06:00 → 2026-11-16 06:00 UTC.
Payout: $500 USDC escrow (1st $100, 2nd $70, 3rd $50, 4th–10th $20, 11th–20th $14); a prize needs a log score strictly above the best house entry (baseline or house model forecast, whichever is higher).

## 0. Requirements (GET /api/competitions/<slug>/requirements, 2026-11-11)

- Deliverables: `repo_url`, `commit_sha` (40-char), transcript `bundle` (.tar.gz via kit CLI) or loose `transcripts`, `entry_message` + `entry_signature` (Ed25519, 10-min validity), `LICENSE` (OSI-approved), `main.ts` or `main.py` at repo root, optional `.makefun/owner.json`.
- Checks that matter: `repo_created_after_open` (repo must be created after 2026-11-11T06:00:00Z — the user-created repo qualifies), `repo_owner` (owner is the account's GitHub login OR `.makefun/owner.json` names an account wallet — included), `duplicate_snapshot`, `committed_secret`, `thinking_reproduces` (≥30% of lines accounted for by transcripts, entrypoint ≥1 authored line), `thinking_turns` (≥5 assistant turns), `thinking_revised` (≥2 writes to a file), `run_completes` (300s / 1 CPU / 1024 MB / 64 pids), `run_deterministic` (2 runs byte-identical), `dev_profitable` (profitable on the dev window), `hindsight_clean`.
- Quota: max 5 submissions/day, 20 total; last submission before deadline is final. Refused attempts count.
- Note: escrow deadlines list `entries_deadline 2026-11-16T06:00:00Z`, but the requirements endpoint's `deadlines.entries_close` is **2026-11-13T06:00:00Z** — the requirements endpoint wins; submit before Nov 13 06:00 UTC.
- `grade-local` does NOT score this competition locally ("its scorer and reference data are not published in the kit, so it is graded on the server only"). All "local grades" below are my own reconstruction of the practice answer (see §1).

## 1. Local practice scorer (validated)

- The practice answer is the 10 archive reads in the window 2026-11-07T23:00 → 2026-11-10T23:00Z, reconstructed from the public archive repo `Kenny-Dhub/roblox-snapshots` (git history of `data/discover/latest.json`, `explore:up-and-coming` ranks). Conversion verified: 97/97 exact rank matches against the kit history's last snapshot.
- Scorer implements RULES.txt exactly: per read, per place k, p = named row or `(1 − sum of place-k rows) / (# chart games not named at k)`, floored at 0.0001; read score = Σ ln p over 10 places; log score = mean over reads. Also computes top-10 hits and agreement.
- **Validation: the house baseline (kit `starter/main.ts`) scores −43.512039, typical p 0.012891, 6.4 hits, 53.5 agreement — exactly the numbers published in RULES.txt.** The scorer is correct.
- Tools: `analysis/score.py`, `analysis/practice_answer.json`, `analysis/build_practice_answer.py`.

## 2. Data findings (practice history: 34 archive reads, 2026-10-30 → 2026-11-07)

- Chart: 92–99 games; top-10 overlap between consecutive ~6h reads 6–10/10 (median 9); rank moves are large (median |move| 2–8 ranks per ~6h).
- The chart churns hard over 72h: the practice window's last read holds only 5/10 of the close's top 10; the close #1 ([💡UPD] Survive Your Ideas) was #1 in 0/10 practice reads.
- New games are the hardest part: 8/100 practice top-10 slots were games not on the close chart at all (Do Not Stop. → #1 for the last 3 reads, Bloody Ragdoll Physics → #2, Build a Bridge → #10). Deep risers also matter (Sweep The Moon close #29 → #1, Impact Lab close #64 → #7–8, The Ditch Hotel close #84 → #6–10).
- 42/97 close-chart games left the chart entirely during the practice window.
- Playing counts do NOT predict rank moves at 6h (corr 0.018). Playing growth over 24h correlates −0.317 with next-24h rank change, but a joint regression shows it carries **no independent signal beyond rank velocity** (OLS: next dlogr/hr = 0.0008 + 0.213·rank_vel − 0.019·play_growth, R²=0.049) — dropped.
- Top Trending presence does not add signal beyond rank (confounded). Recent-peak (was top-10 in last 48h) does NOT predict re-rise (ex-top-10 games keep falling, +0.245). Both dropped.
- Tenure at #1 does NOT predict falling (4+ snaps at #1 stay 87% vs 56% for 1 snap) — no tenure term.
- Age: universe `created` (fetched from the public Roblox games API for analysis only) shows a soft age gradient, not a 28-day cliff — games aged 39–324 days are on the chart. Exit hazard per ~6.4h step rises with age (<14d 2.8% → 27d+ 8.9%) and is strongly rank-dependent (top-10 young 0/202 exits; 51+ old 10.1%/step). Gone games in the practice window were older (mean 33.5d vs 21.9d survivors).
- **Mean reversion, not a random walk:** Var(log-rank change) for top-20 games saturates with lag (0.094 @6h → 0.87 @100h) and log-rank autocorrelation decays (0.97 @6h → 0.50 @65h). OU fit: θ≈0.008–0.010/hr, equilibrium log-rank ≈ ln 35.
- Momentum exists: corr(past dlogr, next dlogr) = 0.211; regression slope β = 0.213.
- Newcomer share per place (history pairs, lag 48–96h): place 1: 8.5%, 2: 6.7%, 3: 18.9%, 4: 28.7%, 5: 12.8%, 6: 18.9%, 7: 35.4%, 8: 30.5%, 9: 28.7%, 10: 36.6% (overall 22.6%; the practice window itself was calmer at 8%).

## 3. Model versions

### v1 — independent per-game log-rank diffusion (drift + volatility by rank bucket + momentum)
- **Invalid — no score.** Verification caught the bug: per-place probability sums exceeded 1 (place 1 summed to 1.83) — independent diffusions don't enforce one game per place, and the scorer rejects sums > 1.

### v2 — v1 + per-place normalisation with newcomer reserve
- `p(g,k) = q(g,k)·(1 − newcomer[k]) / Σ_g q(g,k)`; newcomer shares calibrated at runtime from 48–96h pairs, shrunk 30/70 toward the global mean.
- Log score **−37.448** (baseline −43.512).
- Per-place diagnostics showed places 1, 2, 5 worst (mean ln p −4.0, −4.8, −5.0); misses were the newcomers and Sweep The Moon / Ditch Hotel.

### v3 — Ornstein-Uhlenbeck diffusion (the submitted model)
- Replaced the random-walk kernel with an OU step: mean = center + θ·(μ_eq − center) + β·v, sd = rank-bucket σ. θ and μ_eq are re-estimated at runtime by regressing log-rank at lag L on log-rank now across lag buckets (calibrated θ=0.0080, μ_eq=3.57 ≈ rank 35 on practice history).
- Log score **−34.087** (typical p 0.033084, 7.0 hits, 56.2 agreement). Per-read scores degrade over the window (−23.4 first read → −41.1 last), as expected: the chart moves away from any forecast.

## 4. Tuning (and why the tuned values were REJECTED)

Grid on practice (with σ=0.85 baseline): cutoff 0.001–0.008 flat (−34.38…−34.04); σ 0.85 best (−33.88 vs −34.09 at 1.0); θ 0.012–0.016 best (−33.64…−33.67); β 0.1 best (−33.58); velocity window flat (36h kept).
**Backtest validation on two independent windows killed the tuning:**

| config | practice | B (train ≤Nov 5, test Nov 5–8, 12 reads) | C (train ≤Nov 3, test Nov 3–6, 10 reads) | mean |
|---|---|---|---|---|
| baseline (house) | −43.51 | −56.33 | −52.71 | −50.85 |
| v3 defaults (calibrated) | −34.09 | −48.19 | −42.22 | **−41.50** |
| tuned (σ.85, θ.014, β.1) | −33.82 | −50.23 | −44.96 | −43.00 |
| σ.85 + β.1 | −33.58 | −50.27 | −43.61 | −42.49 |
| σ.85 + θ.014 | −33.62 | −49.56 | −44.12 | −42.43 |

The practice-tuned values overfit the 10-read practice window and lose 1–3 nats on B/C. **Submitted model = v3 with fully calibrated defaults** (σ mult 1.0, calibrated θ/β, cutoff 0.003, velocity window 36h). Env overrides (`FORECAST_CUTOFF`, `FORECAST_SIGMA_MULT`, `FORECAST_THETA`, `FORECAST_BETA`, `FORECAST_VEL_WINDOW_H`) remain in the code for experimentation but default to the calibrated values.

Backtest windows were built from the archive repo (`analysis/build_backtest.py`); B is a much harder window for every config (baseline −56.33), i.e. that week had heavier churn.

## 5. Final verification (2026-11-11, after last code edits)

- `python3 -m py_compile main.py` — OK. (VM has Python 3.12.3 only; the grader runs 3.13. Code uses stdlib features present in both; no 3.13-only syntax. **Not run under 3.13 locally — flagging for the record.**)
- Practice: two runs → byte-identical `forecast.json` (deterministic). Shape asserted: exactly keys {probabilities, top10}, 10 distinct top10, per-place sums ≤ 1 (0.69–0.81), 189 rows, 10,948 bytes (< 1 MB cap).
- Practice score (final code): **−34.087135**, typical p 0.033084, 7.0 hits, 56.2 agreement.
- Backtest B: **−48.191684** (4.42 hits). Backtest C: **−42.219333** (5.90 hits).
- Edge cases: single-snapshot history → valid forecast (fallbacks engage, newcomer 0.15 flat); two snapshots with no `details` key → valid forecast.
- `grade-local --driver process`: prints "its scorer and reference data are not published in the kit, so it is graded on the server only" — no local grade exists by design; the scores above are my reconstruction.
- Runtime ≈ 0.5 s on the practice history (limit 300 s). Stdlib only, no network, no randomness, no model calls.

## 6. What the model cannot do (honest limits)

- Brand-new games entering the top 10 during the window are unknowable from history; they are covered only by the unnamed reserve (≈0.002–0.004 per game per place). In the practice window they cost ~8 slots.
- The final history will be denser (hourly recorder reads Nov 11–13 with `details.created`); calibration re-estimates from that input at runtime, but this denser regime was not testable locally (no hourly data exists yet).
- Backtest B shows some 72h windows are much harder than others (heavy churn); the final window's difficulty is unknown.

## 7. Files for the repo (in ~/workspace/makefun/roblox-forecast/)

- `main.py` — the entry (Python 3.13, stdlib only).
- `LICENSE` — MIT, copied verbatim from the kit's `starter/LICENSE`.
- `README.md` — from the requirements `readme_template` (owner Siskaxxx; no referral code was available, so the `?ref=` parameter is omitted).
- `.makefun/owner.json` — `{"wallet": "FsUBASCTuJDbtZabdoNMj2SrvseMTtqncW699adHaJ54"}` (agent wallet on the @retrodropid account; proves repo ownership for a wallet-only account).
- `BUILD_LOG.md` — this file.
- NOT pushed: `analysis/` (local scorer, practice answer, backtests, archive-repo clone), `out/` (local runs). No secrets anywhere in this directory (API key lives only in `~/workspace/makefun/credentials.json`, never in the repo).

## 8. Still needed to submit (parent's steps)

1. Push the five files above to `Siskaxxx/roblox-up-and-coming-forecast` (main branch; repo is empty, created by the user 2026-11-11).
2. Bundle the build transcripts (`node cli/makefun.mjs bundle --repo <repo> --out transcripts.tar.gz` from inside the unpacked kit) — must include this build session and the parent session.
3. `node cli/makefun.mjs check --kit . --repo <repo> --transcripts transcripts.tar.gz`, then `submit --competition roblox-up-and-coming-forecast-01 --repo-url https://github.com/Siskaxxx/roblox-up-and-coming-forecast --commit <sha> --title "OU diffusion forecaster" --bundle transcripts.tar.gz --keypair ../makefun-keypair.json` — before **2026-11-13 06:00 UTC**.
4. Consider a second submission after the final history is in (the program re-calibrates from the fuller history at grading time anyway — the server re-runs the final entry on the whole history up to the deadline, so no new code is needed for that; a resubmission would only help if the model itself changes).
