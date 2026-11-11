# BUILD_LOG — Roblox Up-and-Coming top-10 forecaster (build 2)

Bounty: `roblox-up-and-coming-forecast-01` — "Predict Roblox's Up-and-Coming top 10"
Account: @retrodropid (agent wallet `FsUBASCTuJDbtZabdoNMj2SrvseMTtqncW699adHaJ54`)
Repo: `Siskaxxx/roblox-up-and-coming-forecast` (public, created by the user 2026-11-11)
Entry: `main.py` (Python 3.13, stdlib only) at repo root + `LICENSE` (MIT) + `README.md` + `.makefun/owner.json`
Deadline: entries close **2026-11-13 06:00 UTC** (13:00 WIB). Final window 2026-11-13 06:00 → 2026-11-16 06:00 UTC.
Payout: $500 USDC escrow (1st $100, 2nd $70, 3rd $50, 4th–10th $20, 11th–20th $14); a prize needs a log score strictly above the best house entry (baseline or house model forecast, whichever is higher).

## Why a build 2 exists

Build 1 (2026-11-11 ~16:01 WIB) produced a working `main.py` (practice −34.09) and was
pushed to the repo (commit `90af81ef`), but its session transcript was lost to
compaction: 0% of its lines trace to any surviving record, so it cannot pass the
transcript gate (`thinking_reproduces` needs ≥30% of lines accounted for by the
sessions that wrote the code). Build 2 re-implements the same design fresh, in a
session recorded live to `~/workspace/makefun/session-build-2.jsonl` via
`rec_session.py`, so the transcripts that wrote the code exist. Build 1's code was
never read during build 2; the design spec below is what was re-implemented.

## 0. Requirements (GET /api/competitions/<slug>/requirements, 2026-11-11)

- Deliverables: `repo_url`, `commit_sha` (40-char), transcript `bundle` (.tar.gz via kit CLI) or loose `transcripts`, `entry_message` + `entry_signature` (Ed25519, 10-min validity), `LICENSE` (OSI-approved), `main.ts` or `main.py` at repo root, optional `.makefun/owner.json`.
- Checks that matter: `repo_created_after_open` (repo created after 2026-11-11T06:00:00Z — the user-created repo qualifies), `repo_owner` (owner is the account's GitHub login OR `.makefun/owner.json` names an account wallet — included), `duplicate_snapshot`, `committed_secret`, `thinking_reproduces` (≥30% of lines accounted for by transcripts, entrypoint ≥1 authored line), `thinking_turns` (≥5 assistant turns), `thinking_revised` (≥2 writes to a file), `run_completes` (300s / 1 CPU / 1024 MB / 64 pids), `run_deterministic` (2 runs byte-identical), `dev_profitable`, `hindsight_clean`.
- Quota: max 5 submissions/day, 20 total; last submission before deadline is final.
- Note: escrow deadlines list `entries_deadline 2026-11-16T06:00:00Z`, but the requirements endpoint's `deadlines.entries_close` is **2026-11-13T06:00:00Z** — the requirements endpoint wins; submit before Nov 13 06:00 UTC.
- `grade-local` does NOT score this competition locally ("its scorer and reference data are not published in the kit, so it is graded on the server only"). All "local grades" below are my own reconstruction of the practice answer (see §1).

## 1. Local practice scorer (validated, reused from build 1's analysis)

- The practice answer is the 10 archive reads in the window 2026-11-07T23:00 → 2026-11-10T23:00Z, reconstructed from the public archive repo `Kenny-Dhub/roblox-snapshots` (git history of `data/discover/latest.json`, `explore:up-and-coming` ranks). Conversion verified: 97/97 exact rank matches against the kit history's last snapshot.
- Scorer implements RULES.txt exactly: per read, per place k, p = named row or `(1 − sum of place-k rows) / (# chart games not named at k)`, floored at 0.0001; read score = Σ ln p over 10 places; log score = mean over reads. Also computes top-10 hits and agreement.
- **Validation: the house baseline (kit `starter/main.ts`) scores −43.512039, typical p 0.012891, 6.4 hits, 53.5 agreement — exactly the numbers published in RULES.txt.** The scorer is correct.
- Tools: `analysis/score.py`, `analysis/practice_answer.json`.

## 2. Data findings (practice history: 34 archive reads, 2026-10-30 → 2026-11-07)

- Chart: 92–99 games; top-10 overlap between consecutive ~6h reads 6–10/10 (median 9); rank moves are large (median |move| 2–8 ranks per ~6h).
- The chart churns hard over 72h: the practice window's last read holds only 5/10 of the close's top 10; the close #1 ([💡UPD] Survive Your Ideas) was #1 in 0/10 practice reads.
- New games are the hardest part: 8/100 practice top-10 slots were games not on the close chart at all. Deep risers also matter (Sweep The Moon close #29 → #1, Impact Lab close #64 → #7–8, The Ditch Hotel close #84 → #6–10).
- 42/97 close-chart games left the chart entirely during the practice window.
- Playing counts do NOT predict rank moves at 6h (corr 0.018). Playing growth over 24h correlates −0.317 with next-24h rank change, but a joint regression shows it carries **no independent signal beyond rank velocity** — dropped.
- Top Trending presence does not add signal beyond rank (confounded). Recent-peak (was top-10 in last 48h) does NOT predict re-rise. Both dropped.
- Tenure at #1 does NOT predict falling — no tenure term.
- Age: a soft age gradient, not a 28-day cliff — games aged 39–324 days are on the chart. Exit hazard per ~6.4h step rises with age (<14d 2.8% → 27d+ 8.9%) and is strongly rank-dependent (top-10 young 0 exits; 51+ old 10.1%/step).
- **Mean reversion, not a random walk:** Var(log-rank change) for top-20 games saturates with lag (0.094 @6h → 0.87 @100h) and log-rank autocorrelation decays (0.97 @6h → 0.50 @65h). OU fit: θ≈0.008–0.010/hr, equilibrium log-rank ≈ ln 35.
- Momentum exists: corr(past dlogr, next dlogr) = 0.211; regression slope β = 0.213.
- Newcomer share per place (history pairs, lag 48–96h): place 1: 8.5%, 2: 6.7%, 3: 18.9%, 4: 28.7%, 5: 12.8%, 6: 18.9%, 7: 35.4%, 8: 30.5%, 9: 28.7%, 10: 36.6% (overall 22.6%; the practice window itself was calmer at 8%).

## 3. Model (implemented fresh in build 2)

Per-game Ornstein-Uhlenbeck forward diffusion over log-rank, bins 0.1 wide
(ranks 1–181 + off-chart state), 72 hourly steps, time-averaged:

- **OU step:** for a bin at log-rank x with mass m, the mass spreads as
  Normal(mean = x + θ(μ−x) + β·v, sd = σ_bucket(x)), truncated at ±4σ; mass
  below rank 1 folds into bin 0, mass past the top folds OFF. θ and μ are
  re-estimated at runtime by regressing log-rank at lag L on log-rank now
  across lag buckets (median across buckets; fallbacks θ=0.008, μ=ln 35).
- **Volatility:** σ per rank bucket as mean(Δlogr²/dt) over consecutive pairs,
  shrunk toward the global value (pseudo-count 20); fallbacks from the practice
  history's diffusion estimates (0.14 / 0.19 / 0.11 / 0.12 / 0.10 / 0.09 / 0.06
  for buckets 1 / 2–3 / 4–5 / 6–10 / 11–20 / 21–50 / 51+). Env override
  `FORECAST_SIGMA_MULT` (default 1.0).
- **Momentum:** v = the game's log-rank velocity over the last 36h (env
  `FORECAST_VEL_WINDOW_H`); β from a triples regression of next dlogr/hr on past
  dlogr/hr (fallback 0.213, clamped to [0, 0.5]; env `FORECAST_BETA`).
- **Exit hazard:** per-hour, by rank group (1–10 / 11–20 / 21–50 / 51+) × age
  group (<7d / 7–14d / 14–21d / 21–28d / 28d+), from consecutive-pair exits with
  hierarchical shrinkage (cell → rank group → global, pseudo-count 10), capped
  at 0.05/hr. Age from `details.created` when present, else first appearance on
  the chart. OFF is absorbing.
- **Newcomer reserve:** per-place shares from 48–96h history pairs (fallback
  24–120h, then 0.15 flat), shrunk 0.7·raw + 0.3·global mean, clamped to
  [0, 0.6]. `p(g,k) = q(g,k)·(1 − newcomer[k]) / Σ_g q(g,k)` for games with
  q ≥ cutoff 0.003 (env `FORECAST_CUTOFF`); the rest stays unnamed.
- **top10:** games ordered by total top-10 occupancy Σ_k q(g,k), ties by close
  rank then universe id. Deterministic: games simulated in sorted universe_id
  order, outputs sorted by (place, −p, universe_id).

## 4. Scores (build 2, calibrated defaults — no practice tuning)

| config | practice (10 reads) | backtest B (Nov 5–8, 12 reads) | backtest C (Nov 3–6, 10 reads) | mean |
|---|---|---|---|---|
| house baseline | −43.51 | −56.33 | −52.71 | −50.85 |
| **build 2 (this entry)** | **−33.81** (typical p 0.034, 7.3 hits, 61.1 agreement) | **−48.20** | **−41.49** | **−41.17** |
| build 1 (for reference) | −34.09 | −48.19 | −42.22 | −41.50 |

No practice-tuned hyperparameters: tuning on the 10-read practice window overfits
(build 1 showed tuned values losing 1–3 nats on B/C), so the submitted model uses
fully calibrated defaults. Env overrides (`FORECAST_CUTOFF`, `FORECAST_SIGMA_MULT`,
`FORECAST_THETA`, `FORECAST_BETA`, `FORECAST_VEL_WINDOW_H`) remain in the code for
experimentation but default to the calibrated values.

## 5. Final verification (2026-11-11, after last code edits)

- `python3 -m py_compile main.py` — OK. (VM has Python 3.12.3 only; the grader runs 3.13. Code uses stdlib features present in both; no 3.13-only syntax. **Not run under 3.13 locally — flagging for the record.**)
- Practice: two runs → byte-identical `forecast.json` (deterministic). Shape asserted: exactly keys {probabilities, top10}, 10 distinct top10, per-place sums ≤ 1 (0.68–0.89), 198 rows, 11,457 bytes (< 1 MB cap).
- Practice score (final code): **−33.813982**, typical p 0.034000, 7.3 hits, 61.1 agreement.
- Backtest B: **−48.199901** (4.42 hits). Backtest C: **−41.491403** (5.90 hits).
- Velocity fix, found during review: the first `velocity()` took the oldest observation at least 36h back, diluting the momentum signal for long-charted games; it now interpolates log-rank at exactly (close − 36h). The fix improved all three windows (practice −33.97 → −33.81, B −48.41 → −48.20, C −42.03 → −41.49), so it is a real correction, not practice overfitting.
- Edge cases: single-snapshot history → valid forecast (fallbacks engage, newcomer 0.15 flat, all places sum 0.85); two snapshots with no `details` key → valid forecast.
- `grade-local --driver process`: prints "its scorer and reference data are not published in the kit, so it is graded on the server only" — no local grade exists by design; the scores above are my reconstruction.
- Runtime ≈ 3 s on the practice history (limit 300 s). Stdlib only, no network, no randomness, no model calls.

## 6. What the model cannot do (honest limits)

- Brand-new games entering the top 10 during the window are unknowable from history; they are covered only by the unnamed reserve (≈0.002–0.004 per game per place). In the practice window they cost ~8 slots.
- The final history will be denser (hourly recorder reads Nov 11–13 with `details.created`); calibration re-estimates from that input at runtime, but this denser regime was not testable locally (no hourly data exists yet).
- Backtest B shows some 72h windows are much harder than others (heavy churn); the final window's difficulty is unknown.

## 7. Files for the repo (in ~/workspace/makefun/roblox-forecast/)

- `main.py` — the entry (Python 3.13, stdlib only).
- `LICENSE` — MIT, copied verbatim from the kit's `starter/LICENSE`.
- `README.md` — from the requirements `readme_template` (owner Siskaxxx; referral code `2e4854b0` in the footer link).
- `.makefun/owner.json` — `{"wallet": "FsUBASCTuJDbtZabdoNMj2SrvseMTtqncW699adHaJ54"}` (agent wallet on the @retrodropid account; proves repo ownership for a wallet-only account).
- `BUILD_LOG.md` — this file.
- `TRANSCRIPT.md` — build record for this session.
- NOT pushed: `analysis/` (local scorer, practice answer, backtests, archive-repo clone), `out/` (local runs). No secrets anywhere in this directory (API key lives only in `~/workspace/makefun/credentials.json`, never in the repo).

## 8. Still needed to submit (parent's steps)

1. Push the files above to `Siskaxxx/roblox-up-and-coming-forecast` (main branch; new commit over build 1's `90af81ef`).
2. Bundle the build transcripts (`node cli/makefun.mjs bundle --repo <repo> --out transcripts.tar.gz --session ~/workspace/makefun/session-build-2.jsonl --no-discover` from inside the unpacked kit) — the live-recorded build-2 session, plus the parent session.
3. `node cli/makefun.mjs check --kit . --repo <repo> --transcripts transcripts.tar.gz`, then `submit --competition roblox-up-and-coming-forecast-01 --repo-url https://github.com/Siskaxxx/roblox-up-and-coming-forecast --commit <sha> --title "OU diffusion forecaster" --bundle transcripts.tar.gz --keypair ../makefun-keypair.json` — before **2026-11-13 06:00 UTC**.
