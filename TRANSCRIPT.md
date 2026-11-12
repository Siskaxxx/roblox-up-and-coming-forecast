# Build 7 — transcript

Built by the main Muse agent directly in the main chat (no
subagent), one assistant message per chat turn, so the session log
(`~/workspace/makefun/session-build-7.jsonl`) carries the provider's
real per-message token usage — the gate that rejected builds 5 and 6.
All times WIB (UTC+7), 2026-11-12.

- **09:20 — session opened (msg-0001/0002).** State of play: build 5
  rejected by the server (thinking_wellformed + thinking_priced),
  build 6 rejected twice on thinking_priced; build 2 (OU, practice
  −33.81, mean −41.17, rank 106/517) is the only valid entry. Plan:
  re-implement build 6's blended k-NN family fresh from the design
  specs and push it further, judging on the mean of practice + B + C.
- **14:41–14:44 — analysis (msg-0003/0004).** Fresh generic k-NN
  machinery for analysis (`analysis/proto7.py`); build 6's adopted
  configuration reproduced exactly (mean −37.26). Nine levers tested
  across three stages (`proto7.py`, `proto7b.py`, `proto7c.py`):
  rejected — log-playing feature, acceleration, anchor recency,
  Gaussian lag weighting, k ≠ 50, age weight ≠ 1.0, all three-way
  blends except multi-scale velocity (small gain, not adopted).
  Adopted: blend α=0.65 plus **place smoothing σ=1.0** (new lever:
  B +0.77 / C +0.90 at screening post-processing) and the grid-tuned
  post-processing (reserve 0.25, T 1.2, reserve slope −0.5).
  Prototype final: **mean −36.70 (P −32.60 / B −41.72 / C −35.77)**.
- **14:45 — entry written + validated (msg-0005/0006).** `main.py`
  written fresh (stdlib only), then two genuine robustness edits
  (used-lags divisor; empty-training-set guard after a single-
  snapshot crash in edge testing). Scores from main.py itself:
  practice −32.596, B −41.713, C −35.798, mean −36.702. Outputs
  byte-identical before/after the edits; determinism verified by
  md5; edge cases (empty / 1 / 3 snapshots) pass; runtime ≤ 2.3 s.
- **14:47 — repository files (this log).** `README.md` from the
  requirements template (referral 2e4854b0), MIT `LICENSE`,
  `.makefun/owner.json`, `BUILD_LOG.md`, `TRANSCRIPT.md`.
- Usage export, session verification, bundle, kit check, push and
  submission follow in the submission phase; the export outcome is
  appended to this file when it happens.
