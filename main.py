#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (build 6).

Model: a blend of two k-NN regressions over chart trajectories — a
continuation of build 5's k-NN family (build 5 is submitted; this build
keeps its strengths and adds the blend that the three-window evidence
supports). No OU diffusion, no parametric Gaussian, no network.

For each game on the closing chart, find the 50 most similar (game, anchor)
pairs in the history — similarity in standardised feature space — and
average what those neighbours actually did: their place at each lag
6-72h after the anchor. The forecast is the lag-averaged neighbour
outcome distribution, a direct empirical estimate of the time-averaged
place distribution the score rewards.

Two components, blended 0.45 / 0.55:
  A: features (log-rank, 36h log-rank velocity) — best on the practice
     window (most recent, best proxy for the final window).
  B: features (log-rank, 24h log-rank velocity, log-age) — best on the
     two independent backtests (B: -41.44, C: -36.46 alone).
Blending hedges the velocity-window / age uncertainty: the blend beats
either component alone on every window (practice -32.63, B -42.28,
C -36.88, mean -37.26 vs build 5's -33.29 / -43.54 / -37.66 / -38.16).

Post-processing: per-place probabilities are the blended neighbour
outcome frequencies, sharpened with temperature 1.2 and normalised to
(1 - reserve_k) per place, where the unnamed reserve falls with place
(reserve_k = 0.30 * (1 - 0.5 * (k-1) / 9)): newcomers win a larger share
of the top places, so less mass is left unnamed there. Probabilities
below 0.0005 are dropped; the top 10 is ranked by total occupancy.

Deterministic: no randomness, no wall-clock, no network. Two runs on the
same history produce byte-identical output. Stdlib only.
"""
import heapq
import json
import math
import os
import sys
from datetime import datetime, timezone

LAGS = [6, 12, 24, 36, 48, 60, 72]
K = 50
BLEND_B = 0.55  # weight of component B (24h velocity + age)
TEMP = 1.2
RESERVE = 0.30
RES_SLOPE = -0.5
CUTOFF = 0.0005


def load_history(input_dir):
    """Read history.ndjson -> snapshots sorted by time.
    Each snapshot: {"at": datetime, "chart": {universe_id: rank}}."""
    snaps = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            chart = s.get("up_and_coming") or []
            if not chart:
                continue
            t = datetime.fromisoformat(s["at"].replace("Z", "+00:00"))
            snaps.append({"at": t,
                          "chart": {r["universe_id"]: r["rank"] for r in chart}})
    snaps.sort(key=lambda s: s["at"])
    return snaps


def logrank_at(snaps, uid, t):
    """Interpolated log-rank of uid at time t (linear in time, log-rank).
    None if the game was never seen at or before t."""
    before = None
    after = None
    for s in snaps:
        if uid in s["chart"]:
            if s["at"] <= t:
                before = s
            elif after is None:
                after = s
                break
    if before is None:
        return None
    if after is None or before is after:
        return math.log(before["chart"][uid])
    t0, t1 = before["at"], after["at"]
    if t1 == t0:
        return math.log(before["chart"][uid])
    f = (t - t0).total_seconds() / (t1 - t0).total_seconds()
    return math.log(before["chart"][uid]) * (1 - f) + math.log(after["chart"][uid]) * f


def velocity(snaps, uid, close_t, window_h):
    """d log-rank / hr over the last window_h hours before close_t."""
    from datetime import timedelta
    x0 = logrank_at(snaps, uid, close_t)
    x1 = logrank_at(snaps, uid, close_t - timedelta(hours=window_h))
    if x0 is None or x1 is None:
        return 0.0
    return (x0 - x1) / window_h


def first_seen(snaps):
    first = {}
    for s in snaps:
        for u in s["chart"]:
            if u not in first:
                first[u] = s["at"]
    return first


def features(snaps, anchor, uid, vel_h, first):
    """[ln rank, velocity, ln(age_h + 1)] at anchor snapshot."""
    r = anchor["chart"][uid]
    v = velocity(snaps, uid, anchor["at"], vel_h)
    age_h = (anchor["at"] - first.get(uid, anchor["at"])).total_seconds() / 3600
    return [math.log(r), v, math.log(age_h + 1.0)]


def build_sets(snaps, vel_h):
    """Per lag: (features [[ln r, v, ln age]], outcomes [place 0..10]).
    Outcome = the game's place at the snapshot nearest anchor + lag,
    or 0 if off-chart / beyond place 10."""
    first = first_seen(snaps)
    n = len(snaps)
    out = {}
    for h in LAGS:
        feats, outcomes = [], []
        tol = max(2.0, 0.15 * h)
        for i, a in enumerate(snaps):
            best, best_err = None, 1e9
            for j in range(i + 1, n):
                dt = (snaps[j]["at"] - a["at"]).total_seconds() / 3600
                err = abs(dt - h)
                if err < best_err:
                    best_err, best = err, j
                if dt > h + tol:
                    break
            if best is None or best_err > tol:
                continue
            b = snaps[best]
            for u in a["chart"]:
                feats.append(features(snaps, a, u, vel_h, first))
                place = b["chart"].get(u, 0)
                outcomes.append(place if place and place <= 10 else 0)
        out[h] = (feats, outcomes)
    return out, first


def standardise(sets):
    """Mean/std of the union of all lags' features (per dimension)."""
    flat = [f for h in LAGS for f in sets[h][0]]
    if not flat:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    mu = [sum(f[d] for f in flat) / len(flat) for d in range(3)]
    sd = []
    for d in range(3):
        m = mu[d]
        var = sum((f[d] - m) ** 2 for f in flat) / len(flat)
        sd.append(math.sqrt(var) + 1e-9)
    return mu, sd


def knn_forecast(snaps, sets, first, vel_h, w_age):
    """k-NN forecast for the closing chart. w_age scales the standardised
    age dimension (0 disables age). Uniform weights over the k nearest
    neighbours per lag; rows averaged over lags."""
    close = snaps[-1]
    mu, sd = standardise(sets)
    wvec = [1.0, 1.0, w_age]
    out = {}
    for u, r in close["chart"].items():
        x = features(snaps, close, u, vel_h, first)
        xs = [(x[d] - mu[d]) / sd[d] * wvec[d] for d in range(3)]
        row = {kk: 0.0 for kk in range(1, 11)}
        used_lags = 0
        for h in LAGS:
            feats, outcomes = sets[h]
            if not feats:
                continue
            used_lags += 1
            # (distance, index) pairs; index breaks ties deterministically
            scored = []
            for idx, f in enumerate(feats):
                d2 = 0.0
                for d in range(3):
                    diff = (f[d] - mu[d]) / sd[d] * wvec[d] - xs[d]
                    d2 += diff * diff
                scored.append((d2, idx))
            near = heapq.nsmallest(K, scored)
            counts = {kk: 0 for kk in range(1, 11)}
            for _, idx in near:
                o = outcomes[idx]
                if o:
                    counts[o] += 1
            n_near = len(near)
            for kk in range(1, 11):
                row[kk] += counts[kk] / n_near
        # Average over the lags actually used (a lag with no training
        # examples contributes nothing instead of diluting the row).
        if used_lags:
            for kk in range(1, 11):
                row[kk] /= used_lags
        out[u] = row
    return out


def blend(q_a, q_b, alpha_b):
    """(1 - alpha_b) * q_a + alpha_b * q_b, per game per place."""
    return {u: {k: (1 - alpha_b) * q_a[u][k] + alpha_b * q_b[u][k]
                for k in range(1, 11)} for u in q_a}


def postprocess(q):
    """Per-place temperature, normalise to (1 - reserve_k), cutoff,
    top 10 by total occupancy."""
    rows = []
    for k in range(1, 11):
        reserve_k = RESERVE * (1 + RES_SLOPE * (k - 1) / 9)
        items = [(u, q[u][k] ** TEMP) for u in q if q[u].get(k, 0) > 0]
        s = sum(p for _, p in items)
        if s <= 0:
            continue
        target = 1.0 - reserve_k
        for u, p in items:
            v = p / s * target
            if v >= CUTOFF:
                rows.append({"universe_id": u, "place": k, "p": round(v, 6)})
    occ = {}
    for r in rows:
        occ[r["universe_id"]] = occ.get(r["universe_id"], 0) + r["p"]
    top10 = [u for u, _ in sorted(occ.items(), key=lambda kv: -kv[1])[:10]]
    return {"probabilities": rows, "top10": top10}


def main():
    input_dir = (sys.argv[1] if len(sys.argv) > 1
                 else os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = (sys.argv[2] if len(sys.argv) > 2
                  else os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    snaps = load_history(input_dir)
    if not snaps or not snaps[-1]["chart"]:
        fc = {"probabilities": [], "top10": []}
    else:
        sets36, first = build_sets(snaps, 36)
        sets24, _ = build_sets(snaps, 24)
        q_a = knn_forecast(snaps, sets36, first, 36, 0.0)
        q_b = knn_forecast(snaps, sets24, first, 24, 1.0)
        q = blend(q_a, q_b, BLEND_B)
        fc = postprocess(q)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
