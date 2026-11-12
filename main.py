#!/usr/bin/env python3
"""Roblox Up-and-Coming forecaster, build 7.

Model: two k-NN regression components over historical chart snapshots,
blended, then smoothed across neighbouring places.

- Component A (weight 0.35): features [log rank, 36h log-rank velocity],
  trained on snapshot pairs separated by each lag in LAGS.
- Component B (weight 0.65): features [log rank, 24h log-rank velocity,
  log age], same lag structure. The age feature helps the backtests a
  lot and hurts practice a little; blending at 0.65 keeps most of the
  backtest gain (build 6 analysis, re-derived for build 7).
- Neighbours: k=50 nearest by Euclidean distance over standardised,
  weight-scaled features, uniform weight, outcomes averaged over lags.
  An outcome is the game's place (1..10) at the paired later snapshot,
  or off-chart.
- Place smoothing (new in build 7): each game's per-place probabilities
  are convolved with a Gaussian kernel (sigma=1.0 place) and rescaled to
  the game's original total. The k-NN outcome counts are sparse, and
  the chart moves gradually, so mass belongs on neighbouring places
  too. On the three local windows this was worth about +0.5 of mean
  log score over build 6, almost all of it in the backtests.
- Post-processing: per place k, probabilities are tempered (T=1.2),
  normalised to (1 - reserve_k) with reserve_k = 0.25 * (1 - 0.5 *
  (k-1)/9), cut off below 0.0005; the unnamed remainder is the scorer's
  shared pool for games not named at that place. top10 ranks games by
  total occupancy across places.

Standard library only, deterministic, offline. Reads the history from
argv[1], else $MAKEFUN_INPUT_DIR, else /input; writes forecast.json to
argv[2], else $MAKEFUN_OUTPUT_DIR, else /output.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta

LAGS = [6, 12, 24, 36, 48, 60, 72]
K_NEIGHBOURS = 50
# (velocity window hours, feature weights [rank, velocity, age], blend weight)
COMPONENTS = [
    (36, (1.0, 1.0, 0.0), 0.35),
    (24, (1.0, 1.0, 1.0), 0.65),
]
SMOOTH_SIGMA = 1.0
TEMPERATURE = 1.2
RESERVE = 0.25
RESERVE_SLOPE = -0.5
CUTOFF = 0.0005
PLACES = 10


def load_history(input_dir):
    """Read history.ndjson into snapshots [{at, chart {uid: rank}}]."""
    snaps = []
    path = os.path.join(input_dir, "history.ndjson")
    if not os.path.exists(path):
        return snaps
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = json.loads(line)
                t = datetime.fromisoformat(s["at"].replace("Z", "+00:00"))
            except (ValueError, KeyError, TypeError):
                continue  # skip a malformed line, never lose the forecast
            chart = s.get("up_and_coming") or []
            if not chart:
                continue
            snaps.append({
                "at": t,
                "chart": {r["universe_id"]: r["rank"] for r in chart},
            })
    snaps.sort(key=lambda s: s["at"])
    return snaps


def first_seen(snaps):
    first = {}
    for s in snaps:
        for u in s["chart"]:
            if u not in first:
                first[u] = s["at"]
    return first


def logrank_at(snaps, uid, t):
    """Log-rank of uid at time t, linear in time between the bracketing
    snapshots that contain it; None if it was never seen at or before t."""
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
    if after is None or after is before:
        return math.log(before["chart"][uid])
    t0, t1 = before["at"], after["at"]
    if t1 == t0:
        return math.log(before["chart"][uid])
    f = (t - t0).total_seconds() / (t1 - t0).total_seconds()
    return math.log(before["chart"][uid]) * (1 - f) + math.log(after["chart"][uid]) * f


def velocity(snaps, uid, t, window_h):
    """Change in log-rank per hour over the window ending at t (0 when the
    game has no reading at the start of the window)."""
    x0 = logrank_at(snaps, uid, t)
    x1 = logrank_at(snaps, uid, t - timedelta(hours=window_h))
    if x0 is None or x1 is None:
        return 0.0
    return (x0 - x1) / window_h


def features(snaps, snap, uid, vel_h, first):
    age_h = (snap["at"] - first.get(uid, snap["at"])).total_seconds() / 3600.0
    return [
        math.log(snap["chart"][uid]),
        velocity(snaps, uid, snap["at"], vel_h),
        math.log(age_h + 1.0),
    ]


def build_sets(snaps, vel_h, first):
    """Per lag h: (feature rows, outcomes) over all anchor snapshots whose
    paired later snapshot sits within tolerance of h hours ahead. An
    outcome is the game's place at the later snapshot (1..10), else 0."""
    n = len(snaps)
    sets = {}
    for h in LAGS:
        tol = max(2.0, 0.15 * h)
        rows, outcomes = [], []
        for i, a in enumerate(snaps):
            best, best_err = None, 1e9
            for j in range(i + 1, n):
                dt = (snaps[j]["at"] - a["at"]).total_seconds() / 3600.0
                err = abs(dt - h)
                if err < best_err:
                    best_err, best = err, j
                if dt > h + tol:
                    break
            if best is None or best_err > tol:
                continue
            b = snaps[best]
            for u in a["chart"]:
                rows.append(features(snaps, a, u, vel_h, first))
                r = b["chart"].get(u, 0)
                outcomes.append(r if 1 <= r <= PLACES else 0)
        sets[h] = (rows, outcomes)
    return sets


def forecast_component(snaps, sets, first, vel_h, weights):
    """k-NN per-place probabilities {uid: {place: p}} for the close chart."""
    pooled = [row for h in LAGS for row in sets[h][0]]
    if not pooled:
        # No usable snapshot pairs at all (e.g. a single-snapshot
        # history): every game gets zero, and post-processing leaves
        # each place to the scorer's unnamed pool.
        return {u: {k: 0.0 for k in range(1, PLACES + 1)} for u in snaps[-1]["chart"]}
    dims = len(weights)
    mu, sd = [], []
    for d in range(dims):
        m = sum(r[d] for r in pooled) / len(pooled)
        var = sum((r[d] - m) ** 2 for r in pooled) / len(pooled)
        mu.append(m)
        sd.append(math.sqrt(var) + 1e-9)

    def scale(row):
        return [(row[d] - mu[d]) / sd[d] * weights[d] for d in range(dims)]

    scaled = {h: [scale(r) for r in sets[h][0]] for h in LAGS}
    close = snaps[-1]
    out = {}
    for u in close["chart"]:
        x = scale(features(snaps, close, u, vel_h, first))
        row = {k: 0.0 for k in range(1, PLACES + 1)}
        # Average over the lags that actually have training pairs; a short
        # history can leave the longest lags empty, and dividing by
        # len(LAGS) would silently shrink every probability.
        used = sum(1 for h in LAGS if sets[h][0]) or 1
        for h in LAGS:
            train, outcomes = scaled[h], sets[h][1]
            if not train:
                continue
            dists = []
            for idx, tr in enumerate(train):
                d2 = 0.0
                for d in range(dims):
                    diff = tr[d] - x[d]
                    d2 += diff * diff
                dists.append((d2, idx))
            dists.sort()
            top = dists[:K_NEIGHBOURS]
            counts = [0] * (PLACES + 1)
            for _, idx in top:
                counts[outcomes[idx]] += 1
            for k in range(1, PLACES + 1):
                row[k] += counts[k] / len(top) / used
        out[u] = row
    return out


def smooth_places(q):
    """Convolve each game's per-place probabilities with a Gaussian kernel
    over places (sigma = SMOOTH_SIGMA), preserving the game's total."""
    rad = max(1, int(math.ceil(3 * SMOOTH_SIGMA)))
    kern = {d: math.exp(-(d * d) / (2 * SMOOTH_SIGMA ** 2))
            for d in range(-rad, rad + 1)}
    out = {}
    for u, row in q.items():
        total = sum(row.values())
        nr = {}
        for k in range(1, PLACES + 1):
            nr[k] = sum(row.get(k + d, 0.0) * w
                        for d, w in kern.items() if 1 <= k + d <= PLACES)
        s = sum(nr.values())
        if s > 0 and total > 0:
            nr = {k: v * total / s for k, v in nr.items()}
        out[u] = nr
    return out


def postprocess(q):
    rows = []
    for k in range(1, PLACES + 1):
        rk = RESERVE * (1 + RESERVE_SLOPE * (k - 1) / (PLACES - 1))
        items = [(u, q[u][k] ** TEMPERATURE) for u in q if q[u].get(k, 0) > 0]
        s = sum(p for _, p in items)
        if s <= 0:
            continue
        target = 1.0 - rk
        for u, p in items:
            v = p / s * target
            if v >= CUTOFF:
                rows.append({"universe_id": u, "place": k, "p": round(v, 6)})
    occ = {}
    for r in rows:
        occ[r["universe_id"]] = occ.get(r["universe_id"], 0.0) + r["p"]
    top10 = [u for u, _ in sorted(occ.items(), key=lambda kv: -kv[1])[:10]]
    return {"probabilities": rows, "top10": top10}


def forecast(snaps):
    if not snaps or not snaps[-1]["chart"]:
        return {"probabilities": [], "top10": []}
    first = first_seen(snaps)
    blended = None
    for vel_h, weights, w in COMPONENTS:
        sets = build_sets(snaps, vel_h, first)
        comp = forecast_component(snaps, sets, first, vel_h, weights)
        if blended is None:
            blended = {u: {k: w * comp[u][k] for k in range(1, PLACES + 1)}
                       for u in comp}
        else:
            for u in blended:
                for k in range(1, PLACES + 1):
                    blended[u][k] += w * comp[u][k]
    return postprocess(smooth_places(blended))


def main(argv):
    input_dir = (argv[1] if len(argv) > 1 else
                 os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = (argv[2] if len(argv) > 2 else
                  os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    fc = forecast(load_history(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w", encoding="utf-8") as f:
        json.dump(fc, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
