#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster.

Entry for make.fun bounty roblox-up-and-coming-forecast-01.
Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Stdlib only, no network, fully deterministic (same input -> byte-identical output).

Model: per-game Ornstein-Uhlenbeck forward diffusion over log-rank.
  * Each game's rank is a distribution over log-rank bins (0.1 wide, ranks 1-181,
    plus an off-chart state). Every hour it mean-reverts toward a common
    equilibrium (log-rank ~ ln 35), diffuses (rank-bucket volatility) and carries
    momentum (recent log-rank velocity, slope ~0.21), and can leave the chart
    (per-hour exit hazard by rank group x age group).
  * The 72 hourly steps are averaged, so p(game, place) is the time-averaged
    probability that the game holds that place over the window.
  * Each place is normalised by its raw total with the newcomer share left
    unnamed, so the unnamed mass covers both the long tail and brand-new games
    entering the chart during the window (unknowable from history).
  * Calibration (OU rate/equilibrium, volatility, momentum, hazard, newcomer
    shares) is re-estimated from the input history at every run, with fallbacks
    when a cell has too little data.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tunables (env overrides; defaults are the calibrated values).
CUTOFF = float(os.environ.get("FORECAST_CUTOFF", "0.003"))
SIGMA_MULT = float(os.environ.get("FORECAST_SIGMA_MULT", "1.0"))
THETA_ENV = os.environ.get("FORECAST_THETA")
BETA_ENV = os.environ.get("FORECAST_BETA")
VEL_WINDOW_H = float(os.environ.get("FORECAST_VEL_WINDOW_H", "36"))

# Log-rank binning: 0.1-wide bins from rank 1 to rank 181, plus OFF.
BIN_W = 0.1
X_MAX = math.log(181.0)
N_BINS = int(math.ceil(X_MAX / BIN_W))  # 52
BIN_EDGES = [i * BIN_W for i in range(N_BINS + 1)]
BIN_CENTERS = [(i + 0.5) * BIN_W for i in range(N_BINS)]

# Rank buckets for volatility / hazard (by rank).
RANK_BUCKETS = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 1000000)]
# Fallback per-hour volatility (instantaneous sigma) per rank bucket, from the
# practice history's diffusion estimates.
SIGMA_FALLBACK = [0.14, 0.19, 0.11, 0.12, 0.10, 0.09, 0.06]
# Fallback OU parameters and momentum.
THETA_FALLBACK = 0.008
MU_FALLBACK = math.log(35.0)
BETA_FALLBACK = 0.213
HAZARD_FALLBACK = 0.008  # per hour
NEWCOMER_FALLBACK = 0.15

AGE_GROUPS = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 100000)]
HAZ_RANK_GROUPS = [(1, 10), (11, 20), (21, 50), (51, 1000000)]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bucket_index(rank):
    for i, (lo, hi) in enumerate(RANK_BUCKETS):
        if lo <= rank <= hi:
            return i
    return len(RANK_BUCKETS) - 1


def haz_rank_group(rank):
    for i, (lo, hi) in enumerate(HAZ_RANK_GROUPS):
        if lo <= rank <= hi:
            return i
    return len(HAZ_RANK_GROUPS) - 1


def age_group(age_days):
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if lo <= age_days < hi:
            return i
    return len(AGE_GROUPS) - 1


def ols(xs, ys):
    """Ordinary least squares: returns (slope, intercept) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    return task, snaps


def game_histories(snaps):
    """universe_id -> list of (time, rank) sorted by time."""
    hist = {}
    for s in snaps:
        t = parse_time(s["at"])
        for r in s.get("up_and_coming") or []:
            hist.setdefault(r["universe_id"], []).append((t, r["rank"]))
    return hist


def game_ages(snaps, hist, close_time):
    """universe_id -> age in days at close_time.

    Prefer details.created (house recorder histories); fall back to the game's
    first appearance on the chart.
    """
    ages = {}
    details = {}
    for s in reversed(snaps):
        d = s.get("details")
        if isinstance(d, dict) and d:
            details = d
            break
    for uid, h in hist.items():
        created = None
        if details:
            det = details.get(str(uid)) or details.get(uid)
            if isinstance(det, dict) and det.get("created"):
                try:
                    created = parse_time(det["created"])
                except ValueError:
                    created = None
        if created is not None:
            ages[uid] = max(0.0, (close_time - created).total_seconds() / 86400.0)
        else:
            ages[uid] = max(0.0, (close_time - h[0][0]).total_seconds() / 86400.0)
    return ages


def calibrate(snaps, hist, close_time):
    """Estimate OU rate/equilibrium, volatility, momentum, hazard and newcomer
    shares from the history, with fallbacks when data is thin."""
    # ---- OU theta / mu_eq: regress log-rank at lag L on log-rank now. ----
    lag_buckets = [(6, 18), (18, 36), (36, 72), (72, 120), (120, 200)]
    thetas, mus = [], []
    for lo_h, hi_h in lag_buckets:
        xs, ys, lags = [], [], []
        for i, a in enumerate(snaps):
            ta = parse_time(a["at"])
            ma = {r["universe_id"]: r["rank"] for r in a.get("up_and_coming") or []}
            for b in snaps[i + 1:]:
                tb = parse_time(b["at"])
                lag = hours_between(ta, tb)
                if lag < lo_h:
                    continue
                if lag >= hi_h:
                    break
                mb = {r["universe_id"]: r["rank"] for r in b.get("up_and_coming") or []}
                for u in set(ma) & set(mb):
                    xs.append(math.log(ma[u]))
                    ys.append(math.log(mb[u]))
                    lags.append(lag)
        if len(xs) < 30:
            continue
        fit = ols(xs, ys)
        if fit is None:
            continue
        slope, intercept = fit
        lag_mean = sum(lags) / len(lags)
        if not (0.05 < slope < 0.999):
            continue
        theta_b = -math.log(slope) / lag_mean
        if not (0.0005 < theta_b < 0.05):
            continue
        mu_b = intercept / (1.0 - slope)
        if not (2.0 < mu_b < 5.0):
            continue
        thetas.append(theta_b)
        mus.append(mu_b)
    if thetas:
        thetas.sort()
        mus.sort()
        theta = thetas[len(thetas) // 2]
        mu_eq = mus[len(mus) // 2]
    else:
        theta, mu_eq = THETA_FALLBACK, MU_FALLBACK
    if THETA_ENV is not None:
        theta = float(THETA_ENV)

    # ---- Volatility per rank bucket: mean of (dlogr^2 / dt) over pairs. ----
    sig2 = [0.0] * len(RANK_BUCKETS)
    cnt = [0] * len(RANK_BUCKETS)
    for a, b in zip(snaps, snaps[1:]):
        dt = hours_between(parse_time(a["at"]), parse_time(b["at"]))
        if dt <= 0:
            continue
        ma = {r["universe_id"]: r["rank"] for r in a.get("up_and_coming") or []}
        mb = {r["universe_id"]: r["rank"] for r in b.get("up_and_coming") or []}
        for u in set(ma) & set(mb):
            dx = math.log(mb[u]) - math.log(ma[u])
            bi = bucket_index(ma[u])
            sig2[bi] += dx * dx / dt
            cnt[bi] += 1
    sig2_global = (
        sum(sig2) / max(1, sum(cnt)) if sum(cnt) > 0 else 0.10 ** 2
    )
    sigmas = []
    for i in range(len(RANK_BUCKETS)):
        if cnt[i] > 0:
            raw = sig2[i] / cnt[i]
            # Shrink toward the global value when the bucket is thin.
            val = (cnt[i] * raw + 20.0 * sig2_global) / (cnt[i] + 20.0)
        else:
            val = SIGMA_FALLBACK[i] ** 2
        sigmas.append(max(0.02, math.sqrt(val)) * SIGMA_MULT)

    # ---- Momentum beta: regress next dlogr/hr on past dlogr/hr (triples). ----
    past, nxt = [], []
    for i in range(2, len(snaps)):
        a, b, c = snaps[i - 2], snaps[i - 1], snaps[i]
        ta, tb, tc = parse_time(a["at"]), parse_time(b["at"]), parse_time(c["at"])
        dtab = hours_between(ta, tb)
        dtbc = hours_between(tb, tc)
        if dtab <= 0 or dtbc <= 0:
            continue
        ma = {r["universe_id"]: r["rank"] for r in a.get("up_and_coming") or []}
        mb = {r["universe_id"]: r["rank"] for r in b.get("up_and_coming") or []}
        mc = {r["universe_id"]: r["rank"] for r in c.get("up_and_coming") or []}
        for u in set(ma) & set(mb) & set(mc):
            past.append((math.log(mb[u]) - math.log(ma[u])) / dtab)
            nxt.append((math.log(mc[u]) - math.log(mb[u])) / dtbc)
    if len(past) >= 100:
        fit = ols(past, nxt)
        beta = fit[0] if fit is not None else BETA_FALLBACK
    else:
        beta = BETA_FALLBACK
    beta = min(0.5, max(0.0, beta))
    if BETA_ENV is not None:
        beta = float(BETA_ENV)

    # ---- Exit hazard per (rank group, age group), per hour. ----
    ages = game_ages(snaps, hist, close_time)
    # cell -> [exits, total, sum_dt]
    cells = {}
    rank_tot = {}
    glob = [0, 0, 0.0]
    for a, b in zip(snaps, snaps[1:]):
        dt = hours_between(parse_time(a["at"]), parse_time(b["at"]))
        if dt <= 0:
            continue
        mb = {r["universe_id"] for r in b.get("up_and_coming") or []}
        for r in a.get("up_and_coming") or []:
            u = r["universe_id"]
            rg = haz_rank_group(r["rank"])
            ag = age_group(ages.get(u, 0.0))
            key = (rg, ag)
            cell = cells.setdefault(key, [0, 0, 0.0])
            cell[1] += 1
            cell[2] += dt
            rt = rank_tot.setdefault(rg, [0, 0, 0.0])
            rt[1] += 1
            rt[2] += dt
            glob[1] += 1
            glob[2] += dt
            if u not in mb:
                cell[0] += 1
                rt[0] += 1
                glob[0] += 1

    def rate_to_hourly(exits, total, sum_dt):
        if total <= 0 or sum_dt <= 0:
            return None
        r = exits / total
        if r <= 0:
            return 0.0
        if r >= 1:
            return 0.05
        return 1.0 - (1.0 - r) ** (1.0 / (sum_dt / total))

    h_global = rate_to_hourly(*glob)
    if h_global is None:
        h_global = HAZARD_FALLBACK
    h_rank = {}
    for rg, (e, t, d) in rank_tot.items():
        h = rate_to_hourly(e, t, d)
        if h is None:
            h = h_global
        else:
            h = (t * h + 10.0 * h_global) / (t + 10.0)
        h_rank[rg] = min(0.05, h)
    hazard = {}
    for rg in range(len(HAZ_RANK_GROUPS)):
        for ag in range(len(AGE_GROUPS)):
            e, t, d = cells.get((rg, ag), [0, 0, 0.0])
            h = rate_to_hourly(e, t, d)
            parent = h_rank.get(rg, h_global)
            if h is None:
                h = parent
            else:
                h = (t * h + 10.0 * parent) / (t + 10.0)
            hazard[(rg, ag)] = min(0.05, max(0.0, h))

    # ---- Newcomer shares per place, from 48-96h pairs. ----
    newcomer = None
    for lo_h, hi_h in ((48, 96), (24, 120)):
        per_place = [[] for _ in range(10)]
        for i, a in enumerate(snaps):
            ta = parse_time(a["at"])
            ids_a = {r["universe_id"] for r in a.get("up_and_coming") or []}
            for b in snaps[i + 1:]:
                lag = hours_between(ta, parse_time(b["at"]))
                if lag < lo_h:
                    continue
                if lag >= hi_h:
                    break
                for k, r in enumerate((b.get("up_and_coming") or [])[:10]):
                    per_place[k].append(0.0 if r["universe_id"] in ids_a else 1.0)
        if per_place[0]:
            raw = [
                (sum(v) / len(v) if v else 0.0) for v in per_place
            ]
            glob_mean = sum(raw) / len(raw)
            newcomer = [
                min(0.6, max(0.0, 0.7 * r + 0.3 * glob_mean)) for r in raw
            ]
            break
    if newcomer is None:
        newcomer = [NEWCOMER_FALLBACK] * 10

    return {
        "theta": theta,
        "mu_eq": mu_eq,
        "sigmas": sigmas,
        "beta": beta,
        "hazard": hazard,
        "newcomer": newcomer,
        "ages": ages,
    }


def velocity(uid, hist, close_time):
    """Recent log-rank velocity (per hour) over the last VEL_WINDOW_H hours.

    Interpolates the game's log-rank at (close_time - VEL_WINDOW_H) from the
    two surrounding observations; falls back to the oldest observation when the
    history is shorter than the window.
    """
    h = hist.get(uid)
    if not h or len(h) < 2:
        return 0.0
    t_close, r_close = h[-1]
    t_target = datetime.fromtimestamp(
        t_close.timestamp() - VEL_WINDOW_H * 3600.0, tz=timezone.utc
    )
    # Bracketing observations around the target time.
    before = None
    after = None
    for t, r in h:
        if t <= t_target:
            before = (t, r)
        if t >= t_target and after is None:
            after = (t, r)
    if before is not None and after is not None and after[0] > before[0]:
        f = (t_target - before[0]).total_seconds() / (
            after[0] - before[0]
        ).total_seconds()
        log_then = math.log(before[1]) + f * (
            math.log(after[1]) - math.log(before[1])
        )
        dt = VEL_WINDOW_H
    elif before is not None:
        log_then = math.log(before[1])
        dt = (t_close - before[0]).total_seconds() / 3600.0
    elif after is not None:
        log_then = math.log(after[1])
        dt = (t_close - after[0]).total_seconds() / 3600.0
    else:
        return 0.0
    if dt <= 0:
        return 0.0
    return (math.log(r_close) - log_then) / dt


# Bin -> rank-k overlap fractions, precomputed once.
def rank_overlap():
    """overlap[j][k] = fraction of bin j's mass at rank k (k = 0..9 for places 1..10)."""
    ov = [[0.0] * 10 for _ in range(N_BINS)]
    for j in range(N_BINS):
        lo_e, hi_e = BIN_EDGES[j], BIN_EDGES[j + 1]
        for k in range(1, 11):
            lo_r = math.log(k - 0.5) if k > 1 else -1e9
            hi_r = math.log(k + 0.5)
            inter = max(0.0, min(hi_e, hi_r) - max(lo_e, lo_r))
            ov[j][k - 1] = inter / BIN_W
    return ov


OVERLAP = rank_overlap()


def forward(uid, rank0, vel, age_days, params, steps):
    """Run one game's OU diffusion for `steps` hours.

    Returns (q, total_top10) where q[k] is the time-averaged probability of
    holding place k+1 (k = 0..9) and total_top10 is the summed top-10 mass.
    """
    theta = params["theta"]
    mu_eq = params["mu_eq"]
    beta = params["beta"]
    sigmas = params["sigmas"]
    hazard = params["hazard"]
    ag = age_group(age_days)

    # Initial distribution: point mass in the bin holding ln(rank0).
    dist = [0.0] * N_BINS
    x0 = math.log(max(1, rank0))
    i0 = min(N_BINS - 1, max(0, int(x0 / BIN_W)))
    dist[i0] = 1.0
    off = 0.0

    acc = [0.0] * 10
    for _ in range(steps):
        new = [0.0] * N_BINS
        new_off = off
        for i, m in enumerate(dist):
            if m <= 1e-12:
                continue
            x = BIN_CENTERS[i]
            mean = x + theta * (mu_eq - x) + beta * vel
            sd = sigmas[bucket_index(max(1, int(round(math.exp(x)))))]
            if sd <= 1e-9:
                j = min(N_BINS - 1, max(0, int(mean / BIN_W)))
                new[j] += m
                continue
            lo_j = max(0, int((mean - 4.0 * sd) / BIN_W))
            hi_j = min(N_BINS - 1, int((mean + 4.0 * sd) / BIN_W))
            # CDF at the edges of the covered bins.
            for j in range(lo_j, hi_j + 1):
                w = norm_cdf((BIN_EDGES[j + 1] - mean) / sd) - norm_cdf(
                    (BIN_EDGES[j] - mean) / sd
                )
                if w > 0:
                    new[j] += m * w
            # Mass below rank 1 folds into bin 0; mass past the top folds OFF.
            below = norm_cdf((BIN_EDGES[0] - mean) / sd)
            if below > 0:
                new[0] += m * below
            above = 1.0 - norm_cdf((BIN_EDGES[N_BINS] - mean) / sd)
            if above > 0:
                new_off += m * above
        # Exit hazard: mass leaves the chart from each bin by rank group.
        kept = [0.0] * N_BINS
        for j, m in enumerate(new):
            if m <= 0:
                continue
            rank_j = max(1, int(round(math.exp(BIN_CENTERS[j]))))
            h = hazard.get((haz_rank_group(rank_j), ag), HAZARD_FALLBACK)
            kept[j] = m * (1.0 - h)
            new_off += m * h
        dist = kept
        off = new_off
        for k in range(10):
            s = 0.0
            for j, m in enumerate(dist):
                if m > 0 and OVERLAP[j][k] > 0:
                    s += m * OVERLAP[j][k]
            acc[k] += s
    q = [a / steps for a in acc]
    return q, sum(q)


def main():
    task, snaps = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    steps = int(task.get("window", {}).get("slots", 72))
    output_name = task.get("output", "forecast.json")
    if not snaps:
        raise SystemExit("no snapshots in history")
    hist = game_histories(snaps)
    close_time = parse_time(snaps[-1]["at"])
    close_chart = sorted(
        snaps[-1].get("up_and_coming") or [], key=lambda r: r["rank"]
    )
    close_rank = {r["universe_id"]: r["rank"] for r in close_chart}
    params = calibrate(snaps, hist, close_time)

    # Forward-simulate every game on the close chart (sorted for determinism).
    results = {}
    for uid in sorted(close_rank):
        vel = velocity(uid, hist, close_time)
        q, tot = forward(
            uid, close_rank[uid], vel, params["ages"].get(uid, 0.0),
            params, steps,
        )
        results[uid] = (q, tot)

    # Per-place normalisation with the newcomer reserve.
    newcomer = params["newcomer"]
    probabilities = []
    for k in range(places):
        raw = {
            uid: q[k] for uid, (q, _tot) in results.items() if q[k] >= CUTOFF
        }
        total = sum(raw.values())
        if total <= 0:
            continue
        share = 1.0 - newcomer[k]
        for uid in sorted(raw):
            p = raw[uid] * share / total
            if p >= 0.0005:
                probabilities.append(
                    {"universe_id": uid, "place": k + 1, "p": round(p, 6)}
                )
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # Top 10: highest total top-10 occupancy, ties by close rank then id.
    ranked = sorted(
        results.items(),
        key=lambda kv: (-kv[1][1], close_rank[kv[0]], kv[0]),
    )
    top10 = [uid for uid, _ in ranked[:places]]
    if len(top10) < places:
        for r in close_chart:
            if r["universe_id"] not in top10:
                top10.append(r["universe_id"])
            if len(top10) >= places:
                break

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w") as f:
        f.write(json.dumps(forecast) + "\n")
    return forecast


if __name__ == "__main__":
    main()
