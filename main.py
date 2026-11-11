#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster.

Entry for make.fun bounty roblox-up-and-coming-forecast-01.
Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Stdlib only, no network, fully deterministic (same input -> byte-identical output).

Model: per-game Ornstein-Uhlenbeck forward diffusion over log-rank.
  * Each game's rank is a distribution over log-rank bins. Every hour it
    mean-reverts toward a common equilibrium (log-rank ~ ln 35: #1 falls fast,
    deep ranks creep up), diffuses (rank-bucket volatility) and carries
    momentum (recent log-rank velocity, slope ~0.21 from the history's
    triples regression). The variance of log-rank changes saturates with lag
    (0.094 at 6h -> 0.87 at 100h for top-20 games) and the log-rank
    autocorrelation decays (0.97 -> 0.50), which is mean reversion, not a
    random walk: a plain diffusion spreads far too wide over 72h.
  * A game can fall off the chart: per-hour exit hazard by rank group x age
    group, calibrated from the input history itself (old + deep games exit
    fastest; top-10 young games essentially never exit).
  * The 72 hourly steps are averaged, so p(game, place) is the time-averaged
    probability that the game holds that place over the scored window.
  * Games are named at a place only when their averaged probability clears the
    cutoff (about the unnamed per-game share); everything else is left
    unnamed, so the unnamed mass covers both the long tail and brand-new games
    that enter the chart during the window (unknowable from history).

Calibration (OU rate/equilibrium, volatility, momentum, hazard, newcomer
shares) is re-estimated from the input history at every run, with fallbacks
from the practice-history analysis when a cell has too little data, so the
same code adapts to the final history's denser hourly recorder reads.
"""

import json
import math
import os
import sys
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables (defaults tuned on the practice window; see BUILD_LOG.md) ---
CUTOFF = float(os.environ.get("FORECAST_CUTOFF", "0.003"))
VEL_WINDOW_H = float(os.environ.get("FORECAST_VEL_WINDOW_H", "36"))
VEL_CLIP = 0.12
BETA_CLIP = (0.0, 0.4)
SIGMA_MULT = float(os.environ.get("FORECAST_SIGMA_MULT", "1.0"))
THETA_OVERRIDE = os.environ.get("FORECAST_THETA", "")
BETA_OVERRIDE = os.environ.get("FORECAST_BETA", "")

# --- fallback calibration (practice history, 2026-10-30..2026-11-07) ---
# (rank_lo, rank_hi, mu per hour, sigma per sqrt(hour))
FALLBACK_DRIFT = [
    (1, 1, 0.0297, 0.1302),
    (2, 3, 0.0070, 0.1878),
    (4, 5, -0.0026, 0.1124),
    (6, 10, 0.0091, 0.1174),
    (11, 20, 0.0039, 0.1010),
    (21, 50, -0.0005, 0.0904),
    (51, 10 ** 9, -0.0037, 0.0619),
]
FALLBACK_BETA = 0.213
FALLBACK_THETA = 0.010  # per-hour mean-reversion rate
FALLBACK_MU = 3.56  # equilibrium log-rank (~ rank 35)
# per-hour exit hazard by (rank group, age group); age = days since created
# (or since first seen on chart when created is unknown)
FALLBACK_HAZARD = {
    ("top10", "young"): 0.0005,
    ("top10", "old"): 0.0025,
    ("mid", "young"): 0.0005,
    ("mid", "old"): 0.0031,
    ("deep", "young"): 0.0117,
    ("deep", "old"): 0.0158,
}
AGE_CUT_DAYS = 21.0

# --- log-rank grid ---
BIN_W = 0.1
NBIN = 52  # log-rank 0 .. 5.2  (ranks 1 .. ~181)
OFF = NBIN  # index of the off-chart state
NSTATES = NBIN + 1
VEL_BINS = [-0.09, -0.045, 0.0, 0.045, 0.09]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rank_group(rank):
    if rank <= 10:
        return "top10"
    if rank <= 50:
        return "mid"
    return "deep"


def drift_sigma_for_rank(rank, drift_table):
    for lo, hi, mu, sig in drift_table:
        if lo <= rank <= hi:
            return mu, sig
    return drift_table[-1][2], drift_table[-1][3]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    # per snapshot: time, {uid: rank}, {uid: created datetime or None}
    parsed = []
    for s in snaps:
        t = parse_time(s["at"])
        ranks = {}
        for r in s.get("up_and_coming") or []:
            ranks[int(r["universe_id"])] = int(r["rank"])
        created = {}
        det = s.get("details") or {}
        if isinstance(det, dict):
            for k, v in det.items():
                if isinstance(v, dict) and v.get("created"):
                    try:
                        created[int(k)] = parse_time(v["created"])
                    except Exception:
                        pass
        parsed.append((t, ranks, created))
    return task, parsed


def calibrate(snaps):
    """Estimate OU rate/equilibrium, volatility, momentum, hazard from the history."""
    drift_table = [list(row) for row in FALLBACK_DRIFT]
    beta = FALLBACK_BETA
    hazard = dict(FALLBACK_HAZARD)
    theta = FALLBACK_THETA
    mu_eq = FALLBACK_MU

    # --- OU rate/equilibrium: regress log-rank at t+L on log-rank at t ---
    lag_buckets = [(4, 10), (10, 18), (18, 36), (36, 60), (60, 100)]
    thetas, mus = [], []
    for lo, hi in lag_buckets:
        xs, ys = [], []
        for i, (ta, ra, _) in enumerate(snaps):
            for (tb, rb, _) in snaps[i + 1:]:
                lag = (tb - ta).total_seconds() / 3600.0
                if lo <= lag < hi:
                    for uid in set(ra) & set(rb):
                        xs.append(math.log(ra[uid]))
                        ys.append(math.log(rb[uid]))
        if len(xs) >= 200:
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            if den > 1e-9:
                b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
                a = my - b * mx
                lag_mid = (lo + hi) / 2.0
                if 0.05 < b < 0.99:
                    thetas.append(-math.log(b) / lag_mid)
                    mus.append(a / (1.0 - b))
    if thetas:
        thetas.sort()
        mus.sort()
        theta = max(0.003, min(0.03, thetas[len(thetas) // 2]))
        mu_eq = max(math.log(10), min(math.log(80), mus[len(mus) // 2]))
    if THETA_OVERRIDE:
        theta = float(THETA_OVERRIDE)

    # --- drift & volatility per rank bucket ---
    buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)]
    acc = {b: [] for b in buckets}
    for (t0, r0, _), (t1, r1, _) in zip(snaps, snaps[1:]):
        dt = (t1 - t0).total_seconds() / 3600.0
        if dt <= 0:
            continue
        for uid in set(r0) & set(r1):
            dlog = math.log(r1[uid] / r0[uid])
            for b in buckets:
                if b[0] <= r0[uid] <= b[1]:
                    acc[b].append((dlog, dt))
                    break
    for i, b in enumerate(buckets):
        v = acc[b]
        if len(v) >= 20:
            mus = [d / dt for d, dt in v]
            sigs = [d / math.sqrt(dt) for d, dt in v]
            mu = sum(mus) / len(mus)
            mean_s = sum(sigs) / len(sigs)
            var_s = sum((x - mean_s) ** 2 for x in sigs) / len(sigs)
            sigma = math.sqrt(max(var_s, 1e-6))
            drift_table[i] = [b[0], b[1],
                              max(-0.05, min(0.05, mu)),
                              max(0.03, min(0.30, sigma))]
    # --- momentum: slope of next velocity on past velocity (triples) ---
    xs, ys = [], []
    for i in range(2, len(snaps)):
        (ta, ra, _), (tb, rb, _), (tc, rc, _) = snaps[i - 2], snaps[i - 1], snaps[i]
        dt1 = (tb - ta).total_seconds() / 3600.0
        dt2 = (tc - tb).total_seconds() / 3600.0
        if dt1 <= 0 or dt2 <= 0:
            continue
        for uid in set(ra) & set(rb) & set(rc):
            xs.append(math.log(rb[uid] / ra[uid]) / dt1)
            ys.append(math.log(rc[uid] / rb[uid]) / dt2)
    if len(xs) >= 100:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        if den > 1e-9:
            beta = max(BETA_CLIP[0], min(BETA_CLIP[1], num / den))
    if BETA_OVERRIDE:
        beta = float(BETA_OVERRIDE)
    # --- exit hazard per (rank group, age group), per hour ---
    # exposure in hours, exits counted; shrinkage toward fallback
    cells = {}
    for rg in ("top10", "mid", "deep"):
        for ag in ("young", "old"):
            cells[(rg, ag)] = [0.0, 0]  # [exposure_hours, exits]
    first_seen = {}
    created_map = {}
    for (t, ranks, created) in snaps:
        for uid in ranks:
            first_seen.setdefault(uid, t)
        for uid, c in created.items():
            created_map.setdefault(uid, c)
    for (t0, r0, _), (t1, r1, _) in zip(snaps, snaps[1:]):
        dt = (t1 - t0).total_seconds() / 3600.0
        if dt <= 0:
            continue
        for uid, rk in r0.items():
            c = created_map.get(uid)
            if c is not None:
                age = (t0 - c).total_seconds() / 86400.0
            else:
                age = (t0 - first_seen[uid]).total_seconds() / 86400.0
            ag = "young" if age < AGE_CUT_DAYS else "old"
            cell = cells[(rank_group(rk), ag)]
            cell[0] += dt
            if uid not in r1:
                cell[1] += 1
    for key, (expo, exits) in cells.items():
        if expo >= 50:
            h = (exits + 0.5) / (expo + 100.0)
            hazard[key] = max(0.0005, min(0.05, h))
    # --- newcomer share per place: fraction of top-10 slots at lag 48-96h
    # occupied by games not on the earlier chart (shrunk toward the global mean)
    new_k = {k: [0, 0] for k in range(1, 11)}  # [new, total]
    for i, (ta, ra, _) in enumerate(snaps):
        for (tb, rb, _) in snaps[i + 1:]:
            lag = (tb - ta).total_seconds() / 3600.0
            if 48.0 <= lag <= 96.0:
                for k in range(1, 11):
                    row = sorted(rb.items(), key=lambda kv: kv[1])
                    if len(row) >= k:
                        uid_k = row[k - 1][0]
                        new_k[k][1] += 1
                        if uid_k not in ra:
                            new_k[k][0] += 1
    tot_new = sum(v[0] for v in new_k.values())
    tot_n = sum(v[1] for v in new_k.values())
    glob = (tot_new / tot_n) if tot_n else 0.15
    newcomer = {}
    for k in range(1, 11):
        n_new, n_tot = new_k[k]
        emp = (n_new / n_tot) if n_tot else glob
        newcomer[k] = max(0.05, min(0.40, 0.3 * emp + 0.7 * glob))
    return drift_table, beta, hazard, newcomer, theta, mu_eq


def velocity(snaps, uid, t_end):
    """Least-squares slope of log(rank) vs hours over the recent window."""
    pts = []
    for (t, ranks, _) in snaps:
        if uid in ranks:
            age_h = (t_end - t).total_seconds() / 3600.0
            if age_h <= VEL_WINDOW_H:
                pts.append((age_h, math.log(ranks[uid])))
    if len(pts) < 2:
        # fall back to all observations
        pts = [( (t_end - t).total_seconds() / 3600.0, math.log(ranks[uid]))
               for (t, ranks, _) in snaps if uid in ranks]
    if len(pts) < 2:
        return 0.0
    # slope of log-rank vs time-ago (negative age_h = recent); fit vs -age_h
    xs = [-a for a, _ in pts]
    ys = [y for _, y in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-9:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return max(-VEL_CLIP, min(VEL_CLIP, slope))


def build_kernels(drift_table, beta, theta, mu_eq):
    """K[vbin][src_bin] = [(tgt_bin, prob), ...]; OFF is index NBIN.
    OU step: mean = center + theta*(mu_eq - center) + beta*vb, sd = bucket sigma."""
    kernels = []
    for vb in VEL_BINS:
        rows = []
        for src in range(NBIN):
            center = (src + 0.5) * BIN_W
            rank_c = math.exp(center)
            _, sig = drift_sigma_for_rank(rank_c, drift_table)
            m = center + theta * (mu_eq - center) + beta * vb
            s = max(sig * SIGMA_MULT, 0.03)
            row = []
            # target bins within +-3.5 sigma
            lo = max(0, int(math.floor((m - 3.5 * s) / BIN_W)))
            hi = min(NBIN - 1, int(math.floor((m + 3.5 * s) / BIN_W)))
            for tgt in range(lo, hi + 1):
                p = norm_cdf(((tgt + 1) * BIN_W - m) / s) - norm_cdf((tgt * BIN_W - m) / s)
                if p > 1e-7:
                    row.append((tgt, p))
            # mass below 0 folds into bin 0; mass above grid goes OFF
            p_below = norm_cdf((0.0 - m) / s)
            if p_below > 1e-7:
                row = [(0, p_below + (row[0][1] if row and row[0][0] == 0 else 0.0))] + \
                      [(t, p) for (t, p) in row if t != 0]
            p_above = 1.0 - norm_cdf((NBIN * BIN_W - m) / s)
            if p_above > 1e-7:
                row.append((OFF, p_above))
            # renormalise (truncation); never let a row vanish
            tot = sum(p for _, p in row)
            if tot > 0:
                row = [(t, p / tot) for (t, p) in row]
            else:
                row = [(OFF, 1.0)]
            rows.append(row)
        kernels.append(rows)
    return kernels


def rank_weights():
    """W[k][bin] = overlap of bin with [ln(k-0.5), ln(k+0.5)) for k=1..10."""
    W = {}
    for k in range(1, 11):
        lo = math.log(k - 0.5)
        hi = math.log(k + 0.5)
        w = []
        for b in range(NBIN):
            blo, bhi = b * BIN_W, (b + 1) * BIN_W
            ov = max(0.0, min(bhi, hi) - max(blo, lo))
            w.append(ov / BIN_W if ov > 0 else 0.0)
        W[k] = w
    return W


def main():
    task, snaps = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    n_slots = int(task.get("window", {}).get("slots", 72))
    out_name = task.get("output", "forecast.json")

    t_end, close_ranks, close_created = snaps[-1]
    drift_table, beta, hazard, newcomer, theta, mu_eq = calibrate(snaps)
    kernels = build_kernels(drift_table, beta, theta, mu_eq)
    W = rank_weights()

    # first-seen + created per game (for age in hazard)
    first_seen = {}
    created_map = {}
    for (t, ranks, created) in snaps:
        for uid in ranks:
            first_seen.setdefault(uid, t)
        for uid, c in created.items():
            created_map.setdefault(uid, c)

    # bin -> rank group (by bin centre rank)
    bin_group = []
    for b in range(NBIN):
        bin_group.append(rank_group(max(1, int(round(math.exp((b + 0.5) * BIN_W))))))

    occ = {}  # uid -> [P(place k) averaged over hours], index 1..places
    for uid, r0 in sorted(close_ranks.items()):
        v = velocity(snaps, uid, t_end)
        vb = min(range(len(VEL_BINS)), key=lambda i: abs(VEL_BINS[i] - v))
        kernel = kernels[vb]
        c = created_map.get(uid)
        if c is not None:
            age0 = (t_end - c).total_seconds() / 86400.0
        else:
            age0 = (t_end - first_seen[uid]).total_seconds() / 86400.0
        # initial distribution: point mass at bin of ln(r0)
        dist = [0.0] * NSTATES
        dist[min(NBIN - 1, int(math.log(max(1, r0)) / BIN_W))] = 1.0
        acc = [0.0] * (places + 1)
        for step in range(1, n_slots + 1):
            # diffuse
            nd = [0.0] * NSTATES
            for src in range(NBIN):
                p_src = dist[src]
                if p_src <= 0.0:
                    continue
                for (tgt, p) in kernel[src]:
                    nd[tgt] += p_src * p
            nd[OFF] += dist[OFF]
            # exit hazard (age grows with time)
            age = age0 + step / 24.0
            ag = "young" if age < AGE_CUT_DAYS else "old"
            off_add = 0.0
            for b in range(NBIN):
                h = hazard[(bin_group[b], ag)]
                keep = nd[b] * (1.0 - h)
                off_add += nd[b] - keep
                nd[b] = keep
            nd[OFF] += off_add
            dist = nd
            # accumulate occupancy of places 1..10
            for k in range(1, places + 1):
                wk = W[k]
                s = 0.0
                for b in range(NBIN):
                    if wk[b]:
                        s += dist[b] * wk[b]
                acc[k] += s
        occ[uid] = [a / n_slots for a in acc]

    # --- allocate probabilities per place ---
    # Independent per-game diffusions do not enforce one game per place, so
    # the raw occupancies sum to more (or less) than 1 at a place. Normalise
    # each place by its raw total, keeping the newcomer share unnamed:
    #   p(g,k) = q(g,k) * (1 - newcomer[k]) / sum_g q(g,k)
    probabilities = []
    for k in range(1, places + 1):
        s_k = sum(occ[uid][k] for uid in occ)
        if s_k <= 0:
            continue
        scale = (1.0 - newcomer.get(k, 0.15)) / s_k
        cand = sorted(((occ[uid][k] * scale, uid) for uid in occ
                       if occ[uid][k] * scale >= CUTOFF),
                      key=lambda x: (-x[0], x[1]))
        for q, uid in cand[:100]:
            p = math.floor(q * 1e6) / 1e6
            if p > 0:
                probabilities.append({"universe_id": int(uid), "place": k, "p": p})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # --- top10 by expected rank (off-top-10 counts as 30) ---
    def exp_rank(uid):
        q = occ[uid]
        s_top = sum(q[1:])
        return sum(k * q[k] for k in range(1, places + 1)) + 30.0 * (1.0 - s_top)
    ranked = sorted(occ.keys(), key=lambda u: (exp_rank(u), -sum(occ[u][1:]), u))
    top10 = [int(u) for u in ranked[:places]]

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True) + "\n")
    # diagnostics to stderr (not into the forecast)
    n_rows = len(probabilities)
    sums = {}
    for k in range(1, places + 1):
        sums[k] = round(sum(r["p"] for r in probabilities if r["place"] == k), 4)
    print(f"games={len(occ)} rows={n_rows} beta={beta:.3f} theta={theta:.4f} "
          f"mu_eq={mu_eq:.2f} (rank {math.exp(mu_eq):.0f}) newcomer={ {k: round(v, 2) for k, v in newcomer.items()} } "
          f"place_sums={sums} cutoff={CUTOFF}", file=sys.stderr)


if __name__ == "__main__":
    main()
