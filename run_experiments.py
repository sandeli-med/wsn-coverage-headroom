"""
run_experiments.py — runs the full experiment of the revised manuscript and
writes results/paper_numbers.json + per-run logs.

Usage:  python3 run_experiments.py [outdir]
"""
from __future__ import annotations
import json, sys, time, os
import numpy as np
from scipy.stats import wilcoxon, pearsonr, spearmanr

import core
from core import (Oracle, TARGETS, STATICS, kmax, cr_ub, run_lattice,
                  run_greedy, run_vfa, run_pso, run_ea, run_cgwo, cs95,
                  n_components)

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(OUT, exist_ok=True)

REGIMES = {  # name: (Nm, r, Gmax, seed_base)
    "A": (10, 10.0, 50, 4000),
    "B": (15, 15.0, 50, 2000),
    "C": (20, 20.0, 30, 6000),
}
POP = 40
NREP = 30
METHOD_IDX = {"VFA": 1, "PSO": 2, "GA": 3, "ES": 4, "GA_LR": 5, "GA_LR_Rnd": 6,
              "GA_LR_S": 7, "GA_LR_S_inv": 8, "CGWO_unc": 9, "CGWO_cor": 10,
              "GA_parity": 11}

def seed_for(base, rep, m):  # deterministic scheme of Eq. (seed)
    return base + 101 * rep + 7 * METHOD_IDX[m]

def init_pop_for(base, rep, n_mobile):
    rng = np.random.default_rng(base + 101 * rep)     # shared across methods
    return rng.uniform(0.0, core.FIELD, size=(POP, n_mobile, 2))

def summarize(runs, orc, n_mobile, rc):
    out = {}
    for k in ("cr", "oracle", "ct", "disp"):
        v = np.array([getattr(r, k) for r in runs], float)
        out[k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
    out["cs95"] = float(np.mean([cs95(r.hist) for r in runs]))
    out["kappa"] = float(np.mean([n_components(np.vstack([STATICS, r.mob]), rc)
                                  for r in runs]))
    out["cr_all"] = [float(r.cr) for r in runs]
    return out

def run_regime(name, res=None, only=None, save_cb=None):
    Nm, r, Gmax, base = REGIMES[name]
    rc = 2 * r
    if res is None:
        res = {}
    curves = res.get("_curves", {})
    # deterministic baselines
    if only is not None and "det" not in only:
        pass
    else:
     orc = Oracle(TARGETS, STATICS, r)
     lat = run_lattice(orc, Nm)
     res["Lattice"] = {"cr": [lat.cr, 0.0], "oracle": [1, 0], "ct": [lat.ct, 0.0],
                       "disp": [0.0, 0.0], "cs95": 0.0,
                       "kappa": float(n_components(np.vstack([STATICS, lat.mob]), rc)),
                       "cr_all": [lat.cr]}
     orc = Oracle(TARGETS, STATICS, r)
     gre = run_greedy(orc, Nm, 60)
     res["Greedy"] = {"cr": [gre.cr, 0.0], "oracle": [Nm + 1, 0], "ct": [gre.ct, 0.0],
                      "disp": [0.0, 0.0], "cs95": 0.0,
                      "kappa": float(n_components(np.vstack([STATICS, gre.mob]), rc)),
                      "cr_all": [gre.cr]}
     res["_greedy_mob"] = gre.mob.tolist()
     res["Greedy"]["gain_evals"] = int(gre.gain_evals)
     # greedy resolution study
     res["_greedy_res"] = {}
     for gs in (20, 60, 100):
         o2 = Oracle(TARGETS, STATICS, r)
         g2 = run_greedy(o2, Nm, gs)
         res["_greedy_res"][gs * gs] = g2.cr
    # stochastic methods
    meths = ("VFA", "PSO", "GA", "ES", "GA_LR", "GA_LR_Rnd", "GA_LR_S",
             "GA_LR_S_inv", "CGWO_unc", "CGWO_cor")
    for m in meths:
        if only is not None and m not in only:
            continue
        if m in res:
            continue
        runs = []
        for rep in range(NREP):
            rng = np.random.default_rng(seed_for(base, rep, m))
            P0 = init_pop_for(base, rep, Nm)
            orc = Oracle(TARGETS, STATICS, r)
            if m == "VFA":
                f0 = orc.f_batch(P0, count=False)
                best0 = P0[int(np.argmax(f0))]
                out = run_vfa(orc, Nm, best0, 50)
            elif m == "PSO":
                out = run_pso(orc, Nm, P0, Gmax, rng)
            elif m in ("GA", "ES", "GA_LR", "GA_LR_Rnd", "GA_LR_S", "GA_LR_S_inv"):
                out = run_ea(orc, Nm, P0, Gmax, rng, method=m)
            elif m == "CGWO_unc":
                out = run_cgwo(orc, Nm, P0, Gmax, rng, corrected=False)
            else:
                out = run_cgwo(orc, Nm, P0, Gmax, rng, corrected=True)
            runs.append(out)
        res[m] = summarize(runs, orc, Nm, rc)
        if m in ("GA", "ES", "PSO", "GA_LR", "GA_LR_S", "CGWO_cor", "CGWO_unc",
                 "GA_LR_Rnd"):
            H = np.array([r_.hist for r_ in runs])
            curves[m] = {"mean": H.mean(axis=0).tolist(),
                         "std": H.std(axis=0, ddof=1).tolist()}
        if m == "CGWO_unc":
            res[m]["reject_rate"] = float(np.mean([r_.reject_rate for r_ in runs]))
            res[m]["stuck_frac"] = float(np.mean([r_.stuck_at_init for r_ in runs]))
        if m in ("GA_LR_S", "GA_LR_S_inv"):
            allr = np.concatenate([r_.rhos for r_ in runs])
            res[m]["rho"] = [float(allr.mean()), float(allr.std(ddof=1))]
            res[m]["rho_by_gen"] = np.array([r_.rhos for r_ in runs]).mean(axis=0).tolist()
        # keep final layouts of GA_LR for discretisation + figure
        if m == "GA_LR":
            res["_galr_mobs"] = [r_.mob.tolist() for r_ in runs]
        res["_curves"] = curves
        if save_cb: save_cb(res)
    res["_curves"] = curves
    return res

def wilcoxon_table(reg):
    comps = [
        ("A", "GA", "GA_LR"), ("B", "GA", "GA_LR"), ("C", "GA", "GA_LR"),
        ("B", "GA_LR", "GA_LR_S"), ("B", "GA_LR_Rnd", "GA_LR_S"),
        ("C", "GA_LR", "GA_LR_S"), ("C", "GA_LR_Rnd", "GA_LR_S"),
        ("C", "GA_LR", "GA_LR_S_inv"),
        ("A", "CGWO_unc", "CGWO_cor"), ("B", "CGWO_unc", "CGWO_cor"),
        ("C", "CGWO_unc", "CGWO_cor"),
    ]
    rows = []
    rng = np.random.default_rng(12345)
    for (R, m1, m2) in comps:
        a = np.array(reg[R][m1]["cr_all"], float)
        b = np.array(reg[R][m2]["cr_all"], float)
        dcr = b.mean() - a.mean()
        # Paired design: identical initial populations per replicate, so the
        # Wilcoxon signed-rank test on the within-replicate differences is the
        # matching test. All-zero differences (saturated instance) -> p = 1.
        if np.allclose(a, b):
            p = 1.0
        else:
            p = float(wilcoxon(a, b, zero_method="zsplit").pvalue)
        # A12 of m2 over m1
        gt = (b[:, None] > a[None, :]).mean()
        eq = (b[:, None] == a[None, :]).mean()
        a12 = float(gt + 0.5 * eq)
        # paired bootstrap CI on the mean within-replicate difference
        n = len(a)
        idx = rng.integers(0, n, size=(10000, n))
        d_pair = b - a
        diffs = d_pair[idx].mean(axis=1)
        ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        # paired win rate: fraction of replicates on which m2 beats m1
        win = float((b > a).mean() + 0.5 * (b == a).mean())
        rows.append({"regime": R, "m1": m1, "m2": m2, "dcr": float(dcr),
                     "p_raw": p, "a12": a12, "win": win, "ci": ci})
    # Holm adjustment over the family
    ps = np.array([r["p_raw"] for r in rows])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    for r, p in zip(rows, adj):
        r["p_holm"] = float(p)
    return rows

def run_sweep(numbers=None, save_cb=None):
    """16-cell sweep, 15 reps, GA / GA-LR (+ deterministic greedy)."""
    cells = numbers.get("sweep_cells", []) if numbers is not None else []
    done = {(c["Nm"], c["r"]) for c in cells}
    for Nm in (5, 10, 15, 20):
        for r in (8.0, 12.0, 16.0, 20.0):
            if (Nm, r) in done:
                continue
            km = kmax(r)
            ub = cr_ub(10 + Nm, r, km)
            base = 8000 + Nm * 97 + int(r) * 13
            ga, galr = [], []
            for rep in range(15):
                P0 = init_pop_for(base, rep, Nm)
                for m, store in (("GA", ga), ("GA_LR", galr)):
                    rng = np.random.default_rng(seed_for(base, rep, m))
                    orc = Oracle(TARGETS, STATICS, r)
                    out = run_ea(orc, Nm, P0, 50, rng, method=m)
                    store.append(out.cr)
            o2 = Oracle(TARGETS, STATICS, r)
            gre = run_greedy(o2, Nm, 60)
            ga = np.array(ga); galr = np.array(galr)
            cells.append({"Nm": Nm, "r": r, "kmax": km, "ub": ub,
                          "ga": [float(ga.mean()), float(ga.std(ddof=1))],
                          "galr": [float(galr.mean()), float(galr.std(ddof=1))],
                          "greedy": float(gre.cr)})
            if numbers is not None:
                numbers["sweep_cells"] = cells
                if save_cb: save_cb(numbers)
    return cells

def run_equal_budget(reg):
    out = {}
    for R, (Nm, r, Gmax, base) in REGIMES.items():
        O_h = reg[R]["GA_LR"]["oracle"][0]
        gens = int(round((O_h - POP) / POP))
        crs = []
        for rep in range(NREP):
            rng = np.random.default_rng(seed_for(base, rep, "GA_parity"))
            P0 = init_pop_for(base, rep, Nm)
            orc = Oracle(TARGETS, STATICS, r)
            o = run_ea(orc, Nm, P0, gens, rng, method="GA")
            crs.append(o.cr)
        crs = np.array(crs)
        out[R] = {"gens": gens, "ga_cr": [float(crs.mean()), float(crs.std(ddof=1))],
                  "galr_cr": reg[R]["GA_LR"]["cr"]}
    return out

def run_k_sensitivity():
    Nm, r, Gmax, base = REGIMES["C"]
    out = {}
    for K in (0.05, 0.10, 0.20, 0.30, 0.50):
        crs, os_, cts = [], [], []
        for rep in range(15):
            rng = np.random.default_rng(seed_for(base + 500, rep, "GA_LR"))
            P0 = init_pop_for(base + 500, rep, Nm)
            orc = Oracle(TARGETS, STATICS, r)
            o = run_ea(orc, Nm, P0, Gmax, rng, method="GA_LR", K=K)
            crs.append(o.cr); os_.append(o.oracle); cts.append(o.ct)
        out[str(int(K * 100))] = {"cr": [float(np.mean(crs)), float(np.std(crs, ddof=1))],
                                  "oracle": float(np.mean(os_)),
                                  "ct": [float(np.mean(cts)), float(np.std(cts, ddof=1))]}
    return out

def run_discretisation(reg):
    out = {}
    for R, (Nm, r, Gmax, base) in REGIMES.items():
        mobs = [np.array(m) for m in reg[R]["_galr_mobs"]]
        row = {}
        for side in (20, 40, 80, 160):
            T = core.targets_grid(side)
            orc = Oracle(T, STATICS, r)
            vals = [orc.f(m, count=False) for m in mobs]
            row[side * side] = [float(np.mean(vals)), float(np.std(vals, ddof=1))]
        out[R] = row
    return out

def run_cgwo_validation():
    """Corrected CGWO on Shaikh et al. Case 1: 20 nodes, 100x100 m, R=20 m.
    Coverage objective on a 100x100 monitoring grid (grid-cell centres)."""
    L = 100.0
    side = 50
    step = L / side
    xs = (np.arange(side) + 0.5) * step
    X, Y = np.meshgrid(xs, xs)
    T = np.column_stack([X.ravel(), Y.ravel()])
    S = np.zeros((0, 2))
    crs = []
    for rep in range(15):
        rng = np.random.default_rng(999 + rep)
        orc = Oracle(T, S, 20.0)
        core_field = core.FIELD
        core.FIELD = L                     # temporary field bound
        try:
            P0 = rng.uniform(0, L, size=(30, 20, 2))
            o = run_cgwo(orc, 20, P0, 60, rng, corrected=True)
        finally:
            core.FIELD = core_field
        crs.append(o.cr)
    crs = np.array(crs)
    return {"published": 95.9077, "ours": [float(crs.mean()), float(crs.std(ddof=1))],
            "diff_pp": float(crs.mean() - 95.9077),
            "rel_err_pct": float(abs(crs.mean() - 95.9077) / 95.9077 * 100)}

def load_partial():
    p = os.path.join(OUT, "partial.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def save_partial(numbers):
    with open(os.path.join(OUT, "partial.json"), "w") as f:
        json.dump(numbers, f)

def main():
    stage = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    numbers = load_partial()
    if stage in ("all", "kmax"):
        numbers["kmax"] = {str(r): kmax(float(r)) for r in (8, 10, 12, 15, 16, 20)}
        print("kmax:", numbers["kmax"], flush=True); save_partial(numbers)
    if stage.startswith("reg"):
        numbers.setdefault("regimes", {})
        spec = stage[3:]
        R = spec[0]
        only = spec[2:].split(",") if ":" in spec else None
        res = numbers["regimes"].get(R, {})
        def cb(r_):
            numbers["regimes"][R] = r_
            save_partial(numbers)
        print(f"--- regime {R} only={only}", time.time() - t0, flush=True)
        numbers["regimes"][R] = run_regime(R, res=res, only=only, save_cb=cb)
        save_partial(numbers)
    if stage == "sweep":
        run_sweep(numbers, save_partial)
        numbers["sweep"] = numbers["sweep_cells"]; save_partial(numbers)
    if stage not in ("all", "final"):
        print("STAGE DONE", stage, time.time() - t0, flush=True); return
    reg = numbers["regimes"]
    print("--- stats", time.time() - t0, flush=True)
    numbers["stats"] = wilcoxon_table(reg); save_partial(numbers)
    print("--- sweep", time.time() - t0, flush=True)
    if "sweep" not in numbers:
        numbers["sweep"] = run_sweep(numbers, save_partial); save_partial(numbers)
    # headroom fit
    h = np.array([c["ub"] - c["ga"][0] for c in numbers["sweep"]])
    gain = np.array([c["galr"][0] - c["ga"][0] for c in numbers["sweep"]])
    pr, pp = pearsonr(h, gain); sr, sp = spearmanr(h, gain)
    slope, intercept = np.polyfit(h, gain, 1)
    numbers["fit"] = {"pearson": [float(pr), float(pp)],
                      "spearman": [float(sr), float(sp)],
                      "slope": float(slope), "intercept": float(intercept),
                      "root": float(-intercept / slope) if slope != 0 else None,
                      "greedy_wins": int(sum(c["greedy"] >= c["galr"][0] - 1e-9
                                             for c in numbers["sweep"]))}
    print("--- equal budget", time.time() - t0, flush=True)
    if "equal_budget" not in numbers:
        numbers["equal_budget"] = run_equal_budget(reg); save_partial(numbers)
    print("--- K sensitivity", time.time() - t0, flush=True)
    if "k_sens" not in numbers:
        numbers["k_sens"] = run_k_sensitivity(); save_partial(numbers)
    print("--- discretisation", time.time() - t0, flush=True)
    numbers["discretisation"] = run_discretisation(reg); save_partial(numbers)
    print("--- cgwo validation", time.time() - t0, flush=True)
    numbers["cgwo_validation"] = run_cgwo_validation(); save_partial(numbers)
    numbers["_elapsed_sec"] = time.time() - t0
    with open(os.path.join(OUT, "paper_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    print("DONE", time.time() - t0, flush=True)

if __name__ == "__main__":
    main()
