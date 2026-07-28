"""
core.py — problem, methods and metrics for the revised manuscript.

Implements exactly the protocol of the paper:
  field 200x200, |T|=400 targets on a 20x20 lattice spanning [1,200],
  Ns=10 statics = first 10 points (row-major) of a 4x4 lattice spanning [1,200].

Methods: Lattice, Greedy, VFA, PSO, GA, ES, GA-LR, GA-LR-Rnd, GA-LR-S,
GA-LR-S(inv), CGWO uncorrected, CGWO corrected.

Oracle accounting (matches Sec. "budget accounting" of the paper):
  * every evaluation of f counts 1, including each of the 9 refiner probes
  * GA-LR:      per gen  40 (prelim) + K*pop*9*Nm (refiner) + 40 (final)
  * GA-LR-Rnd:  per gen  K*pop*9*Nm + 40
  * GA-LR-S:    per gen  K*pop*9*Nm + 40
  * GA/ES/PSO/CGWO: pop per gen (+pop init)
  * Greedy: Nm+1 ; Lattice: 1 ; VFA: 1 + n_iter
"""
from __future__ import annotations
import time
import numpy as np


# ---------------------------------------------------------------- instance ---
FIELD = 200.0

def targets_grid(n_side: int = 20) -> np.ndarray:
    xs = np.linspace(1.0, FIELD, n_side)
    X, Y = np.meshgrid(xs, xs)
    return np.column_stack([X.ravel(), Y.ravel()])

def static_nodes() -> np.ndarray:
    xs = np.linspace(1.0, FIELD, 4)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts[:10].copy()

TARGETS = targets_grid(20)          # (400,2)
STATICS = static_nodes()            # (10,2)


class Oracle:
    """Coverage oracle with call counting."""
    def __init__(self, targets: np.ndarray, statics: np.ndarray, radius: float):
        self.T = targets
        self.S = statics
        self.r = radius
        self.calls = 0
        # static coverage mask precomputed (statics never move)
        d = np.linalg.norm(self.T[:, None, :] - self.S[None, :, :], axis=2)
        self.static_mask = (d <= radius).any(axis=1)

    def mobile_mask(self, mob: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self.T[:, None, :] - mob[None, :, :], axis=2)
        return (d <= self.r).any(axis=1)

    def f(self, mob: np.ndarray, count: bool = True) -> float:
        if count:
            self.calls += 1
        m = self.static_mask | self.mobile_mask(mob)
        return 100.0 * m.mean()

    def f_batch(self, mobs: np.ndarray, count: bool = True) -> np.ndarray:
        """mobs: (P, Nm, 2). Counts P oracle calls."""
        P = mobs.shape[0]
        if count:
            self.calls += P
        d = np.linalg.norm(self.T[None, :, None, :] - mobs[:, None, :, :], axis=3)
        cov = (d <= self.r).any(axis=2) | self.static_mask[None, :]
        return 100.0 * cov.mean(axis=1)


def kmax(radius: float, step: float = 0.5) -> int:
    """Largest number of lattice targets enclosed by any disk of given radius."""
    xs = np.arange(1.0, FIELD + step / 2, step)
    best = 0
    T = TARGETS
    for x in xs:
        d2 = (T[:, 0] - x) ** 2
        # vector over y-centres in chunks
        ys = xs
        dy2 = (T[:, 1][:, None] - ys[None, :]) ** 2
        cnt = ((d2[:, None] + dy2) <= radius ** 2).sum(axis=0)
        m = int(cnt.max())
        if m > best:
            best = m
    return best


def cr_ub(n_sensors: int, radius: float, kmax_val: int, n_targets: int = 400) -> float:
    return min(100.0, 100.0 * n_sensors * kmax_val / n_targets)


# ------------------------------------------------------------------ helpers ---
def clip_field(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, FIELD)

def displacement(final: np.ndarray, init: np.ndarray) -> float:
    return float(np.linalg.norm(final - init, axis=1).sum())

def n_components(layout_all: np.ndarray, rc: float) -> int:
    n = layout_all.shape[0]
    d = np.linalg.norm(layout_all[:, None, :] - layout_all[None, :, :], axis=2)
    adj = d <= rc
    seen = np.zeros(n, bool)
    comps = 0
    for i in range(n):
        if not seen[i]:
            comps += 1
            stack = [i]
            seen[i] = True
            while stack:
                j = stack.pop()
                nb = np.where(adj[j] & ~seen)[0]
                seen[nb] = True
                stack.extend(nb.tolist())
    return comps

def cs95(history: np.ndarray) -> int:
    """First generation index (1-based; init=0) reaching 95% of final best."""
    final = history[-1]
    thr = 0.95 * final
    idx = int(np.argmax(history >= thr))
    return idx

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


# ------------------------------------------------------------------ result ---
class Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# =============================================================== BASELINES ===
def run_lattice(orc: Oracle, n_mobile: int) -> Result:
    t0 = time.perf_counter()
    rows = int(np.floor(np.sqrt(n_mobile)))
    while n_mobile % rows != 0 and rows > 1:
        rows -= 1
    cols = n_mobile // rows
    xs = (np.arange(cols) + 0.5) * FIELD / cols
    ys = (np.arange(rows) + 0.5) * FIELD / rows
    X, Y = np.meshgrid(xs, ys)
    mob = np.column_stack([X.ravel(), Y.ravel()])[:n_mobile]
    orc.calls = 0
    cr = orc.f(mob)
    return Result(cr=cr, mob=mob, oracle=orc.calls, ct=time.perf_counter() - t0,
                  disp=0.0, hist=np.array([cr]))


def greedy_masks(orc: Oracle, grid_side: int):
    xs = np.linspace(1.0, FIELD, grid_side)
    X, Y = np.meshgrid(xs, xs)
    cand = np.column_stack([X.ravel(), Y.ravel()])
    # mask[c, t] in chunks
    masks = np.zeros((cand.shape[0], orc.T.shape[0]), dtype=bool)
    chunk = 500
    for i in range(0, cand.shape[0], chunk):
        d = np.linalg.norm(orc.T[None, :, :] - cand[i:i + chunk, None, :], axis=2)
        masks[i:i + chunk] = d <= orc.r
    return cand, masks


def run_greedy(orc: Oracle, n_mobile: int, grid_side: int = 60) -> Result:
    t0 = time.perf_counter()
    cand, masks = greedy_masks(orc, grid_side)
    covered = orc.static_mask.copy()
    chosen = []
    for _ in range(n_mobile):
        gains = (masks & ~covered[None, :]).sum(axis=1)
        j = int(np.argmax(gains))
        chosen.append(j)
        covered |= masks[j]
    mob = cand[chosen]
    orc.calls = 0
    cr = orc.f(mob)                      # 1 final oracle call
    orc.calls = n_mobile + 1             # paper convention: one call per placement + final
    return Result(cr=cr, mob=mob, oracle=orc.calls, ct=time.perf_counter() - t0,
                  disp=0.0, hist=np.array([cr]))


def run_vfa(orc: Oracle, n_mobile: int, init_mob: np.ndarray,
            n_iter: int = 50) -> Result:
    """Virtual force algorithm (Zou & Chakrabarty style, pairwise threshold)."""
    t0 = time.perf_counter()
    orc.calls = 0
    mob = init_mob.copy()
    d_th = orc.r * np.sqrt(3.0)          # hexagonal ideal spacing
    max_step = orc.r / 2.0
    best_cr = orc.f(mob)
    best_mob = mob.copy()
    hist = [best_cr]
    allp = np.vstack([STATICS, mob])
    for _ in range(n_iter):
        allp = np.vstack([STATICS, mob])
        F = np.zeros_like(mob)
        for i in range(n_mobile):
            diff = allp - mob[i]
            dist = np.linalg.norm(diff, axis=1)
            mask = dist > 1e-9
            u = np.zeros_like(diff)
            u[mask] = diff[mask] / dist[mask][:, None]
            att = (dist > d_th)[:, None] * u * 0.5 * (dist - d_th)[:, None]
            rep = ((dist < d_th) & mask)[:, None] * u * 2.0 * (dist - d_th)[:, None]
            F[i] = (att + rep).sum(axis=0)
        norm = np.linalg.norm(F, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        step = np.minimum(np.linalg.norm(F, axis=1), max_step)[:, None]
        mob = clip_field(mob + F / norm * step)
        cr = orc.f(mob)
        if cr > best_cr:
            best_cr, best_mob = cr, mob.copy()
        hist.append(best_cr)
    return Result(cr=best_cr, mob=best_mob, oracle=orc.calls,
                  ct=time.perf_counter() - t0,
                  disp=displacement(best_mob, init_mob), hist=np.array(hist))


def run_pso(orc: Oracle, n_mobile: int, init_pop: np.ndarray, gmax: int,
            rng: np.random.Generator) -> Result:
    t0 = time.perf_counter()
    orc.calls = 0
    pop = init_pop.shape[0]
    X = init_pop.reshape(pop, -1).copy()
    V = rng.uniform(-1, 1, X.shape) * 4.0
    fit = orc.f_batch(X.reshape(pop, n_mobile, 2))
    P = X.copy(); Pf = fit.copy()
    g = int(np.argmax(fit)); G = X[g].copy(); Gf = fit[g]
    init_best = init_pop[g].copy()
    hist = [Gf]
    w, c1, c2, vmax = 0.729, 1.49445, 1.49445, 0.2 * FIELD
    for _ in range(gmax):
        r1 = rng.random(X.shape); r2 = rng.random(X.shape)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0.0, FIELD)
        fit = orc.f_batch(X.reshape(pop, n_mobile, 2))
        imp = fit > Pf
        P[imp] = X[imp]; Pf[imp] = fit[imp]
        j = int(np.argmax(Pf))
        if Pf[j] > Gf:
            Gf = Pf[j]; G = P[j].copy()
        hist.append(Gf)
    mob = G.reshape(n_mobile, 2)
    return Result(cr=Gf, mob=mob, oracle=orc.calls, ct=time.perf_counter() - t0,
                  disp=displacement(mob, init_best), hist=np.array(hist))


# ============================================================ EA MACHINERY ===
def ga_offspring(popX: np.ndarray, fit: np.ndarray, n_mobile: int,
                 rng: np.random.Generator, pm: float = 0.10,
                 jitter: float = 4.0) -> np.ndarray:
    pop, D = popX.shape
    # tournament size 3
    idx = rng.integers(0, pop, size=(pop, 3))
    winners = idx[np.arange(pop), np.argmax(fit[idx], axis=1)]
    parents = popX[winners]
    # uniform crossover, rate 1.0, on consecutive pairs
    off = parents.copy()
    for i in range(0, pop - 1, 2):
        mask = rng.random(D) < 0.5
        a, b = parents[i].copy(), parents[i + 1].copy()
        off[i][mask], off[i + 1][mask] = b[mask], a[mask]
    # mutation: deterministic count round(pm*D) coordinates jittered
    k = int(round(pm * D))
    for i in range(pop):
        cols = rng.choice(D, size=k, replace=False)
        off[i, cols] += rng.normal(0.0, jitter, size=k)
    return np.clip(off, 0.0, FIELD)


def refine(orc: Oracle, mob: np.ndarray, delta: float = 6.0) -> np.ndarray:
    """Oracle-gated steepest-ascent sweep; 9 counted probes per node."""
    dirs = np.array([[np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)]
                     for k in range(8)] + [[0.0, 0.0]])
    mob = mob.copy()
    Nm = mob.shape[0]
    # coverage count per target from statics + all mobiles
    d = np.linalg.norm(orc.T[:, None, :] - mob[None, :, :], axis=2)
    node_masks = d <= orc.r                       # (T, Nm)
    counts = orc.static_mask.astype(int) + node_masks.sum(axis=1)
    for i in range(Nm):
        base = counts - node_masks[:, i].astype(int)   # coverage without node i
        cand = clip_field(mob[i][None, :] + delta * dirs)      # (9,2)
        dc = np.linalg.norm(orc.T[:, None, :] - cand[None, :, :], axis=2)
        cmask = dc <= orc.r                                    # (T,9)
        covs = ((base[:, None] > 0) | cmask).mean(axis=0)      # (9,)
        orc.calls += 9
        cur_cov = ((base > 0) | node_masks[:, i]).mean()
        j = int(np.argmax(covs))
        if covs[j] >= cur_cov:                    # ties accepted (plateau moves)
            mob[i] = cand[j]
            new_mask = dc[:, j] <= orc.r
            counts = base + new_mask.astype(int)
            node_masks[:, i] = new_mask
    return mob


def phi_raw(mobs: np.ndarray) -> np.ndarray:
    return mobs.reshape(mobs.shape[0], -1)


def phi_inv(orc: Oracle, mobs: np.ndarray, k: int = 4) -> np.ndarray:
    """Permutation-invariant descriptors (see Sec. method:surrogate)."""
    P, Nm, _ = mobs.shape
    feats = np.zeros((P, k * k + 5))
    edges = np.linspace(0.0, FIELD, k + 1)
    for p in range(P):
        mob = mobs[p]
        d = np.linalg.norm(orc.T[:, None, :] - mob[None, :, :], axis=2)
        cov = orc.static_mask | (d <= orc.r).any(axis=1)
        unc = orc.T[~cov]
        # per-bin uncovered counts
        if unc.shape[0]:
            bx = np.clip(np.digitize(unc[:, 0], edges) - 1, 0, k - 1)
            by = np.clip(np.digitize(unc[:, 1], edges) - 1, 0, k - 1)
            H = np.zeros((k, k))
            np.add.at(H, (bx, by), 1.0)
        else:
            H = np.zeros((k, k))
        feats[p, :k * k] = H.ravel()
        dd = np.linalg.norm(mob[:, None, :] - mob[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(axis=1)
        overlap = np.maximum(0.0, 2 * orc.r - dd[np.triu_indices(Nm, 1)]).sum()
        cen = unc.mean(axis=0) if unc.shape[0] else np.array([FIELD / 2] * 2)
        feats[p, k * k:] = [overlap, nn.mean(), nn.min(),
                            np.linalg.norm(mob - cen, axis=1).mean(),
                            float(((mob < orc.r) | (mob > FIELD - orc.r)).any(axis=1).sum())]
    return feats


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-6):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    w = np.linalg.solve(A, Xa.T @ y)
    return w, mu, sd

def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    Xs = (X - mu) / sd
    return np.hstack([Xs, np.ones((Xs.shape[0], 1))]) @ w


def run_ea(orc: Oracle, n_mobile: int, init_pop: np.ndarray, gmax: int,
           rng: np.random.Generator, method: str = "GA",
           K: float = 0.20, delta: float = 6.0, sigma_es: float = 3.0) -> Result:
    """GA / ES / GA-LR / GA-LR-Rnd / GA-LR-S / GA-LR-S-inv."""
    t0 = time.perf_counter()
    orc.calls = 0
    pop = init_pop.shape[0]
    D = n_mobile * 2
    X = init_pop.reshape(pop, D).copy()
    fit = orc.f_batch(X.reshape(pop, n_mobile, 2))
    g0 = int(np.argmax(fit))
    init_best = init_pop[g0].copy()
    bestX, bestF = X[g0].copy(), fit[g0]
    hist = [bestF]
    n_ref = max(1, int(round(K * pop)))
    surrogate = method in ("GA_LR_S", "GA_LR_S_inv")
    inv = method == "GA_LR_S_inv"
    arch_X, arch_y = [], []
    if surrogate:
        Feat = phi_inv(orc, X.reshape(pop, n_mobile, 2)) if inv else X.copy()
        arch_X.append(Feat); arch_y.append(fit.copy())
    rhos = []
    for g in range(gmax):
        if method == "ES":
            par = X[rng.integers(0, pop, size=pop)]
            off = np.clip(par + rng.normal(0, sigma_es, size=par.shape), 0, FIELD)
            off_fit = orc.f_batch(off.reshape(pop, n_mobile, 2))
            allX = np.vstack([X, off]); allF = np.concatenate([fit, off_fit])
            order = np.argsort(-allF)[:pop]
            X, fit = allX[order], allF[order]
        else:
            off = ga_offspring(X, fit, n_mobile, rng)
            offm = off.reshape(pop, n_mobile, 2)
            if method == "GA":
                pass
            elif method == "GA_LR":
                prelim = orc.f_batch(offm)                 # pop prelim calls
                top = np.argsort(-prelim)[:n_ref]
                for i in top:
                    offm[i] = refine(orc, offm[i], delta)
            elif method == "GA_LR_Rnd":
                top = rng.choice(pop, size=n_ref, replace=False)
                for i in top:
                    offm[i] = refine(orc, offm[i], delta)
            elif surrogate:
                Feat = phi_inv(orc, offm) if inv else off.copy()
                model = ridge_fit(np.vstack(arch_X), np.concatenate(arch_y))
                pred = ridge_predict(model, Feat)
                top = np.argsort(-pred)[:n_ref]
                for i in top:
                    offm[i] = refine(orc, offm[i], delta)
            off = offm.reshape(pop, D)
            off_fit = orc.f_batch(offm)                    # pop final calls
            if surrogate:
                rhos.append(spearman(pred, off_fit))
                Feat2 = phi_inv(orc, offm) if inv else off.copy()
                arch_X.append(Feat2); arch_y.append(off_fit.copy())
            # elitist replacement: offspring + best parent kept
            bi = int(np.argmax(fit))
            wi = int(np.argmin(off_fit))
            if fit[bi] > off_fit[wi]:
                off[wi] = X[bi]; off_fit[wi] = fit[bi]
            X, fit = off, off_fit
        j = int(np.argmax(fit))
        if fit[j] > bestF:
            bestF = fit[j]; bestX = X[j].copy()
        hist.append(bestF)
    mob = bestX.reshape(n_mobile, 2)
    return Result(cr=bestF, mob=mob, oracle=orc.calls, ct=time.perf_counter() - t0,
                  disp=displacement(mob, init_best), hist=np.array(hist),
                  rhos=np.array(rhos) if rhos else None)


# ==================================================================== CGWO ===
def run_cgwo(orc: Oracle, n_mobile: int, init_pop: np.ndarray, gmax: int,
             rng: np.random.Generator, corrected: bool) -> Result:
    t0 = time.perf_counter()
    orc.calls = 0
    pop = init_pop.shape[0]
    D = n_mobile * 2
    X = init_pop.reshape(pop, D).copy()
    fit = orc.f_batch(X.reshape(pop, n_mobile, 2))
    g0 = int(np.argmax(fit))
    init_best = init_pop[g0].copy()
    bestF = fit[g0]; bestX = X[g0].copy()
    hist = [bestF]
    # logistic chaotic state, mu=4, initial in U(0.1,0.9), fixed-point guard
    fixed = {0.0, 0.25, 0.5, 0.75, 1.0}
    x = float(rng.uniform(0.1, 0.9))
    def chaos_scalar():
        nonlocal x
        x = 4.0 * x * (1.0 - x)
        if min(abs(x - fp) for fp in fixed) < 1e-12:
            x = float(rng.uniform(0.1, 0.9))
        return x
    def chaos_array(shape):
        out = np.empty(int(np.prod(shape)))
        for i in range(out.size):
            out[i] = chaos_scalar()
        return out.reshape(shape)
    n_prop = 0; n_rej = 0
    for t in range(gmax):
        a = 2.0 - 2.0 * t / gmax
        order = np.argsort(-fit)
        Xa, Xb, Xd = X[order[0]], X[order[1]], X[order[2]]
        if corrected:
            r1 = chaos_array((3, pop, D))
            r2 = rng.random((3, pop, D))
        else:
            c = chaos_scalar()                     # single scalar for all
            r1 = np.full((3, pop, D), c)
            r2 = rng.random((3, pop, D))
        A = 2 * a * r1 - a
        C = 2 * r2
        X1 = Xa - A[0] * np.abs(C[0] * Xa - X)
        X2 = Xb - A[1] * np.abs(C[1] * Xb - X)
        X3 = Xd - A[2] * np.abs(C[2] * Xd - X)
        Xn = np.clip((X1 + X2 + X3) / 3.0, 0.0, FIELD)
        fn = orc.f_batch(Xn.reshape(pop, n_mobile, 2))
        if corrected:
            X, fit = Xn, fn                        # unconditional replacement
        else:
            n_prop += pop
            acc = fn > fit                         # accept/reject gate
            n_rej += int((~acc).sum())
            X[acc] = Xn[acc]; fit[acc] = fn[acc]
        j = int(np.argmax(fit))
        if fit[j] > bestF:
            bestF = fit[j]; bestX = X[j].copy()
        hist.append(bestF)
    mob = bestX.reshape(n_mobile, 2)
    rej = (n_rej / n_prop) if n_prop else 0.0
    return Result(cr=bestF, mob=mob, oracle=orc.calls, ct=time.perf_counter() - t0,
                  disp=displacement(mob, init_best), hist=np.array(hist),
                  reject_rate=rej,
                  stuck_at_init=bool(np.allclose(mob, init_best)))
