"""make_figures.py — regenerates every figure of the paper from paper_numbers.json"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/paper_numbers.json"))
os.makedirs("figures", exist_ok=True)
REG = ("A", "B", "C")
UB = {"A": 20.00, "B": 56.25, "C": 100.00}
plt.rcParams.update({"font.size": 9, "figure.dpi": 300})

METHODS = ["Lattice", "Greedy", "VFA", "PSO", "GA", "ES", "GA_LR",
           "GA_LR_Rnd", "GA_LR_S", "CGWO_unc", "CGWO_cor"]
LBL = {"GA_LR": "GA--LR", "GA_LR_Rnd": "GA--LR--Rnd", "GA_LR_S": "GA--LR--S",
       "GA_LR_S_inv": "GA--LR--S (inv)", "CGWO_unc": "CGWO unc.",
       "CGWO_cor": "CGWO corr."}
def lbl(m): return LBL.get(m, m)

# ------------------------------------------------------------ fig_ceiling ---
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), sharey=False)
for ax, R in zip(axes, REG):
    vals = [d["regimes"][R][m]["cr"][0] for m in METHODS]
    errs = [d["regimes"][R][m]["cr"][1] for m in METHODS]
    x = np.arange(len(METHODS))
    ax.bar(x, vals, yerr=errs, capsize=2, color="#4878a8", edgecolor="k",
           linewidth=0.4)
    ax.axhline(UB[R], ls="--", c="k", lw=1)
    ax.text(0.02, UB[R], f"$CR^{{ub}}$ = {UB[R]:.2f}", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([lbl(m) for m in METHODS],
                                         rotation=70, fontsize=6.5)
    ax.set_title(f"Regime {R}")
    if R == "A": ax.set_ylabel("final CR (%)")
fig.tight_layout(); fig.savefig("figures/fig_ceiling.png"); plt.close(fig)

# --------------------------------------------------------- fig_convergence ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
show = ["GA", "ES", "PSO", "GA_LR", "GA_LR_S", "CGWO_cor", "CGWO_unc"]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for ax, R in zip(axes, REG):
    cur = d["regimes"][R]["_curves"]
    for i, m in enumerate(show):
        mu = np.array(cur[m]["mean"]); sd = np.array(cur[m]["std"])
        g = np.arange(len(mu))
        ax.plot(g, mu, label=lbl(m), color=colors[i], lw=1.2)
        ax.fill_between(g, mu - sd, mu + sd, color=colors[i], alpha=0.15, lw=0)
    ax.axhline(UB[R], ls="--", c="k", lw=1, label="$CR^{ub}$" if R == "A" else None)
    ax.axhline(d["regimes"][R]["Greedy"]["cr"][0], ls="-.", c="gray", lw=1,
               label="Greedy" if R == "A" else None)
    ax.set_title(f"Regime {R}"); ax.set_xlabel("generation")
    if R == "A": ax.set_ylabel("best-so-far CR (%)")
axes[0].legend(fontsize=6, ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig_convergence.png"); plt.close(fig)

# ------------------------------------------------------------ fig_headroom ---
fig, ax = plt.subplots(figsize=(4.2, 3.2))
h = np.array([c["ub"] - c["ga"][0] for c in d["sweep"]])
gain = np.array([c["galr"][0] - c["ga"][0] for c in d["sweep"]])
nm = np.array([c["Nm"] for c in d["sweep"]])
sc = ax.scatter(h, gain, c=nm, cmap="viridis", s=35, edgecolor="k", lw=0.4)
xs = np.linspace(h.min(), h.max(), 50)
sl, ic = d["fit"]["slope"], d["fit"]["intercept"]
ax.plot(xs, sl * xs + ic, "r-", lw=1.2,
        label=f"gain = {sl:.3f}h {ic:+.3f}")
cb = plt.colorbar(sc, ax=ax); cb.set_label("$N_m$")
ax.set_xlabel("headroom $h$ (pp)"); ax.set_ylabel("refinement gain (pp)")
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig("figures/fig_headroom.png"); plt.close(fig)

# ----------------------------------------------------------------- fig_rho ---
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for R, c in zip(REG, ("#1f77b4", "#ff7f0e", "#2ca02c")):
    raw = np.array(d["regimes"][R]["GA_LR_S"]["rho_by_gen"])
    inv = np.array(d["regimes"][R]["GA_LR_S_inv"]["rho_by_gen"])
    ax.plot(np.arange(1, len(raw) + 1), raw, c=c, lw=1.1,
            label=f"{R}: $\\phi_{{raw}}$")
    ax.plot(np.arange(1, len(inv) + 1), inv, c=c, lw=1.1, ls="--",
            label=f"{R}: $\\phi_{{inv}}$")
ax.set_ylim(0, 1.02); ax.set_xlabel("generation")
ax.set_ylabel("Spearman $\\rho$ (pred vs true, held-out offspring)")
ax.legend(fontsize=6.5, ncol=2); fig.tight_layout()
fig.savefig("figures/fig_rho.png"); plt.close(fig)

# ----------------------------------------------------------------- fig_box ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
bm = ["GA", "ES", "PSO", "GA_LR", "GA_LR_Rnd", "GA_LR_S", "GA_LR_S_inv",
      "CGWO_unc", "CGWO_cor"]
for ax, R in zip(axes, REG):
    data = [d["regimes"][R][m]["cr_all"] for m in bm]
    ax.boxplot(data, tick_labels=[lbl(m) for m in bm], widths=0.6,
               flierprops={"markersize": 2})
    ax.axhline(UB[R], ls="--", c="k", lw=1)
    ax.axhline(d["regimes"][R]["Greedy"]["cr"][0], ls="-.", c="gray", lw=1)
    ax.set_title(f"Regime {R}")
    ax.tick_params(axis="x", rotation=70, labelsize=6.5)
    if R == "A": ax.set_ylabel("final CR (%)")
fig.tight_layout(); fig.savefig("figures/fig_box.png"); plt.close(fig)

# -------------------------------------------------------------- fig_layout ---
from core import TARGETS, STATICS, Oracle
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
r = 20.0
galr_mob = np.array(d["regimes"]["C"]["_galr_mobs"][0])
greedy_mob = np.array(d["regimes"]["C"]["_greedy_mob"])
_o = Oracle(TARGETS, STATICS, r)
for ax, mob, title in ((axes[0], galr_mob,
                        f"GA--LR (CR = {_o.f(galr_mob, count=False):.2f}%)"),
                       (axes[1], greedy_mob,
                        f"Greedy (CR = {_o.f(greedy_mob, count=False):.2f}%)")):
    orc = Oracle(TARGETS, STATICS, r)
    cov = orc.static_mask | orc.mobile_mask(mob)
    unc = TARGETS[~cov]
    for s in STATICS:
        ax.add_patch(plt.Circle(s, r, fill=False, ls="--", ec="#888", lw=0.7))
    for m_ in mob:
        ax.add_patch(plt.Circle(m_, r, fill=False, ec="#4878a8", lw=0.8))
    ax.scatter(*STATICS.T, marker="s", c="k", s=22, zorder=3, label="static")
    ax.scatter(*mob.T, marker="^", c="#4878a8", s=26, zorder=3, label="mobile")
    if len(unc):
        ax.scatter(*unc.T, marker=".", c="red", s=8, zorder=2, label="uncovered")
    ax.set_xlim(-22, 222); ax.set_ylim(-22, 222); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig_layout.png"); plt.close(fig)

print("figures written:", os.listdir("figures"))
