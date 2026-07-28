"""make_figures.py — regenerates every figure of the paper from paper_numbers.json.

Output conforms to Applied Intelligence artwork guidelines:
  * vector PDF with fonts embedded (Type 42)
  * sized to the journal text width (31pc = 131 mm) so figures are placed 1:1
    and lettering renders at 7-8 pt (the required 2-3 mm)
  * files named Fig1 ... Fig6 in order of citation
  * figure parts denoted by lowercase letters (a), (b), (c)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/paper_numbers.json"))
os.makedirs("figures", exist_ok=True)
REG = ("A", "B", "C")
UB = {"A": 20.00, "B": 56.25, "C": 100.00}

TW = 5.1667          # journal text width in inches (31pc = 131 mm)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,       # embed TrueType, avoid Type-3
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

METHODS = ["Lattice", "Greedy", "VFA", "PSO", "GA", "ES", "GA_LR",
           "GA_LR_Rnd", "GA_LR_S", "CGWO_unc", "CGWO_cor"]
LBL = {"GA_LR": "GA-LR", "GA_LR_Rnd": "GA-LR-Rnd", "GA_LR_S": "GA-LR-S",
       "GA_LR_S_inv": "GA-LR-S (inv)", "CGWO_unc": "CGWO uncorr.",
       "CGWO_cor": "CGWO corr."}
def lbl(m): return LBL.get(m, m)

BLUE = "#4878a8"


# --------------------------------------------------- Fig1: coverage ceiling ---
# Regimes stacked as rows with horizontal bars so that method names are set
# horizontally and stay legible at 7 pt.
fig, axes = plt.subplots(3, 1, figsize=(TW, 5.15))
for k, (ax, R) in enumerate(zip(axes, REG)):
    vals = [d["regimes"][R][m]["cr"][0] for m in METHODS]
    errs = [d["regimes"][R][m]["cr"][1] for m in METHODS]
    y = np.arange(len(METHODS))
    ax.barh(y, vals, xerr=errs, capsize=2, color=BLUE, edgecolor="k",
            linewidth=0.4, height=0.68)
    ax.axvline(UB[R], ls="--", c="k", lw=0.9)
    ax.text(UB[R], -0.6, f"  $CR^{{ub}}$ = {UB[R]:.2f}", va="center",
            ha="left", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl(m) for m in METHODS])
    ax.set_ylim(len(METHODS) - 0.4, -1.1)
    ax.set_xlim(0, max(UB[R] * 1.14, max(vals) * 1.2))
    ax.set_title(f"({chr(97 + k)}) Regime {R}", loc="left")
    ax.set_xlabel("final coverage CR (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/Fig1.pdf"); plt.close(fig)


# ------------------------------------------------------- Fig2: convergence ---
fig, axes = plt.subplots(1, 3, figsize=(TW, 2.15))
show = ["GA", "ES", "PSO", "GA_LR", "GA_LR_S", "CGWO_cor", "CGWO_unc"]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for k, (ax, R) in enumerate(zip(axes, REG)):
    cur = d["regimes"][R]["_curves"]
    for i, m in enumerate(show):
        mu = np.array(cur[m]["mean"]); sd = np.array(cur[m]["std"])
        g = np.arange(len(mu))
        ax.plot(g, mu, label=lbl(m), color=colors[i], lw=0.9)
        ax.fill_between(g, mu - sd, mu + sd, color=colors[i], alpha=0.15, lw=0)
    ax.axhline(UB[R], ls="--", c="k", lw=0.8,
               label="$CR^{ub}$" if k == 0 else None)
    ax.axhline(d["regimes"][R]["Greedy"]["cr"][0], ls="-.", c="gray", lw=0.8,
               label="Greedy" if k == 0 else None)
    ax.set_title(f"({chr(97 + k)}) Regime {R}", loc="left")
    ax.set_xlabel("generation")
    ax.tick_params(width=0.5)
    if k == 0:
        ax.set_ylabel("best-so-far CR (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, -0.13), frameon=False)
fig.tight_layout()
fig.savefig("figures/Fig2.pdf"); plt.close(fig)


# ---------------------------------------------------------- Fig3: headroom ---
fig, ax = plt.subplots(figsize=(TW * 0.86, 3.1))
h = np.array([c["ub"] - c["ga"][0] for c in d["sweep"]])
gain = np.array([c["galr"][0] - c["ga"][0] for c in d["sweep"]])
nm = np.array([c["Nm"] for c in d["sweep"]])
sc = ax.scatter(h, gain, c=nm, cmap="viridis", s=30, edgecolor="k", lw=0.4)
xs = np.linspace(h.min(), h.max(), 50)
sl, ic = d["fit"]["slope"], d["fit"]["intercept"]
ax.plot(xs, sl * xs + ic, "r-", lw=1.0, label=f"gain = {sl:.3f}h {ic:+.3f}")
cb = plt.colorbar(sc, ax=ax); cb.set_label("$N_m$"); cb.ax.tick_params(labelsize=7)
ax.set_xlabel("headroom $h$ (pp)")
ax.set_ylabel("refinement gain (pp)")
ax.legend(); ax.grid(lw=0.3, alpha=0.4); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/Fig3.pdf"); plt.close(fig)


# -------------------------------------------------- Fig4: rank correlation ---
fig, ax = plt.subplots(figsize=(TW, 2.9))
for R, c in zip(REG, ("#1f77b4", "#ff7f0e", "#2ca02c")):
    raw = np.array(d["regimes"][R]["GA_LR_S"]["rho_by_gen"])
    inv = np.array(d["regimes"][R]["GA_LR_S_inv"]["rho_by_gen"])
    ax.plot(np.arange(1, len(raw) + 1), raw, c=c, lw=0.9,
            label=f"{R}: $\\phi_{{raw}}$")
    ax.plot(np.arange(1, len(inv) + 1), inv, c=c, lw=0.9, ls="--",
            label=f"{R}: $\\phi_{{inv}}$")
ax.set_ylim(0, 1.02)
ax.set_xlabel("generation")
ax.set_ylabel("Spearman $\\rho$")
ax.legend(fontsize=6.5, ncol=3, loc="lower right")
ax.grid(lw=0.3, alpha=0.4); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/Fig4.pdf"); plt.close(fig)


# ---------------------------------------------------------- Fig5: boxplots ---
bm = ["GA", "ES", "PSO", "GA_LR", "GA_LR_Rnd", "GA_LR_S", "GA_LR_S_inv",
      "CGWO_unc", "CGWO_cor"]
fig, axes = plt.subplots(3, 1, figsize=(TW, 4.9))
for k, (ax, R) in enumerate(zip(axes, REG)):
    data = [d["regimes"][R][m]["cr_all"] for m in bm]
    ax.boxplot(data, tick_labels=[lbl(m) for m in bm], vert=False, widths=0.6,
               flierprops={"markersize": 2})
    ax.axvline(UB[R], ls="--", c="k", lw=0.9)
    ax.axvline(d["regimes"][R]["Greedy"]["cr"][0], ls="-.", c="gray", lw=0.9)
    ax.invert_yaxis()
    ax.set_title(f"({chr(97 + k)}) Regime {R}", loc="left")
    ax.set_xlabel("final coverage CR (%)")
    ax.grid(axis="x", lw=0.3, alpha=0.4); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/Fig5.pdf"); plt.close(fig)


# -------------------------------------------------------- Fig6: deployments ---
from core import TARGETS, STATICS, Oracle
fig, axes = plt.subplots(1, 2, figsize=(TW, 2.85))
r = 20.0
galr_mob = np.array(d["regimes"]["C"]["_galr_mobs"][0])
greedy_mob = np.array(d["regimes"]["C"]["_greedy_mob"])
_o = Oracle(TARGETS, STATICS, r)
panels = ((axes[0], galr_mob, f"(a) GA-LR, CR = {_o.f(galr_mob, count=False):.2f}%"),
          (axes[1], greedy_mob, f"(b) Greedy, CR = {_o.f(greedy_mob, count=False):.2f}%"))
for ax, mob, title in panels:
    orc = Oracle(TARGETS, STATICS, r)
    cov = orc.static_mask | orc.mobile_mask(mob)
    unc = TARGETS[~cov]
    for s in STATICS:
        ax.add_patch(plt.Circle(s, r, fill=False, ls="--", ec="#888", lw=0.5))
    for m_ in mob:
        ax.add_patch(plt.Circle(m_, r, fill=False, ec=BLUE, lw=0.6))
    ax.scatter(*STATICS.T, marker="s", c="k", s=9, zorder=3, label="static")
    ax.scatter(*mob.T, marker="^", c=BLUE, s=11, zorder=3, label="mobile")
    if len(unc):
        ax.scatter(*unc.T, marker=".", c="red", s=3, zorder=2, label="uncovered")
    ax.set_xlim(-24, 224); ax.set_ylim(-24, 224); ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.tick_params(width=0.5)
axes[0].legend(fontsize=6, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig("figures/Fig6.pdf"); plt.close(fig)

print("figures written:", sorted(os.listdir("figures")))
