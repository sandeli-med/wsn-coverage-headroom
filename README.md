# wsn_paper — reproduction package

Reproduces every number, table and figure of
"Coverage headroom predicts when local refinement pays" (Sandeli & Kenidra).

* `core.py`            — problem instance, oracle with call counting, and all 12
                         methods (Lattice, Greedy, VFA, PSO, GA, ES, GA-LR,
                         GA-LR-Rnd, GA-LR-S, GA-LR-S(inv), CGWO uncorrected /
                         corrected), exactly as specified in the paper.
* `run_experiments.py` — staged driver: 3 regimes x 30 replicates, 16-cell
                         sweep (15 reps), Wilcoxon/Holm/A12/bootstrap stats,
                         equal-oracle-budget runs, K sensitivity, discretisation
                         study, CGWO validation on Shaikh et al. Case 1.
                         Writes `results/paper_numbers.json`.
* `make_figures.py`    — regenerates the 6 figures into `figures/`.
* `reproduce.sh`       — one command for everything (~25 min, single core).

Seeds: seed = base + 101*rep + 7*method_index, bases 4000/2000/6000 for
regimes A/B/C; the shared initial population of a replicate uses
base + 101*rep. Re-running on the same NumPy version yields identical numbers
(verified: a clean-room re-run reproduced all 1412 reported values exactly).

Metric definitions worth noting:
  * `oracle` counts full-layout evaluations of the coverage objective. Greedy
    additionally performs |C|*Nm marginal-gain lookups, reported separately as
    `gain_evals`; speed claims in the paper rest on wall-clock time, not on
    either count.
  * `disp` is the minimum total travel to realise a layout from the initial
    deployment, computed as an optimal assignment (sensors are interchangeable).
  * Statistical comparisons use the paired Wilcoxon signed-rank test and a
    paired bootstrap, matching the shared-initial-population design.
  * phi_inv features are derived from the uncovered-target set and therefore
    consume the oracle; those calls are deliberately uncounted, which makes
    GA-LR-S(inv) an optimistic upper bound rather than a practical method.

Environment used for the paper: Intel Xeon @ 2.10 GHz (1 core), 4 GB RAM,
Python 3.12, NumPy 2.4.4, SciPy 1.17.1, replicates sequential.
