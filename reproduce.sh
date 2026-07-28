#!/usr/bin/env bash
# Regenerates every number and figure in the paper from scratch.
# Total runtime: ~25 min on a single 2.1 GHz core.
set -e
python3 run_experiments.py results kmax
python3 run_experiments.py results "regA:det,VFA,PSO,GA,ES"
python3 run_experiments.py results "regA:GA_LR,GA_LR_Rnd"
python3 run_experiments.py results "regA:GA_LR_S,GA_LR_S_inv,CGWO_unc,CGWO_cor"
python3 run_experiments.py results "regB:det,VFA,PSO,GA,ES"
python3 run_experiments.py results "regB:GA_LR,GA_LR_Rnd"
python3 run_experiments.py results "regB:GA_LR_S,GA_LR_S_inv,CGWO_unc,CGWO_cor"
python3 run_experiments.py results "regC:det,VFA,PSO,GA,ES"
python3 run_experiments.py results "regC:GA_LR,GA_LR_Rnd"
python3 run_experiments.py results "regC:GA_LR_S,GA_LR_S_inv,CGWO_unc,CGWO_cor"
python3 run_experiments.py results sweep
python3 run_experiments.py results final
python3 make_figures.py
echo "All results in results/paper_numbers.json, figures in figures/"
