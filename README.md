# IEOR 6616 Programming Project: Computational Optimal Transport

**Author:** Ioannis Vourkas  
**UNI:** 2299  
**Affiliation:** Visiting PhD Student, Department of Electrical Engineering, CUNY City College

This repository contains a complete script-based solution to the computational optimal transport programming project. It covers:

- Part 1: exact OT as a linear program
- Part 1 optional extra credit: custom NumPy PDHG
- Part 2: quadratic regularization with ADMM and IPM
- Part 3: entropic regularization with Sinkhorn and a conic baseline
- Section 7: side-by-side coupling visualization
- Part 4: Gaussian OT with the closed-form Bures-Wasserstein benchmark

The implementation is organized as reusable Python modules in `ot_lab/` plus report-facing experiment scripts in `scripts/`.

## Repository Layout

- `ot_lab/problem_setup.py`: Gaussian sampling, OT instance generation, cost matrices
- `ot_lab/lp.py`: exact OT LP formulation and simplex/IPM/PDLP wrappers
- `ot_lab/pdhg.py`: optional custom PDHG solver
- `ot_lab/quadratic_ot.py`: quadratic OT, custom ADMM, and IPM baseline
- `ot_lab/entropic_ot.py`: Sinkhorn (POT), custom log-domain Sinkhorn, and conic baseline
- `ot_lab/visualization.py`: shared coupling-visualization helpers
- `ot_lab/gaussian_ot.py`: Gaussian closed-form Wasserstein-2 utilities
- `scripts/run_part1_lp.py`: required Part 1 experiments
- `scripts/run_part1_optional_pdhg.py`: optional PDHG experiments
- `scripts/run_part2_quadratic.py`: Part 2 experiments
- `scripts/run_part3_entropic.py`: Part 3 experiments
- `scripts/run_section7_visualization.py`: Section 7 visualization
- `scripts/run_part4_gaussian.py`: Part 4 experiments
- `results/`: generated tables and figures
- `report/report.tex`: Overleaf-ready report source

## Environment

The project was developed with Python `3.11`. A local virtual environment `.venv/` was used during development.

Install dependencies from the project root with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer not to use `.venv`, any Python environment with the packages in `requirements.txt` should work.

## How to Reproduce the Results

Run the scripts from the repository root.

### Part 1

```bash
.venv/bin/python scripts/run_part1_lp.py
.venv/bin/python scripts/run_part1_optional_pdhg.py
```

Outputs:

- `results/part1/part1a_timing_loglog.png`
- `results/part1/part1b_results_table.md`
- `results/part1/part1_small_case_verification.md`
- `results/part1_optional_pdhg/*`

### Part 2

```bash
.venv/bin/python scripts/run_part2_quadratic.py
```

Outputs:

- `results/part2/part2a_runtime_table.md`
- `results/part2/part2b_regularization_path.png`
- `results/part2/part2c_rho_sensitivity.png`

### Part 3

```bash
.venv/bin/python scripts/run_part3_entropic.py
```

Outputs:

- `results/part3/part3ab_comparison_table.md`
- `results/part3/part3c_convergence.png`
- `results/part3/part3d_regularization_path.png`

### Section 7 Visualization

```bash
.venv/bin/python scripts/run_section7_visualization.py
```

Outputs:

- `results/section7_visualization/section7_coupling_panels.png`
- `results/section7_visualization/section7_coupling_summary.md`

### Part 4

```bash
.venv/bin/python scripts/run_part4_gaussian.py
```

Outputs:

- `results/part4/part4a_closed_form.md`
- `results/part4/part4bc_summary.md`
- `results/part4/part4bc_convergence.png`
- `results/part4/part4d_geometry.png`

## Report

The report source is:

- `report/report.tex`

It is written to be copy-paste friendly for Overleaf. The figure paths are relative to the `report/` directory, so if you upload the report to Overleaf, also upload the corresponding images from `results/` while preserving their relative folder structure, or update the paths in the LaTeX file.

## Notes on Solver Choices

- Part 1 exact OT:
  - simplex: SciPy HiGHS dual simplex
  - IPM: SciPy HiGHS interior point
  - PDLP: OR-Tools PDLP
- Part 2 quadratic OT:
  - ADMM: custom implementation
  - IPM baseline: Clarabel via CVXPY
- Part 3 entropic OT:
  - Sinkhorn: POT
  - conic baseline: Clarabel via CVXPY
  - extra credit: custom log-domain Sinkhorn
- Part 4 discrete Gaussian experiments:
  - exact OT solver: SciPy HiGHS interior point

## Submission Checklist

- `report/report.tex` prepared
- `README.md` with reproduction instructions included
- All required figures and tables generated under `results/`
- Optional extra-credit implementations included:
  - custom NumPy PDHG
  - custom log-domain Sinkhorn

