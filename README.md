# End-of-month futures options and the Gittins index

Numerical code accompanying **The end-of-month option in futures contracts and the Gittins index of holding the underlying**, by **Erhan Bayraktar**, University of Michigan.

The code studies the fair delivery price and optimal delivery rule for an end-of-month futures option. It includes geometric Brownian motion with constant convenience yield and the Gibson-Schwartz model with stochastic convenience yield. The saved results reproduce the three numerical figures in the October 2026 manuscript.

## Install

Use **Python 3.12**. From a terminal:

```sh
git clone https://github.com/ebayr/eom-gittins.git
cd eom-gittins
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

The four direct dependencies are pinned to the versions used for release verification. Numba accelerates the refined GBM solver. No proprietary software or market-data subscription is required.

## Reproduce the figures and tables from saved results

Run these commands from the repository root:

```sh
python code/make_figures.py
python code/make_tables.py
python code/smoke_check.py
```

The first command writes PDF and PNG plots to `code/figures/`. Curves have distinct line styles and marker shapes so they remain distinguishable without color. The second writes Markdown and CSV tables to `code/tables/`, directly from the saved numerical results. The third performs small numerical and data-consistency checks; it does not rerun the full manuscript grids.

| Manuscript result | Figure or table | Saved data | Solver |
| --- | --- | --- | --- |
| GBM delivery boundary | Figure 4.1, `fig_boundary.pdf` | `boundaries_refined.json`, `tc_refined.json` | `tc_refined.py` |
| GBM critical delivery-window length | Figure 4.2, `fig_phase.pdf`; Table 1 | `tc_refined.json` | `tc_refined.py` |
| Stochastic-yield price discount and holding index | Figure 4.3, `fig_gs.pdf`; Table 2 | `gs_monotone_results.json` | `gs_monotone.py` |

All saved datasets are in `code/results/`. The nonnumerical theorem-dependency diagram is part of the manuscript's LaTeX source and is not generated here.

The GS figure plots the normalized holding index `k = K/X`. Delivery at the fixed contract price `m*` occurs when `K <= m*`, equivalently `k <= m*/X`. At inception, the line `k = 1` separates immediate delivery at `m* = X0` from initial continuation with `m* < X0`.

## Recompute the numerical data

These commands run the numerical schemes, rather than just plotting saved results:

```sh
# Small GBM critical-horizon example
python code/tc_refined.py --quick

# Full GBM parameter study and delivery boundaries
python code/tc_refined.py

# Both GS windows at all four manuscript grid spacings
python code/gs_monotone.py --spacings 0.01 0.005 0.0025 0.00125

# Plot the newly computed results
python code/make_figures.py --results-dir code/generated --output-dir code/generated/figures
python code/make_tables.py --results-dir code/generated --output-dir code/generated/tables
```

Solver output goes to `code/generated/` by default, keeping the supplied reference results intact. Use `--help` for output-directory and computation options. The quick GBM example is a coarse check and is not sufficient to regenerate the manuscript figures.

Full computations are substantially more expensive than the saved-data commands. The refined GBM study covers 24 parameter combinations and three grid levels, followed by four boundary computations. The finest GS grid has 4,001 by 1,441 nodes and uses approximately 2,468 time steps for the one-week window and 10,691 for the one-month window. Allow several GB of available memory; runtime depends on the machine. The saved reference datasets are supplied so that viewing the results does not require these runs.

## Additional numerical checks

```sh
python code/gbm_solver.py
python code/universal_constant.py
python code/newton_check.py
python code/verify_fixed_point.py
python code/extra_numbers.py
```

These scripts respectively compare GBM solvers, calculate the Brownian constant, illustrate the price iteration, check the delivery rule and price identity by Monte Carlo, and report further GBM prices. The full Brownian finite-difference check and Monte Carlo check can be expensive. Monte Carlo output includes sampling errors and uses a fixed random seed. Consult `--help` on the ancillary scripts for their available options.

For a fast calculation of the Brownian constant without the finite-difference study, run `python code/universal_constant.py --ode-only`. It gives `C = 0.63883322`.

Release checks on 14 October 2026 used a fresh Python 3.12 environment: all saved-data checks passed; both coarse GS windows matched their saved results to within `9e-12`; the GBM/tree comparison had maximum absolute error `4.71e-6`; and figure/table generation, the quick critical-horizon example, and the constant calculation passed. The full publication grids and large Monte Carlo study were not rerun for this release.

## Numerical conventions and limitations

- Time is measured in years; a week is `1/52` and a month is `1/12`. Rates and volatilities are decimals. Price discounts are reported in basis points of initial spot.
- GBM boundaries and critical horizons use a quadratic fit to the first two continuation nodes. The exported critical-horizon table uses the finest saved boundary estimate.
- GS discounts use the finest grid, `h = 0.00125`. The table's index estimates use first-order Richardson extrapolation, `2*k_fine - k_coarse`, from the two finest grids. The figure uses the finest-grid index directly.
- The GS log-coordinate domain is `[-3, 2] x [-0.9, 0.9]`; the price-ratio range is approximately `[0.049787, 7.389056]`. The interior stencil checks positivity of its weights. Boundary values use linear extrapolation, which is not a positivity-preserving closure; truncation and boundary effects should be considered when extending the computations.
- The canonical GS index is signed. The `log(y)` grid covers only positive `y=m/x`, so it resolves only the positive portion of the stopping boundary; `NaN` or omitted estimates indicate an unresolved boundary, not an index of zero.
- Unresolved indices near the computational boundary are omitted from the figure. The saved Python-readable JSON contains `NaN` for unavailable boundary estimates; strict JSON readers must handle these values explicitly. Missing estimates are not zero.
- CSV exports retain numerical precision; Markdown exports round for readability. Small differences from older manuscript tables can arise from corrected rounding.

## Citation and support

Please cite the accompanying paper and identify the Git commit used for your calculations. GitHub's **Cite this repository** menu reads the included `CITATION.cff`. The manuscript is a draft; journal and DOI details will be added when available.

Erhan Bayraktar is partially supported by the National Science Foundation under grant DMS-2602036 and by the Susan M. Smith Chair.

Contact: [erhan@umich.edu](mailto:erhan@umich.edu).
