"""Additional GBM prices and carry comparisons quoted in the paper.

The default run retains the original grids and writes code/generated/extra_rows.json.
It performs six one-week and four one-year PDE solves.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from gbm_solver import solve_implicit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'generated',
                        help='output directory (default: code/generated beside this script)')
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    r, sigma = 0.05, 0.30
    T = 1/52
    rows = []
    for q in [0.05, 0.055, 0.06, 0.08, 0.10, 0.15]:
        sol = solve_implicit(r, q, sigma, T, ny=4001, ns=600)
        c = sol['cstar'][-1]
        rows.append(dict(q=q, cstar=c, carry=np.exp((r-q)*T), B=sol['B'][-1]))
        print(f"q={q:.3f}: c*={c:.6f}  carry bound e^((r-q)T)={np.exp((r-q)*T):.6f}  stochastic part (bp)={(np.exp((r-q)*T)-c)*1e4:6.2f}  total EOM discount (bp)={(1-c)*1e4:6.2f}   B(T)={sol['B'][-1]:.4f}")
    # one-year window, q near r
    print("--- one-year window ---")
    for q in [0.045, 0.05, 0.055, 0.06]:
        sol = solve_implicit(r, q, sigma, 1.0, ny=4001, ns=2000)
        c = sol['cstar'][-1]
        rows.append(dict(q=q, T=1.0, cstar=c, carry=np.exp((r-q)*1.0), B=sol['B'][-1]))
        print(f"q={q:.3f}: c*={c:.6f}  carry bound={np.exp((r-q)*1.0):.6f}  stochastic part (bp)={(np.exp((r-q)*1.0)-c)*1e4:6.2f}  total (bp)={(1-c)*1e4:7.2f}  B(T)={sol['B'][-1]:.4f}")
    with (args.output_dir / 'extra_rows.json').open('w') as output:
        json.dump(rows, output, indent=1)


if __name__ == "__main__":
    main()
