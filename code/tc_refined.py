"""
Critical horizon T_c(r,q,sigma) for geometric Brownian motion, with a convergence study.

The implicit-Euler / Brennan-Schwartz solver of gbm_solver.py is re-implemented with numba for
speed.  Three estimators of T_c = sup{s : B(s) >= 1} are compared on three grid levels:

  (a) "node": the first s at which the node x = 1 leaves the discrete stopping set, i.e. the
      first s with u(s,1) := V(s,1) - 0 > 0, bracketed from below by the first s at which the
      node to the right of 1 leaves it (this bracket is what gbm_solver.T_c used);
  (b) "boundary": the sub-grid boundary B~(s) obtained by fitting u ~ a (x - B)^2 to the first
      two continuation nodes (smooth fit: u vanishes quadratically at the boundary), interpolated
      across 1;
  (c) "sqrt": since u(s,1) ~ a (1 - B(s))^2 ~ a B'(T_c)^2 (s - T_c)^2 for s just above T_c,
      sqrt(u(s,1)) is asymptotically linear in s; a least-squares line through sqrt(u(s_i,1)) on
      the first part of the continuation range is extrapolated to zero.

Also stored: B~(s) on [0, s_max] for the figure and the implied constant of the parabolic law.

The default command reproduces the full study and figure data.  --quick (or the legacy
positional argument quick) runs only q=0.049, sigma=0.30 on grids (ny, ns) = (801, 500)
and (1601, 1000), skips figure solves, and writes tc_refined_quick.json.  Quick results
are for smoke checks, not the paper.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

C = 0.63883322


@njit(cache=True)
def bs_solve(lower, diag, upper, rhs, obstacle):
    n = diag.shape[0]
    d = diag.copy(); b = rhs.copy()
    for i in range(n - 2, -1, -1):
        w = upper[i] / d[i + 1]
        d[i] -= w * lower[i + 1]
        b[i] -= w * b[i + 1]
    v = np.empty(n)
    v[0] = max(b[0] / d[0], obstacle[0])
    for i in range(1, n):
        v[i] = max((b[i] - lower[i] * v[i - 1]) / d[i], obstacle[i])
    return v


def grid(sigma, s_max, ny):
    span = 6.0 * sigma * np.sqrt(max(s_max, 1e-6)) + 2.5
    return np.linspace(-span, span, ny)


def sqrt_extrap(x0, u0, x1, u1):
    s0, s1 = np.sqrt(u0), np.sqrt(u1)
    return (x1 * s0 - x0 * s1) / (s0 - s1)


def solve(r, q, sigma, s_max, ny, ns):
    y = grid(sigma, s_max, ny); dy = y[1] - y[0]; x = np.exp(y); g = 1.0 - x
    i1 = int(np.argmin(np.abs(y)))
    assert abs(y[i1]) < 1e-12
    ds = s_max / ns; mu = r - q - 0.5 * sigma ** 2
    a = 0.5 * sigma ** 2 / dy ** 2; bc = mu / (2 * dy)
    lower = -ds * (a - bc) * np.ones(ny); diag = (1 + ds * (2 * a + r)) * np.ones(ny); upper = -ds * (a + bc) * np.ones(ny)
    diag[0] = 1; upper[0] = 0; diag[-1] = 1; lower[-1] = 0
    V = g.copy()
    s_grid = np.linspace(0, s_max, ns + 1)
    Bn = np.empty(ns + 1); Bn[0] = r / q
    Bnext = np.empty(ns + 1); Bnext[0] = np.nan
    Br = np.empty(ns + 1); Br[0] = r / q
    u1 = np.zeros(ns + 1)
    xstar = np.empty(ns + 1); xstar[0] = 1.0
    for n in range(1, ns + 1):
        s = s_grid[n]
        rhs = V.copy(); rhs[0] = g[0]; rhs[-1] = np.exp(-r * s) - x[-1] * np.exp(-q * s)
        V = bs_solve(lower, diag, upper, rhs, g)
        u = V - g
        idx = np.where(u <= 0.0)[0]
        j = idx.max()                       # last node of the stopping set
        Bn[n] = x[j]
        Bnext[n] = x[j + 1]
        # sub-grid boundary from the first two continuation nodes
        if j + 2 < ny and u[j + 1] > 0 and u[j + 2] > u[j + 1]:
            Br[n] = sqrt_extrap(x[j + 1], u[j + 1], x[j + 2], u[j + 2])
        else:
            Br[n] = np.nan
        u1[n] = u[i1]
        k = np.where(V <= 0)[0]
        if len(k) and k.min() > 0:
            k0 = k.min(); x0, x1 = x[k0 - 1], x[k0]; v0, v1 = V[k0 - 1], V[k0]
            xstar[n] = x0 + (x1 - x0) * v0 / (v0 - v1)
        else:
            xstar[n] = np.nan
    return dict(s=s_grid, Bn=Bn, Bnext=Bnext, Br=Br, u1=u1, xstar=xstar, cstar=1.0 / xstar, x=x, V=V)


def estimators(sol):
    s, Bn, Br, u1 = sol['s'], sol['Bn'], sol['Br'], sol['u1']
    out = {}
    # (a) node brackets
    lt = np.where(u1 > 0)[0]                          # node x=1 has left the stopping set
    le = np.where(Bn <= 1.0)[0]                       # last stopping node is <= 1 (node right of 1 left)
    out['tc_node_lower'] = s[le.min()] if len(le) else np.inf     # first s with B_node <= 1
    out['tc_node_upper'] = s[lt.min()] if len(lt) else np.inf     # first s with B_node < 1
    # (b) refined boundary across 1
    ok = np.isfinite(Br)
    below = np.where(ok & (Br < 1.0))[0]
    if len(below):
        n = below.min()
        # previous finite point
        prev = np.where(ok[:n])[0]
        if len(prev):
            p = prev.max(); s0, s1 = s[p], s[n]; b0, b1 = Br[p], Br[n]
            out['tc_boundary'] = s0 + (s1 - s0) * (b0 - 1) / (b0 - b1)
        else:
            out['tc_boundary'] = s[n]
    else:
        out['tc_boundary'] = np.inf
    # (c) sqrt extrapolation of u(s,1)
    if len(lt):
        n0 = lt.min()
        # use the window of s where 1 - B~(s) is at most ~ 1.5% (quadratic regime), at least 8 points, at most 40
        cand = np.arange(n0, min(len(s), n0 + 400))
        cand = cand[u1[cand] > 0]
        if len(cand) >= 4:
            root = np.sqrt(u1[cand])
            # restrict to the part where sqrt(u) is small relative to its end value: take up to the first 25 points
            m = min(len(cand), 25)
            sel = cand[:m]
            A = np.vstack([s[sel], np.ones(m)]).T
            coef, *_ = np.linalg.lstsq(A, np.sqrt(u1[sel]), rcond=None)
            out['tc_sqrt'] = -coef[1] / coef[0]
        else:
            out['tc_sqrt'] = np.nan
    else:
        out['tc_sqrt'] = np.inf
    return out


def Tc_study(r, q, sigma, levels=((1601, 1000), (3201, 2000), (6401, 4000)), verbose=True):
    guess = ((r / q - 1) / (C * sigma)) ** 2
    res = []
    for (ny, ns) in levels:
        s_max = max(2.0 * guess, 0.02)
        for _ in range(8):
            sol = solve(r, q, sigma, s_max, ny, ns)
            est = estimators(sol)
            if np.isfinite(est['tc_node_upper']):
                tc0 = est['tc_boundary'] if np.isfinite(est['tc_boundary']) else est['tc_node_upper']
                sol = solve(r, q, sigma, 1.6 * tc0, ny, ns)        # finer time resolution around T_c
                est = estimators(sol)
                break
            s_max *= 4
        est.update(ny=ny, ns=ns, s_max=float(sol['s'][-1]))
        res.append(est)
        if verbose:
            print(f"  ny={ny:5d} ns={ns:5d}: node bracket [{est['tc_node_lower']:.6f}, {est['tc_node_upper']:.6f}]  boundary {est['tc_boundary']:.6f}  sqrt {est['tc_sqrt']:.6f}", flush=True)
    return res, sol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['quick'], help='legacy alias for --quick')
    parser.add_argument('--quick', action='store_true', help='two coarse grids for one parameter pair; skip figure data')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'generated',
                        help='output directory (default: code/generated beside this script)')
    args = parser.parse_args(argv)
    quick = args.quick or args.mode == 'quick'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r = 0.05
    qs = [0.049, 0.048, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02]
    sigmas = [0.15, 0.30, 0.50]
    levels = ((1601, 1000), (3201, 2000), (6401, 4000))
    if quick:
        qs = [0.049]; sigmas = [0.30]
        levels = ((801, 500), (1601, 1000))
    output_path = args.output_dir / ('tc_refined_quick.json' if quick else 'tc_refined.json')
    rows = []
    t0 = time.time()
    for sigma in sigmas:
        for q in qs:
            print(f"r={r} q={q} sigma={sigma}  asymptotic {((r/q-1)/(C*sigma))**2:.5f}", flush=True)
            res, sol = Tc_study(r, q, sigma, levels=levels)
            # Richardson-type summary from the two finest levels of the boundary estimator (first order in h)
            tb = [e['tc_boundary'] for e in res]
            rich = 2 * tb[-1] - tb[-2] if len(tb) >= 2 else tb[-1]
            rows.append(dict(r=r, q=q, sigma=sigma, levels=res, asym=((r / q - 1) / (C * sigma)) ** 2, tc_rich=rich))
            print(f"    -> boundary estimator by level: {tb}; Richardson(1st order) {rich:.6f}   [{time.time()-t0:.0f}s]", flush=True)
            with output_path.open('w') as output:
                json.dump(rows, output, indent=1)
    if quick:
        return
    # boundaries for the figure (sigma = 0.30, s in [0, 1]) with the sub-grid refinement
    fig = {}
    for q in [0.045, 0.04, 0.03, 0.02]:
        sol = solve(r, q, 0.30, 1.0, 6401, 4000)
        fig[str(q)] = dict(s=sol['s'].tolist(), Bn=sol['Bn'].tolist(), Br=sol['Br'].tolist(), cstar=sol['cstar'].tolist())
        s = sol['s']; Br = sol['Br']
        for ss in [0.0025, 0.01, 0.04, 0.09]:
            n = int(round(ss / s[1]))
            print(f"q={q} s={ss}: B~={Br[n]:.5f}  implied C = {(1 - Br[n]*q/r)/(0.30*np.sqrt(ss)):.4f}")
    with (args.output_dir / 'boundaries_refined.json').open('w') as output:
        json.dump(fig, output)


if __name__ == "__main__":
    main()
