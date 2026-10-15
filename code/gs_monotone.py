"""
Gibson-Schwartz end-of-month problem, monotone explicit scheme.

  dX = (r - delta) X dt + sigma X dW1,  d delta = kappa (alpha - delta) dt + sigma_d dW2,  corr rho.
  V(t,x,delta;m) = x v(t, y, delta),  y = m/x.  In z = log y the continuation PDE is
    v_t + a_zz v_zz + a_dd v_dd + a_zd v_zd + b_z v_z + b_d v_d - delta v = 0,
    a_zz = sigma^2/2, a_dd = sigma_d^2/2, a_zd = -rho sigma sigma_d (<0),
    b_z = -(r - delta) - sigma^2/2,  b_d = kappa (alpha - delta) + rho sigma sigma_d,
  obstacle v >= y - 1, terminal v(T, .) = y - 1.

Grid: EQUAL spacing h in z and in delta.  The mixed derivative is discretised with the
seven-point stencil adapted to the sign of a_zd (anti-diagonal neighbours for a_zd < 0), so
that all off-diagonal weights are nonnegative provided
    sigma >= rho sigma_d   and   sigma_d >= rho sigma           (true for the oil parameters),
and central first-order differences keep positivity provided h is small enough (checked).
Explicit Euler with dt below the positivity limit gives a monotone scheme; exact projection
on the obstacle after every step.

Outputs for the two windows (one week, one month) and four grid levels:
  * discount 1 - c*(delta0) in bp at the EXACT yields of Table 2 (they are grid nodes),
  * index k(0, delta0) = inf{y : (0,y,delta0) in S}, with sub-grid refinement,
  * critical yield delta_c = d(0,1) = sup{delta : (0,1,delta) in S}, with sub-grid refinement,
    and, independently, the root of k(0, .) = 1.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

PAR = dict(r=0.06, kappa=1.876, alpha=0.106, sigma=0.393, sigma_d=0.371, rho=0.766)
C = 0.63883322
YIELDS = [-0.20, 0.0, 0.02, 0.03, 0.04, 0.06, 0.10, 0.20]


def solve_gs(T, h, zlo=-3.0, zhi=2.0, dlo=-0.9, dhi=0.9, safety=0.9, p=PAR, verbose=True):
    r, kappa, alpha, sigma, sigma_d, rho = p['r'], p['kappa'], p['alpha'], p['sigma'], p['sigma_d'], p['rho']
    nz = int(round((zhi - zlo) / h)) + 1
    nd = int(round((dhi - dlo) / h)) + 1
    z = zlo + h * np.arange(nz); y = np.exp(z)
    d = dlo + h * np.arange(nd)
    assert abs(z[nz // 2 + 0] - 0.0) < 1e-12 or np.min(np.abs(z)) < 1e-12, "y=1 must be a node"
    Y, Dl = np.meshgrid(y, d, indexing='ij')
    G = Y - 1.0
    a_zz = 0.5 * sigma ** 2; a_dd = 0.5 * sigma_d ** 2; c = rho * sigma * sigma_d   # a_zd = -c < 0
    b_z = -(r - Dl) - 0.5 * sigma ** 2
    b_d = kappa * (alpha - Dl) + rho * sigma * sigma_d
    c0 = -Dl
    # stencil weights (per h^2), interior
    wE = (a_zz - 0.5 * c) / h ** 2 + b_z / (2 * h)      # (i+1, j)
    wW = (a_zz - 0.5 * c) / h ** 2 - b_z / (2 * h)      # (i-1, j)
    wN = (a_dd - 0.5 * c) / h ** 2 + b_d / (2 * h)      # (i, j+1)
    wS = (a_dd - 0.5 * c) / h ** 2 - b_d / (2 * h)      # (i, j-1)
    wA = 0.5 * c / h ** 2                                # (i+1, j-1) and (i-1, j+1)
    wC = -(2 * a_zz + 2 * a_dd - c) / h ** 2 + c0        # centre (excluding the "1")
    mono = min(wE.min(), wW.min(), wN.min(), wS.min(), wA) >= 0
    assert mono, "scheme not monotone at this spacing"
    dt_max = safety / (-wC.min())
    nt = int(np.ceil(T / dt_max)); dt = T / nt
    if verbose:
        print(f"  grid {nz}x{nd}, h={h}, nt={nt}, dt={dt:.3e}; min weights E/W/N/S/A = "
              f"{wE.min():.3g}/{wW.min():.3g}/{wN.min():.3g}/{wS.min():.3g}/{wA:.3g}", flush=True)
    v = G.copy()
    I = slice(1, -1)
    t0 = time.time()
    for n in range(nt):
        Lv = (wE[I, I] * v[2:, 1:-1] + wW[I, I] * v[:-2, 1:-1] + wN[I, I] * v[1:-1, 2:] + wS[I, I] * v[1:-1, :-2]
              + wA * (v[2:, :-2] + v[:-2, 2:]) + wC[I, I] * v[1:-1, 1:-1])
        vn = v.copy()
        vn[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * Lv
        # edges: zero second derivative (linear extrapolation)
        vn[0, :] = 2 * vn[1, :] - vn[2, :]; vn[-1, :] = 2 * vn[-2, :] - vn[-3, :]
        vn[:, 0] = 2 * vn[:, 1] - vn[:, 2]; vn[:, -1] = 2 * vn[:, -2] - vn[:, -3]
        v = np.maximum(vn, G)
        if verbose and (n + 1) % max(1, nt // 5) == 0:
            print(f"    step {n+1}/{nt}  [{time.time()-t0:.0f}s]", flush=True)
    return z, y, d, v, G


def sqrt_extrap(x0, u0, x1, u1):
    """zero of u ~ a (x - b)^2 from two continuation nodes x0 < x1 (u0 < u1), both > 0."""
    s0, s1 = np.sqrt(u0), np.sqrt(u1)
    return (x1 * s0 - x0 * s1) / (s0 - s1)


def postprocess(z, y, d, v, G, edge=3):
    nz, nd = v.shape
    u = v - G
    cstar = np.full(nd, np.nan); k_node = np.full(nd, np.nan); k_ref = np.full(nd, np.nan)
    for j in range(nd):
        col = v[:, j]
        kk = np.where(col >= 0)[0]
        if len(kk) and kk.min() > 0:
            i = kk.min(); cstar[j] = y[i - 1] + (y[i] - y[i - 1]) * (-col[i - 1]) / (col[i] - col[i - 1])
        st = (u[:, j] <= 0.0)              # exact zeros from the projection
        # stopping region is an upper interval in y; find its first node, ignoring the top edge nodes
        i_first = None
        for i in range(edge, nz - edge):
            if st[i:nz - edge].all():
                i_first = i; break
        if i_first is not None:
            k_node[j] = y[i_first]
            if i_first >= edge + 2:
                i1, i0 = i_first - 1, i_first - 2      # last two continuation nodes (u>0)
                if u[i0, j] > 0 and u[i1, j] > 0 and u[i0, j] > u[i1, j]:
                    # in z: u ~ a (z_b - z)^2 for z < z_b
                    zb = sqrt_extrap(-z[i0], u[i0, j], -z[i1], u[i1, j])
                    k_ref[j] = np.exp(-zb)
    # critical yield: at y = 1 (z = 0), stopping region in delta is a lower interval {delta <= d(0,1)}
    i1 = int(np.argmin(np.abs(z)))
    row = u[i1, :]
    st = row <= 0.0
    j_last = None
    for j in range(nd - edge - 1, edge - 1, -1):
        if st[edge:j + 1].all():
            j_last = j; break
    dc_node = d[j_last] if j_last is not None else np.nan
    dc_ref = np.nan
    if j_last is not None and j_last + 2 < nd - edge:
        j0, j1 = j_last + 1, j_last + 2
        if row[j0] > 0 and row[j1] > 0 and row[j1] > row[j0]:
            dc_ref = sqrt_extrap(d[j0], row[j0], d[j1], row[j1])
    # independent estimate: root of k(0,delta) = 1
    ok = ~np.isnan(k_ref)
    dc_from_k = np.nan
    jj = np.where(ok)[0]
    if len(jj):
        kk_ = k_ref[jj]; dd_ = d[jj]
        above = np.where(kk_ > 1.0)[0]
        if len(above) and above.min() > 0:
            a_ = above.min(); dc_from_k = dd_[a_ - 1] + (dd_[a_] - dd_[a_ - 1]) * (1.0 - kk_[a_ - 1]) / (kk_[a_] - kk_[a_ - 1])
    return dict(cstar=cstar, k_node=k_node, k_ref=k_ref, dc_node=dc_node, dc_ref=dc_ref, dc_from_k=dc_from_k)


def Sigma(delta, p=PAR):
    return np.sqrt(p['sigma_d'] ** 2 + 2 * p['rho'] * p['sigma'] * p['sigma_d'] * delta + p['sigma'] ** 2 * delta ** 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spacing', nargs='*', type=float, help='legacy positional alternative to --spacings')
    parser.add_argument('--spacings', nargs='+', type=float,
                        help='grid spacings (default: 0.01 0.005 0.0025 0.00125); table yields must be grid nodes')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'generated',
                        help='output directory (default: code/generated beside this script)')
    args = parser.parse_args(argv)
    if args.spacing and args.spacings is not None:
        parser.error('use either positional spacings or --spacings, not both')
    hs = args.spacings or args.spacing or [0.01, 0.005, 0.0025, 0.00125]
    if any(not np.isfinite(h) or h <= 0 for h in hs):
        parser.error('spacings must be positive and finite')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    r = PAR['r']
    for T in [1 / 52, 1 / 12]:
        out[str(T)] = {}
        for h in hs:
            t0 = time.time()
            print(f"=== T = {T*52:.2f} weeks, h = {h} ===", flush=True)
            z, y, d, v, G = solve_gs(T, h)
            res = postprocess(z, y, d, v, G)
            from scipy.optimize import brentq
            dc_formal = brentq(lambda x: x + C * np.sqrt(T) * Sigma(x) - r, -1, 1)
            print(f"  delta_c: node {res['dc_node']:+.5f}  refined {res['dc_ref']:+.5f}  from k=1 {res['dc_from_k']:+.5f}   heuristic {dc_formal:+.5f}   [{time.time()-t0:.0f}s]")
            print("  delta0   1-c* (bp)   carry (bp)   k_node   k_ref")
            rows = []
            for d0 in YIELDS:
                j = int(round((d0 - d[0]) / h)); assert abs(d[j] - d0) < 1e-10
                rows.append(dict(delta0=d0, cstar=float(res['cstar'][j]), disc_bp=float((1 - res['cstar'][j]) * 1e4),
                                 carry_bp=float((1 - np.exp((r - d0) * T)) * 1e4), k_node=float(res['k_node'][j]), k_ref=float(res['k_ref'][j])))
                print(f"  {d0:+.2f}   {rows[-1]['disc_bp']:8.3f}   {rows[-1]['carry_bp']:+8.2f}   {rows[-1]['k_node']:.4f}   {rows[-1]['k_ref']:.4f}", flush=True)
            out[str(T)][str(h)] = dict(rows=rows, dc_node=float(res['dc_node']), dc_ref=float(res['dc_ref']), dc_from_k=float(res['dc_from_k']),
                                        dc_formal=float(dc_formal), delta=d.tolist(), cstar=res['cstar'].tolist(), k_ref=res['k_ref'].tolist(),
                                        k_node=res['k_node'].tolist(), ymin=float(y[0]), ymax=float(y[-1]))
            with (args.output_dir / 'gs_monotone_results.json').open('w') as output:
                json.dump(out, output)


if __name__ == "__main__":
    main()
