"""
Universal constant  C := sup_{tau in (0,1]} E[ int_0^tau B_s ds ] / E[tau]   (B = standard BM, B_0 = 0).

Equivalently C = -beta(1) where beta(s) = -C sqrt(s) is the free boundary of
    w(s, b) = sup_{tau <= s} E_b int_0^tau B_u du      (no discounting),
whose similarity form  w = s^{3/2} f(b/sqrt(s))  solves
    f'' + z f' - 3 f + 2 z = 0  on z > -C,   f(-C) = f'(-C) = 0,   f(z) ~ z  (z -> +inf).
General solution f = z + A g(z), where g is the decaying solution of f'' + z f' - 3f = 0:
    g(z) = e^{-z^2/2} int_0^inf t^3 e^{-z t - t^2/2} dt  (= 6 e^{-z^2/4} D_{-4}(z)).
Smooth fit =>  g(-C) + C g'(-C) = 0.
Cross-check by an explicit finite-difference solve of the obstacle problem.
"""
import argparse

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def g(z):
    return np.exp(-z**2/2) * quad(lambda t: t**3*np.exp(-z*t - t**2/2), 0, np.inf)[0]

def gp(z):   # derivative: d/dz of the integral representation
    I3 = quad(lambda t: t**3*np.exp(-z*t - t**2/2), 0, np.inf)[0]
    I4 = quad(lambda t: t**4*np.exp(-z*t - t**2/2), 0, np.inf)[0]
    return np.exp(-z**2/2) * (-z*I3 - I4)

# ---- cross-check: FD solve of sup_tau E int_0^tau B_u du on horizon 1, boundary at s=1 ----
def fd_boundary(nb=4001, bmax=8.0, s_max=1.0):
    b = np.linspace(-bmax, bmax, nb); db = b[1]-b[0]
    ds = 0.45*db**2; ns = int(np.ceil(s_max/ds)); ds = s_max/ns
    w = np.zeros(nb)
    for n in range(1, ns+1):
        wn = w.copy()
        wn[1:-1] = w[1:-1] + ds*(0.5*(w[2:]-2*w[1:-1]+w[:-2])/db**2 + b[1:-1])
        wn[0] = 0.0; wn[-1] = b[-1]*n*ds     # far field: never stop => b*s
        w = np.maximum(wn, 0.0)
    # boundary = largest b with w == 0
    idx = np.where(w <= 1e-13)[0]
    return -b[idx.max()], b, w

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ode-only', action='store_true', help='compute the quadrature/root result without finite differences')
    group.add_argument('--fd-nodes', nargs='+', type=int, default=[2001, 4001, 8001],
                       help='finite-difference grid sizes (default: 2001 4001 8001)')
    args = parser.parse_args(argv)
    if any(nb < 3 for nb in args.fd_nodes):
        parser.error('finite-difference grids need at least three nodes')

    # sanity: g solves f'' + z f' - 3 f = 0  (check numerically with finite differences)
    z0, h = 0.3, 1e-4
    gpp = (g(z0+h) - 2*g(z0) + g(z0-h))/h**2
    print("ODE residual at z=0.3:", gpp + z0*gp(z0) - 3*g(z0))

    F = lambda c: g(-c) + c*gp(-c)
    C_ode = brentq(F, 0.05, 3.0, xtol=1e-12)
    A = -1/gp(-C_ode)
    print(f"C (similarity ODE) = {C_ode:.8f}   A = {A:.6f}")

    if not args.ode_only:
        for nb in args.fd_nodes:
            Cfd, _, _ = fd_boundary(nb=nb)
            print(f"C (FD, nb={nb}) = {Cfd:.5f}")


if __name__ == "__main__":
    main()
