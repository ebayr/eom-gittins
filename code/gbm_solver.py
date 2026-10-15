"""
Obligation ("forward-type") stopping problem for GBM:

    V(s, x) = sup_{0<=tau<=s} E_x[ e^{-r tau} (1 - X_tau) ],   dX = (r-q) X dt + sigma X dW,

s = time to the end of the delivery window. Strike normalised to 1 (GBM scaling:
V(s,x;m) = m V(s, x/m; 1)).  Mandatory stopping at s = 0 (payoff 1 - x, no positive part).

Outputs:
  * V on a log-grid,
  * free boundary B(s) = sup{x : V(s,x) = 1-x}   (stopping region is {x <= B(s)}),
  * Gittins/universal signal  K(s, x) = x / B(s),
  * x*(s) = root of V(s, .) = 0  => Bjork price  m* = x / x*(s)  (c* = 1/x*),
  * critical horizon T_c = inf{s : B(s) <= 1}.

Scheme: implicit Euler in log-price with the Brennan-Schwartz projected tridiagonal solve
(exact for a one-sided left stopping region), plus an explicit scheme for cross-checks.
"""
import numpy as np
from scipy.linalg import solve_banded


def _grid(sigma, s_max, ny, ylo=None, yhi=None):
    # log-price grid wide enough for the diffusion over the horizon
    span = 6.0 * sigma * np.sqrt(max(s_max, 1e-6)) + 2.5
    ylo = -span if ylo is None else ylo
    yhi = span if yhi is None else yhi
    y = np.linspace(ylo, yhi, ny)
    return y


def brennan_schwartz(lower, diag, upper, rhs, obstacle):
    """Solve  A v = rhs  with  v >= obstacle  by the Brennan-Schwartz projection.
    Valid when the active (stopping) set is a left interval {i <= i*}.
    Elimination from the right, projection during the back-substitution from the left...
    Standard form: forward-eliminate from the *right* end so that the projection sweeps
    from the stopping region (left) outward.
    """
    n = len(diag)
    # eliminate the upper diagonal, sweeping from right to left
    d = diag.copy(); b = rhs.copy()
    for i in range(n - 2, -1, -1):
        w = upper[i] / d[i + 1]
        d[i] -= w * lower[i + 1]
        b[i] -= w * b[i + 1]
    # back-substitute from left to right with projection
    v = np.empty(n)
    v[0] = max(b[0] / d[0], obstacle[0])
    for i in range(1, n):
        v[i] = max((b[i] - lower[i] * v[i - 1]) / d[i], obstacle[i])
    return v


def solve_implicit(r, q, sigma, s_max, ny=1601, ns=2000, return_all=False):
    y = _grid(sigma, s_max, ny)
    dy = y[1] - y[0]
    x = np.exp(y)
    g = 1.0 - x                               # obstacle (payoff)
    ds = s_max / ns
    mu = r - q - 0.5 * sigma ** 2
    a = 0.5 * sigma ** 2 / dy ** 2
    bcoef = mu / (2 * dy)
    # operator L v = a (v_{i+1} - 2 v_i + v_{i-1}) + b (v_{i+1} - v_{i-1}) - r v_i
    lower = -ds * (a - bcoef) * np.ones(ny)
    diag = (1 + ds * (2 * a + r)) * np.ones(ny)
    upper = -ds * (a + bcoef) * np.ones(ny)
    V = g.copy()
    s_grid = np.linspace(0, s_max, ns + 1)
    B = np.empty(ns + 1); B[0] = (r / q) if q > 0 else np.inf
    xstar = np.empty(ns + 1); xstar[0] = 1.0
    Vall = [V.copy()] if return_all else None
    for n in range(1, ns + 1):
        s = s_grid[n]
        rhs = V.copy()
        # Dirichlet boundaries: left = stop (1 - x), right = hold to the end (e^{-rs} - x e^{-qs})
        rhs[0] = g[0]
        rhs[-1] = np.exp(-r * s) - x[-1] * np.exp(-q * s)
        dg = diag.copy(); lo = lower.copy(); up = upper.copy()
        dg[0] = 1; up[0] = 0; dg[-1] = 1; lo[-1] = 0
        V = brennan_schwartz(lo, dg, up, rhs, g)
        # boundary: last index where V == g (within tolerance)
        idx = np.where(V - g <= 1e-10)[0]
        B[n] = x[idx.max()] if len(idx) else np.nan
        # x*(s): root of V = 0 (V decreasing in x)
        k = np.where(V <= 0)[0]
        if len(k):
            k0 = k.min()
            if k0 > 0:
                # linear interpolation between k0-1 (V>0) and k0 (V<=0)
                x0, x1 = x[k0 - 1], x[k0]; v0, v1 = V[k0 - 1], V[k0]
                xstar[n] = x0 + (x1 - x0) * v0 / (v0 - v1)
            else:
                xstar[n] = x[0]
        else:
            xstar[n] = np.nan
        if return_all:
            Vall.append(V.copy())
    out = dict(s=s_grid, B=B, xstar=xstar, cstar=1.0 / xstar, y=y, x=x, V=V)
    if return_all:
        out['Vall'] = np.array(Vall)
    return out


def solve_explicit(r, q, sigma, s_max, ny=1201):
    """Explicit scheme (for cross-checking)."""
    y = _grid(sigma, s_max, ny)
    dy = y[1] - y[0]
    x = np.exp(y); g = 1.0 - x
    mu = r - q - 0.5 * sigma ** 2
    ds = 0.45 * dy ** 2 / sigma ** 2
    ns = int(np.ceil(s_max / ds)); ds = s_max / ns
    V = g.copy()
    for n in range(1, ns + 1):
        s = n * ds
        Vn = V.copy()
        Vn[1:-1] = V[1:-1] + ds * (0.5 * sigma ** 2 * (V[2:] - 2 * V[1:-1] + V[:-2]) / dy ** 2
                                   + mu * (V[2:] - V[:-2]) / (2 * dy) - r * V[1:-1])
        Vn[0] = g[0]; Vn[-1] = np.exp(-r * s) - x[-1] * np.exp(-q * s)
        V = np.maximum(Vn, g)
    idx = np.where(V - g <= 1e-12)[0]
    B = x[idx.max()]
    k0 = np.where(V <= 0)[0].min()
    x0, x1 = x[k0 - 1], x[k0]; v0, v1 = V[k0 - 1], V[k0]
    xstar = x0 + (x1 - x0) * v0 / (v0 - v1)
    return dict(x=x, V=V, B=B, xstar=xstar, cstar=1 / xstar)


def binomial(r, q, sigma, s_max, x0, N=4000):
    """CRR tree for V(s_max, x0) with mandatory exercise at the end (cross-check)."""
    dt = s_max / N
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(N + 1)
    S = x0 * u ** (N - j) * d ** j
    V = 1 - S
    for n in range(N - 1, -1, -1):
        S = x0 * u ** (n - np.arange(n + 1)) * d ** np.arange(n + 1)
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), 1 - S)
    return V[0]


def T_c(r, q, sigma, s_max=None, ny=2001, ns=4000):
    """critical horizon: first s with B(s) <= 1 (requires 0 < q < r)."""
    assert 0 < q < r
    if s_max is None:
        s_max = 4.0 * ((r / q - 1) / (0.5 * sigma)) ** 2 + 0.5
    for _ in range(6):
        sol = solve_implicit(r, q, sigma, s_max, ny=ny, ns=ns)
        below = np.where(sol['B'] <= 1.0)[0]
        if len(below):
            n = below.min()
            # refine by linear interpolation of B across 1
            s0, s1 = sol['s'][n - 1], sol['s'][n]; b0, b1 = sol['B'][n - 1], sol['B'][n]
            return s0 + (s1 - s0) * (b0 - 1) / (b0 - b1), sol
        s_max *= 3
    return np.inf, sol


if __name__ == "__main__":
    import time
    r, q, sigma = 0.05, 0.02, 0.30
    t0 = time.time()
    sol = solve_implicit(r, q, sigma, s_max=1.0, ny=1601, ns=2000)
    print("implicit  (s=1): B=%.5f  x*=%.5f  c*=%.5f   [%.1fs]" % (sol['B'][-1], sol['xstar'][-1], sol['cstar'][-1], time.time() - t0))
    t0 = time.time()
    ex = solve_explicit(r, q, sigma, s_max=1.0, ny=1201)
    print("explicit  (s=1): B=%.5f  x*=%.5f  c*=%.5f   [%.1fs]" % (ex['B'], ex['xstar'], ex['cstar'], time.time() - t0))
    for x0 in [0.8, 1.0, 1.2, 1.5]:
        vb = binomial(r, q, sigma, 1.0, x0, N=3000)
        vi = np.interp(np.log(x0), sol['y'], sol['V'])
        ve = np.interp(np.log(x0), np.log(ex['x']), ex['V'])
        print(f"x0={x0}: tree {vb:+.6f}  implicit {vi:+.6f}  explicit {ve:+.6f}   payoff {1-x0:+.3f}")
