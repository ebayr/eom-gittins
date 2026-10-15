"""
Verify (Monte Carlo) the level-crossing theorem and the fixed-point identity for m*:

  K(t,x) = x / B(T-t)                      (universal signal = Gittins index of 'holding')
  tau*   = inf{t : K(t,X_t) <= m*} ^ T
  (i)   E[e^{-r tau*} (m* - X_tau*)] = 0          (m* is Bjork's price, V(m*) = 0)
  (ii)  E[e^{-r tau*} X_tau*] / E[e^{-r tau*}] = m*   (ratio formula attained)
  (iii) e^{-rT}(E X_T - m*) = E int_0^T r e^{-rt} (m* - inf_{s<t} K_s)^+ dt   (fixed-point identity)
Case: r=5%, q=4.5%, sigma=30%, T=1 (T_c = 0.3015 < T so m* < x).
"""
import argparse

import numpy as np
from gbm_solver import solve_implicit

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    r, q, sigma, T = 0.05, 0.045, 0.30, 1.0
    sol = solve_implicit(r, q, sigma, T, ny=3201, ns=4000)
    s, B = sol['s'], sol['B']
    cstar = sol['cstar'][-1]
    print(f"c* = m*/x = {cstar:.6f}   (x* = {1/cstar:.6f});  B(T)= {B[-1]:.4f}, B(0+)= {r/q:.4f}")
    m = cstar                      # x = 1
    Bfun = lambda tt: np.interp(T - tt, s, B)     # boundary as function of calendar time t

    rng = np.random.default_rng(7)
    M, N = 400_000, 4000
    dt = T / N
    X = np.ones(M)
    tau_idx = np.full(M, -1)
    Xtau = np.zeros(M)
    runinf = np.full(M, np.inf)          # inf_{s<t} K_s
    rhs_acc = np.zeros(M)                # int_0^t r e^{-rt} (m - runinf)^+ dt
    for n in range(0, N + 1):
        t = n * dt
        K = X / Bfun(t)
        # (iii) integrand uses inf over s<t, i.e. before updating with K_t
        rhs_acc += r * np.exp(-r * t) * np.maximum(m - runinf, 0.0) * dt
        runinf = np.minimum(runinf, K)
        hit = (K <= m) & (tau_idx < 0)
        tau_idx[hit] = n; Xtau[hit] = X[hit]
        if n < N:
            X = X * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(M))
    nots = tau_idx < 0
    tau_idx[nots] = N; Xtau[nots] = X[nots]
    tau = tau_idx * dt
    disc = np.exp(-r * tau)
    val = disc * (m - Xtau)
    print(f"(i)   E[e^-r tau* (m* - X)]      = {val.mean():+.5f} +- {val.std()/np.sqrt(M):.5f}   (should be 0)")
    print(f"(ii)  E[e^-r tau X]/E[e^-r tau]  = {(disc*Xtau).mean()/disc.mean():.5f}   vs m* = {m:.5f}")
    lhs = np.exp(-r*T) * (X.mean() - m)
    print(f"(iii) lhs e^-rT(E X_T - m*)      = {lhs:.5f}   rhs E int r e^-rt (m* - inf K)^+ = {rhs_acc.mean():.5f} +- {rhs_acc.std()/np.sqrt(M):.5f}")
    print(f"      P(deliver before T) = {np.mean(tau_idx < N):.3f},  E[tau*] = {tau.mean():.3f}")


if __name__ == "__main__":
    main()
