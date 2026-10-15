"""Dinkelbach-Newton iteration (Theorem 'newton') for GBM, computed by PDE:
  m_{k+1} = E[e^{-r tau_k} X_{tau_k}] / E[e^{-r tau_k}],  tau_k = inf{t : X_t <= m_k B(T-t)} ^ T.
The two expectations are solutions of linear backward PDEs with the value forced to the
payoff (X resp. 1) inside the stopping set {x <= m_k B(T-t)}.  Explicit scheme in log-price."""
import argparse

import numpy as np
from gbm_solver import solve_implicit

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    r, q, sigma, T = 0.05, 0.045, 0.30, 1.0
    sol = solve_implicit(r, q, sigma, T, ny=3201, ns=4000)
    s_grid, B, cstar = sol['s'], sol['B'], sol['cstar'][-1]
    print(f"PDE fixed point: c* = {cstar:.6f}")

    def expectations(m, ny=2001):
        span = 6*sigma*np.sqrt(T) + 2.5
        y = np.linspace(-span, span, ny); dy = y[1]-y[0]; x = np.exp(y)
        mu = r - q - 0.5*sigma**2
        ds = 0.45*dy**2/sigma**2; ns = int(np.ceil(T/ds)); ds = T/ns
        N = x.copy(); D = np.ones(ny)          # at s=0 (t=T): forced stop: N = X, D = 1
        for n in range(1, ns+1):
            s = n*ds; b = m*np.interp(s, s_grid, B)
            for V in (N, D):
                Vn = V.copy()
                Vn[1:-1] = V[1:-1] + ds*(0.5*sigma**2*(V[2:]-2*V[1:-1]+V[:-2])/dy**2 + mu*(V[2:]-V[:-2])/(2*dy) - r*V[1:-1])
                Vn[0] = V[0]; Vn[-1] = V[-1]
                V[:] = Vn
            stop = x <= b
            N[stop] = x[stop]; D[stop] = 1.0
        i = np.argmin(np.abs(x-1.0))
        return N[i], D[i]

    m = 1.0
    for k in range(3):
        Nk, Dk = expectations(m)
        m_new = Nk/Dk
        print(f"k={k}: m_k = {m:.6f}  E[e^-r tau X] = {Nk:.6f}  E[e^-r tau] = {Dk:.6f}  ->  m_{k+1} = {m_new:.6f}   (m_{k+1}-c*)/(m_k-c*) = {(m_new-cstar)/(m-cstar) if abs(m-cstar)>1e-12 else float('nan'):.4f}")
        m = m_new
    print("contraction bound 1 - e^{-rT} =", 1-np.exp(-r*T))


if __name__ == "__main__":
    main()
