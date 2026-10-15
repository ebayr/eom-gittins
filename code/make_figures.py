"""Figures for the revised paper, from the refined/monotone computations:
   fig_boundary.pdf  (boundaries_refined.json, tc_refined.json)
   fig_phase.pdf     (tc_refined.json)
   fig_gs.pdf        (gs_monotone_results.json, finest grid; index censored at the domain floor)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results",
                    help="Directory containing the three numerical result datasets.")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = args.results_dir

def read_results(name):
    with (DATA_DIR / name).open() as stream:
        return json.load(stream)

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{name}.pdf")
    fig.savefig(args.output_dir / f"{name}-1.png", dpi=180)
    plt.close(fig)

# Parameter identity uses both a line pattern and a marker shape.
STYLES = ["-", "--", "-."]
MARKERS = ["o", "s", "^"]
CORE_PAL = ["#0072B2", "#D55E00", "#009E73"]

def numerical_style(k, count):
    return dict(color=CORE_PAL[k], linestyle=STYLES[k], marker=MARKERS[k],
                markevery=(k + 1, max(1, count // 8)), markersize=4.2,
                markerfacecolor=CORE_PAL[k], markeredgewidth=0.7,
                linewidth=1.7, zorder=3)

def asymptotic_style(k, count):
    return dict(color="#202020", linestyle=":", marker=MARKERS[k],
                markevery=(max(1, count // 8), max(1, count // 4)), markersize=4.8,
                markerfacecolor="white", markeredgewidth=0.9,
                linewidth=1.25, zorder=4)

def add_asymptotic_key(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#202020", linestyle=":", linewidth=1.25))
    labels.append("asymptotic: open symbols")
    ax.legend(handles, labels, handlelength=3.2, **kwargs)

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#4a4a4a", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
    "figure.dpi": 150,
})
C = 0.63883322
r, sigma = 0.05, 0.30

tcrows = read_results("tc_refined.json")
def tc_of(q, sg):
    for row in tcrows:
        if abs(row['q']-q) < 1e-12 and abs(row['sigma']-sg) < 1e-12:
            return row['levels'][-1]['tc_boundary']
    return np.nan

# ---------- Figure 4.1: boundary for GBM ----------
bd = read_results("boundaries_refined.json")
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for k, q in enumerate([0.03, 0.04, 0.045]):
    d = bd[str(q)]; s = np.array(d['s']); B = np.array(d['Br']); Bn = np.array(d['Bn'])
    B = np.where(np.isfinite(B), B, Bn)
    ax.plot(s, B, **numerical_style(k, len(s)), label=f"q = {q*100:.1f}%  (r/q = {r/q:.2f})")
    tc = tc_of(q, 0.30)
    if np.isfinite(tc) and tc <= 1.0:
        ax.plot([tc], [1.0], marker=MARKERS[k], color=CORE_PAL[k], ms=5, zorder=5)
        ax.annotate(f"$T_c$ = {tc:.3f}", (tc, 1.0), xytext=(4, 6 if k % 2 == 0 else -13), textcoords="offset points", color=INK, fontsize=8)
    ss = np.linspace(0, 0.2, 50)
    ax.plot(ss, (r/q)*(1 - C*sigma*np.sqrt(ss)), **asymptotic_style(k, len(ss)))
ax.axhline(1.0, color=INK, lw=0.8)
ax.set_xlabel("time to end of window  s  (years)")
ax.set_ylabel("B(s)  (delivery boundary, price 1)")
ax.set_title("GBM delivery boundary  (r = 5%, σ = 30%)", fontsize=9, color=INK, loc="left")
ax.set_ylim(0.85, 1.75); ax.set_xlim(0, 1.0)
add_asymptotic_key(ax, loc="upper right", fontsize=8)
save_figure(fig, "fig_boundary")

# ---------- Figure 4.2: phase diagram ----------
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for k, sg in enumerate([0.15, 0.30, 0.50]):
    pts = sorted([(row['q']/r, row['levels'][-1]['tc_boundary']) for row in tcrows if abs(row['sigma']-sg) < 1e-9])
    qs = np.array([p[0] for p in pts]); tcs = np.array([p[1] for p in pts])
    style = numerical_style(k, len(qs))
    style["markevery"] = 1
    ax.plot(qs, tcs*52, **style, label=f"σ = {int(sg*100)}%  (PDE)")
    qq = np.linspace(0.4, 0.995, 200)
    ax.plot(qq, ((1/qq - 1)/(C*sg))**2*52, **asymptotic_style(k, len(qq)))
ax.axhline(1.0, color=INK, lw=0.8); ax.text(0.41, 1.15, "one-week window", color=INK, fontsize=8)
ax.set_yscale("log"); ax.set_xlim(0.38, 1.0)
ax.set_xlabel("convenience yield / interest rate   q / r")
ax.set_ylabel("critical horizon  $T_c$  (weeks, log scale)")
ax.set_title("Critical delivery-window length  (r = 5%)", fontsize=9, color=INK, loc="left")
add_asymptotic_key(ax, loc="lower left", fontsize=8)
save_figure(fig, "fig_phase")

# ---------- Figure 4.3: Gibson-Schwartz ----------
gs = read_results("gs_monotone_results.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
for k, (key, lab) in enumerate([(str(1/52), "one-week window"), (str(1/12), "one-month window")]):
    hs = sorted(gs[key].keys(), key=float)
    R = gs[key][hs[0]]                                  # finest grid
    dl = np.array(R['delta']); cs = np.array(R['cstar']); kr = np.array(R['k_ref']); kn = np.array(R['k_node'])
    ok = (dl >= -0.25) & (dl <= 0.3)
    ax1.plot(dl[ok]*100, (1-cs[ok])*1e4, **numerical_style(k, int(ok.sum())), label=lab)
    ax1.axvline(R['dc_ref']*100, color=INK, ls=":", lw=1.0, zorder=2)
    ax1.plot([R['dc_ref']*100], [0], marker=MARKERS[k], color=INK,
             markerfacecolor="white", ms=5.5, markeredgewidth=1.1, zorder=5)
    # index: show only where resolved (not at the lower domain edge) and inside the plotted range
    floor = R['ymin']*np.exp(3*float(hs[0]))*1.05
    res = ok & np.isfinite(kr) & (kn > floor)
    ax2.plot(dl[res]*100, kr[res], **numerical_style(k, int(res.sum())), label=lab)
ax1.axvline(6.0, color=INK, lw=0.8); ax1.text(6.3, 60, "δ₀ = r", fontsize=8, color=INK)
ax1.set_xlabel("initial convenience yield  δ₀  (%)"); ax1.set_ylabel("discount  1 − m*/X₀  (bp)")
ax1.set_title("Futures price discount", fontsize=9, loc="left", color=INK)
ax1.set_xlim(-25, 30); ax1.set_ylim(-4, 130)
ax1.legend(fontsize=8, loc="upper left", handlelength=3.2, frameon=True,
           facecolor="white", framealpha=1, edgecolor="none")
ax2.axhline(1.0, color=INK, lw=0.8); ax2.text(-24, 1.07, "k = 1", fontsize=8, color=INK)
ax2.set_xlabel("initial convenience yield  δ₀  (%)"); ax2.set_ylabel("index per unit of spot  k(0, δ₀)")
ax2.set_title("Holding index", fontsize=9, loc="left", color=INK)
ax2.set_xlim(-25, 30); ax2.set_ylim(0, 6); ax2.legend(fontsize=8, loc="upper left", handlelength=3.2)
save_figure(fig, "fig_gs")

print("Three core figures written to", args.output_dir)
