"""Check the release data and run inexpensive independent solver checks.

Run ``python code/smoke_check.py --data-only`` with the standard library alone.
The default also runs one coarse Gibson--Schwartz grid and a small GBM/CRR
comparison; it requires the numerical dependencies in requirements.txt.
These checks detect broken inputs and regressions, not discretization error in
the published convergence studies. No result files are written.
"""

import argparse
import json
import math
from pathlib import Path
import sys


RESULTS = Path(__file__).resolve().parent / "results"
TC_QS = (0.049, 0.048, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02)
TC_SIGMAS = (0.15, 0.30, 0.50)
TC_LEVELS = ((1601, 1000), (3201, 2000), (6401, 4000))
GS_TIMES = (1 / 52, 1 / 12)
GS_SPACINGS = (0.01, 0.005, 0.0025, 0.00125)
GS_YIELDS = (-0.20, 0.0, 0.02, 0.03, 0.04, 0.06, 0.10, 0.20)
ROUNDING_TOL = 1e-10


class CheckFailure(Exception):
    """A failed data or numerical invariant."""


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def close(actual, expected, label, atol=ROUNDING_TOL, allow_nan=False):
    if allow_nan and math.isnan(actual) and math.isnan(expected):
        return
    require(math.isfinite(actual) and math.isfinite(expected),
            f"{label}: expected finite values, got {actual!r} and {expected!r}")
    require(abs(actual - expected) <= atol,
            f"{label}: {actual:.12g} != {expected:.12g} (absolute tolerance {atol:g})")


def monotone(values, label, increasing=True, atol=ROUNDING_TOL):
    direction = 1 if increasing else -1
    require(all(math.isfinite(x) for x in values), f"{label}: nonfinite value")
    for i, (a, b) in enumerate(zip(values, values[1:])):
        require(direction * (b - a) >= -atol,
                f"{label}: monotonicity failed at entries {i}, {i + 1}: {a}, {b}")


def bounded(values, lo, hi, label, allow_nan=False):
    for i, value in enumerate(values):
        if allow_nan and math.isnan(value):
            continue
        require(math.isfinite(value) and lo <= value <= hi,
                f"{label}[{i}]: {value!r} outside [{lo}, {hi}]")


def read_result(name):
    # The archived GS arrays use NaN for a boundary outside the usable grid.
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def check_critical_horizons(rows):
    expected = {(sigma, q) for sigma in TC_SIGMAS for q in TC_QS}
    require(len(rows) == 24, "tc_refined.json: expected 24 parameter rows")
    require({(row["sigma"], row["q"]) for row in rows} == expected,
            "tc_refined.json: unexpected or duplicate parameter combinations")
    for row in rows:
        label = f"TC sigma={row['sigma']}, q={row['q']}"
        close(row["r"], 0.05, label + " r")
        levels = row["levels"]
        require([(level["ny"], level["ns"]) for level in levels] == list(TC_LEVELS),
                label + ": expected all three grid levels in order")
        for level in levels:
            vals = [level[key] for key in ("tc_node_lower", "tc_node_upper", "s_max")]
            bounded(vals, 0.0, math.inf, label + " estimates")
            require(vals[0] > 0, label + ": nonpositive critical horizon")
            monotone(vals, label + " node bracket inside horizon")
            # Fitted extrapolations are separate estimators and need not lie
            # inside the discrete node bracket at a finite grid spacing.
            bounded([level["tc_boundary"]], 0.0, level["s_max"], label + " fitted boundary estimate")
            bounded([level["tc_sqrt"]], 0.0, level["s_max"], label + " sqrt estimate")
        close(row["tc_rich"], 2 * levels[-1]["tc_boundary"] - levels[-2]["tc_boundary"],
              label + " Richardson estimate")
        close(row["asym"], ((row["r"] / row["q"] - 1) / (0.63883322 * row["sigma"])) ** 2,
              label + " asymptotic estimate", atol=1e-8)
        require(row["tc_rich"] > 0, label + ": nonpositive Richardson estimate")
    lookup = {(row["sigma"], row["q"]): row for row in rows}
    for sigma in TC_SIGMAS:
        monotone([lookup[sigma, q]["tc_rich"] for q in TC_QS],
                 f"TC increases as q decreases, sigma={sigma}")
    for q in TC_QS:
        monotone([lookup[sigma, q]["tc_rich"] for sigma in TC_SIGMAS],
                 f"TC decreases as sigma increases, q={q}", increasing=False)


def check_boundaries(data):
    require(set(data) == {"0.045", "0.04", "0.03", "0.02"},
            "boundaries_refined.json: expected four q series")
    for q_text, row in data.items():
        q = float(q_text)
        label = f"GBM boundary q={q}"
        require(set(row) == {"s", "Bn", "Br", "cstar"}, label + ": unexpected fields")
        require(all(len(values) == 4001 for values in row.values()),
                label + ": expected 4001 points per array")
        for i, s in enumerate(row["s"]):
            close(s, i / 4000, label + f" time[{i}]")
        for key in ("Bn", "Br"):
            bounded(row[key], 1e-12, 0.05 / q + ROUNDING_TOL, label + " " + key)
            close(row[key][0], 0.05 / q, label + " initial " + key)
        # Sub-grid square-root fits have small sawtooth movements. The actual
        # stopping-node boundary has the monotonicity invariant.
        monotone(row["Bn"], label + " node boundary", increasing=False)
        bounded(row["cstar"], 1e-12, 1 + ROUNDING_TOL, label + " cstar")
        close(row["cstar"][0], 1.0, label + " initial cstar")
        monotone(row["cstar"], label + " price over time", increasing=False)


def check_gs(data):
    require(set(data) == {str(T) for T in GS_TIMES},
            "gs_monotone_results.json: expected one-week and one-month horizons")
    for T in GS_TIMES:
        levels = data[str(T)]
        require(set(levels) == {str(h) for h in GS_SPACINGS},
                f"GS T={T}: expected all four grid spacings")
        for h in GS_SPACINGS:
            record = levels[str(h)]
            label = f"GS T={T}, h={h}"
            n = round(1.8 / h) + 1
            for key in ("delta", "cstar", "k_node", "k_ref"):
                require(len(record[key]) == n, label + f": {key} length must be {n}")
            for i, delta in enumerate(record["delta"]):
                close(delta, -0.9 + i * h, label + f" delta[{i}]")
            close(record["ymin"], math.exp(-3), label + " ymin")
            close(record["ymax"], math.exp(2), label + " ymax")
            bounded(record["cstar"], 1e-12, 1 + ROUNDING_TOL, label + " cstar")
            for key in ("k_node", "k_ref"):
                bounded(record[key], record["ymin"], record["ymax"],
                        label + " " + key, allow_nan=True)
            for key in ("dc_node", "dc_ref", "dc_from_k", "dc_formal"):
                bounded([record[key]], -0.2, 0.2, label + " " + key)
            rows = record["rows"]
            require([row["delta0"] for row in rows] == list(GS_YIELDS),
                    label + ": expected eight table yields in order")
            for row in rows:
                delta = row["delta0"]
                j = round((delta + 0.9) / h)
                close(record["delta"][j], delta, label + " table yield on grid")
                for key in ("cstar", "k_node", "k_ref"):
                    close(row[key], record[key][j], label + f" {key} at delta={delta}",
                          allow_nan=(key == "k_ref" and delta == -0.2))
                close(row["disc_bp"], (1 - row["cstar"]) * 1e4,
                      label + " discount in bp", atol=1e-8)
                close(row["carry_bp"], (1 - math.exp((0.06 - delta) * T)) * 1e4,
                      label + " carry in bp", atol=1e-8)
            monotone([row["cstar"] for row in rows], label + " price by yield", increasing=False)
            monotone([row["disc_bp"] for row in rows], label + " discount by yield")
            # Restrict boundary monotonicity to reported interior yields; low
            # yields and remote edges can be clipped or have no finite fit.
            for key in ("k_node", "k_ref"):
                monotone([row[key] for row in rows if 0 <= row["delta0"] <= 0.2],
                         label + " interior " + key + " by yield")
    for h in GS_SPACINGS:
        week = data[str(GS_TIMES[0])][str(h)]
        month = data[str(GS_TIMES[1])][str(h)]
        for w, m in zip(week["rows"], month["rows"]):
            require(m["cstar"] <= w["cstar"] + ROUNDING_TOL,
                    f"GS h={h}, delta={w['delta0']}: monthly price exceeds weekly price")


def check_extra_rows(rows):
    expected = [(q, 1 / 52) for q in (0.05, 0.055, 0.06, 0.08, 0.10, 0.15)]
    expected += [(q, 1.0) for q in (0.045, 0.05, 0.055, 0.06)]
    require([(row["q"], row.get("T", 1 / 52)) for row in rows] == expected,
            "extra_rows.json: expected six weekly and four annual rows")
    for row in rows:
        q, T = row["q"], row.get("T", 1 / 52)
        label = f"extra q={q}, T={T}"
        close(row["carry"], math.exp((0.05 - q) * T), label + " carry")
        bounded([row["B"]], 1e-12, 0.05 / q + ROUNDING_TOL, label + " boundary")
        bounded([row["cstar"]], 1e-12, 1 + ROUNDING_TOL, label + " price")
        # The coarse saved PDE/interpolation can exceed the carry bound by
        # about 5e-9. Allow 1e-8 in normalized price (0.0001 bp).
        require(row["cstar"] <= min(1.0, row["carry"]) + 1e-8,
                label + ": price exceeds immediate-delivery/hold-to-end bound")
    for group in (rows[:6], rows[6:]):
        monotone([row["cstar"] for row in group], "extra price by yield", increasing=False)
        monotone([row["B"] for row in group], "extra boundary by yield", increasing=False)


def check_solvers(gs_data):
    # Keep imports lazy so --data-only needs no third-party packages.
    import numpy as np
    from gbm_solver import binomial, solve_implicit
    from gs_monotone import PAR, postprocess, solve_gs

    T, h = 1 / 52, 0.01
    saved = gs_data[str(T)][str(h)]
    z, y, delta, value, obstacle = solve_gs(T, h, verbose=False)
    require(value.shape == (501, 181), "GS smoke grid: unexpected shape")
    require(bool(np.all(np.isfinite(value))), "GS smoke grid: nonfinite solution")
    require(bool(np.all(value >= obstacle - 1e-12)), "GS smoke grid: obstacle violated")
    result = postprocess(z, y, delta, value, obstacle)
    for key in ("dc_node", "dc_ref", "dc_from_k"):
        close(float(result[key]), saved[key], "GS recomputation " + key, atol=1e-7)
    for row in saved["rows"]:
        d0 = row["delta0"]
        j = round((d0 - float(delta[0])) / h)
        close(float(delta[j]), d0, "GS recomputation table grid")
        for key in ("cstar", "k_node", "k_ref"):
            close(float(result[key][j]), row[key], f"GS recomputation {key}, delta={d0}",
                  atol=1e-7, allow_nan=(key == "k_ref" and d0 == -0.2))
        close((1 - float(result["cstar"][j])) * 1e4, row["disc_bp"],
              f"GS recomputation discount, delta={d0}", atol=1e-5)
        close((1 - math.exp((PAR["r"] - d0) * T)) * 1e4, row["carry_bp"],
              f"GS recomputation carry, delta={d0}", atol=1e-8)
    print("PASS coarse GS: obstacle, three critical-yield estimates, all eight table rows")

    r, q, sigma, T = 0.05, 0.045, 0.30, 1.0
    sol = solve_implicit(r, q, sigma, T, ny=1001, ns=800)
    require(bool(np.all(np.isfinite(sol["V"]))), "GBM smoke grid: nonfinite solution")
    require(bool(np.all(sol["V"] >= 1 - sol["x"] - 1e-12)),
            "GBM smoke grid: obstacle violated")
    monotone(sol["V"], "GBM value by underlying price", increasing=False)
    errors = []
    for x0 in (0.8, 1.0, 1.2):
        fd = float(np.interp(math.log(x0), sol["y"], sol["V"]))
        tree = float(binomial(r, q, sigma, T, x0, N=1200))
        close(fd, tree, f"GBM implicit versus CRR at x={x0}", atol=1e-4)
        errors.append(abs(fd - tree))
    print(f"PASS GBM: obstacle, value monotonicity, three CRR comparisons "
          f"(max absolute error {max(errors):.3g}; tolerance 1e-4)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true",
                        help="validate saved JSON with the standard library; skip solver runs")
    args = parser.parse_args(argv)
    try:
        checks = (
            ("tc_refined.json", check_critical_horizons, "24 critical-horizon rows and three levels"),
            ("boundaries_refined.json", check_boundaries, "four GBM boundary/price series"),
            ("gs_monotone_results.json", check_gs, "two GS horizons, four levels, eight yields"),
            ("extra_rows.json", check_extra_rows, "ten supplementary GBM rows"),
        )
        data = {}
        for filename, check, description in checks:
            data[filename] = read_result(filename)
            check(data[filename])
            print("PASS saved data: " + description)
        if not args.data_only:
            check_solvers(data["gs_monotone_results.json"])
    except ImportError as error:
        print(f"FAIL numerical dependencies: {error}. Install requirements.txt, "
              "or use --data-only for saved-data checks.", file=sys.stderr)
        return 1
    except (CheckFailure, OSError, ValueError, TypeError, KeyError, IndexError, AssertionError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("All smoke checks passed" + (" (saved data only)." if args.data_only else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
