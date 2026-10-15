"""Export the core paper's numerical tables from saved results.

The Markdown files use explicit display rounding. CSV files retain the saved
floating-point values and the unrounded Richardson calculation; unresolved
indices are blank in CSV, never replaced with zero. No solver is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent


def read_results(data_dir, name):
    # The archived solver output uses Python JSON's NaN for unresolved indices.
    with (data_dir / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def fixed(value, places):
    """Round a finite value to decimal places, with decimal half-up ties."""
    value = Decimal(str(value))
    rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return format(rounded, "f")


def significant(value, digits=4):
    value = Decimal(str(value))
    places = digits - 1 - value.adjusted() if value else digits - 1
    return fixed(value, places)


def finite_or_none(value):
    return value if math.isfinite(value) else None


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def critical_horizons(data):
    rows = []
    for item in sorted(data, key=lambda row: (row["r"], row["sigma"], -row["q"] / row["r"])):
        finest = max(item["levels"], key=lambda level: (level["ny"], level["ns"]))
        rows.append({
            "r": item["r"],
            "q": item["q"],
            "q_over_r": item["q"] / item["r"],
            "sigma": item["sigma"],
            "price_nodes": finest["ny"],
            "time_steps": finest["ns"],
            "tc_boundary_years": finest["tc_boundary"],
            "asymptotic_years": item["asym"],
        })
    display = [[fixed(row["r"] * 100, 1), fixed(row["sigma"] * 100, 0),
                fixed(row["q_over_r"], 2), significant(row["tc_boundary_years"]),
                significant(row["asymptotic_years"])] for row in rows]
    text = "# Critical horizons\n\n"
    text += ("Source: `results/tc_refined.json`. The numerical horizon uses the "
             "fitted-boundary estimate on the finest saved price/time grid "
             "(6401 price nodes and 4000 time steps in the supplied data). "
             "It does not use a Richardson estimate. The approximation is the "
             "saved value of `((r/q - 1)/(C*sigma))**2`, with `C = 0.63883322`.\n\n"
             "Both horizons are in years and are rounded to four significant "
             "figures using decimal half-up rounding. `tc.csv` contains the "
             "unrounded values and grid sizes. Values are generated from the "
             "data, so the last displayed digit can differ from the paper's "
             "manually entered table.\n\n")
    text += markdown_table(["r (%)", "sigma (%)", "q/r", "Numerical T_c (years)",
                            "Asymptotic T_c (years)"], display) + "\n"
    return rows, text


def gibson_schwartz(data):
    rows, grids = [], []
    for horizon, levels in sorted(data.items(), key=lambda item: float(item[0])):
        hkeys = sorted(levels, key=float)
        if len(hkeys) < 2:
            raise ValueError("GS index extrapolation requires two saved grid levels")
        hf, hc = map(float, hkeys[:2])
        if not math.isclose(hc, 2 * hf, rel_tol=1e-12):
            raise ValueError("GS table expects the two finest grid spacings to differ by a factor of two")
        fine, coarse = levels[hkeys[0]], levels[hkeys[1]]
        coarse_rows = {row["delta0"]: row for row in coarse["rows"]}
        T = float(horizon)
        if math.isclose(T, 1 / 52):
            label = "one week"
        elif math.isclose(T, 1 / 12):
            label = "one month"
        else:
            label = f"{T:g} years"
        grids.append(f"{label}: h_fine = {hf:g}, h_coarse = {hc:g}, y_min = {fine['ymin']:.8g}")
        for fine_row in sorted(fine["rows"], key=lambda row: row["delta0"]):
            coarse_row = coarse_rows[fine_row["delta0"]]
            kf = finite_or_none(fine_row["k_ref"])
            kc = finite_or_none(coarse_row["k_ref"])
            kr = 2 * kf - kc if kf is not None and kc is not None else None
            rows.append({
                "window": label,
                "window_years": T,
                "delta0": fine_row["delta0"],
                "h_fine": hf,
                "h_coarse": hc,
                "y_min": fine["ymin"],
                "cstar_fine": fine_row["cstar"],
                "discount_bp": fine_row["disc_bp"],
                "carry_bp": fine_row["carry_bp"],
                "k_node_fine": finite_or_none(fine_row["k_node"]),
                "k_ref_fine": kf,
                "k_ref_coarse": kc,
                "k_richardson": kr,
                "index_status": "resolved" if kr is not None else "unresolved",
                "critical_yield_fine": fine["dc_ref"],
            })
    display = [[row["window"], fixed(row["delta0"] * 100, 0), fixed(row["discount_bp"], 2),
                fixed(row["carry_bp"], 1),
                fixed(row["k_richardson"], 2) if row["k_richardson"] is not None else "unresolved"]
               for row in rows]
    text = "# Gibson–Schwartz results\n\n"
    text += ("Source: `results/gs_monotone_results.json`. Discounts and carry "
             "values use the finest saved grid. The index uses first-order "
             "Richardson extrapolation from the two finest grids: "
             "`k_R = 2*k_ref(h_fine) - k_ref(h_coarse)`, where "
             "`h_coarse = 2*h_fine`. This is the index used in the paper's "
             "table; the figure instead displays the finest-grid index.\n\n")
    text += "Saved grids: " + "; ".join(grids) + ".\n\n"
    text += ("Discount = `10000*(1-cstar)`; carry = "
             "`10000*(1-exp((r-delta0)*T))`, with `r = 0.06`. Both are in "
             "basis points. Discounts are rounded to two decimal places, "
             "carry to one, and indices to two, using decimal half-up rounding. "
             "`gs.csv` preserves the unrounded saved values, both inputs to "
             "the extrapolation, and its unrounded result. Last displayed "
             "digits can differ from the paper's manually entered table.\n\n"
             "An unresolved index is not extrapolated or replaced with zero. "
             "For the supplied table rows this occurs when the stopping region "
             "reaches the computational floor near y = 0.05 (reported as "
             "`<0.05` in the paper). Corresponding CSV index fields are blank "
             "and `index_status` records `unresolved`.\n\n")
    text += markdown_table(["Window", "Initial yield (%)", "Discount (bp)",
                            "Carry (bp)", "Index k_R"], display) + "\n"
    return rows, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", "--data-dir", dest="data_dir", type=Path,
                        default=CODE_DIR / "results", help="directory containing the saved JSON results")
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "tables")
    args = parser.parse_args()
    tc_rows, tc_text = critical_horizons(read_results(args.data_dir, "tc_refined.json"))
    gs_rows, gs_text = gibson_schwartz(read_results(args.data_dir, "gs_monotone_results.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows, text in (("tc", tc_rows, tc_text), ("gs", gs_rows, gs_text)):
        write_csv(args.output_dir / f"{name}.csv", rows)
        (args.output_dir / f"{name}.md").write_text(text, encoding="utf-8")
    print(f"Wrote tc.md, tc.csv, gs.md and gs.csv to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
