# Gibson–Schwartz results

Source: `results/gs_monotone_results.json`. Discounts and carry values use the finest saved grid. The index uses first-order Richardson extrapolation from the two finest grids: `k_R = 2*k_ref(h_fine) - k_ref(h_coarse)`, where `h_coarse = 2*h_fine`. This is the index used in the paper's table; the figure instead displays the finest-grid index.

Saved grids: one week: h_fine = 0.00125, h_coarse = 0.0025, y_min = 0.049787068; one month: h_fine = 0.00125, h_coarse = 0.0025, y_min = 0.049787068.

Discount = `10000*(1-cstar)`; carry = `10000*(1-exp((r-delta0)*T))`, with `r = 0.06`. Both are in basis points. Discounts are rounded to two decimal places, carry to one, and indices to two, using decimal half-up rounding. `gs.csv` preserves the unrounded saved values, both inputs to the extrapolation, and its unrounded result. Last displayed digits can differ from the paper's manually entered table.

An unresolved index is not extrapolated or replaced with zero. For the supplied table rows this occurs when the stopping region reaches the computational floor near y = 0.05 (reported as `<0.05` in the paper). Corresponding CSV index fields are blank and `index_status` records `unresolved`.

| Window | Initial yield (%) | Discount (bp) | Carry (bp) | Index k_R |
| --- | --- | --- | --- | --- |
| one week | -20 | 0.00 | -50.1 | unresolved |
| one week | 0 | 0.00 | -11.5 | 0.59 |
| one week | 2 | 0.00 | -7.7 | 0.92 |
| one week | 3 | 0.07 | -5.8 | 1.09 |
| one week | 4 | 0.51 | -3.8 | 1.26 |
| one week | 6 | 2.36 | 0.0 | 1.60 |
| one week | 10 | 8.33 | 7.7 | 2.28 |
| one week | 20 | 26.77 | 26.9 | 3.97 |
| one month | -20 | 0.00 | -219.0 | unresolved |
| one month | 0 | 1.22 | -50.1 | 1.27 |
| one month | 2 | 5.72 | -33.4 | 1.61 |
| one month | 3 | 9.00 | -25.0 | 1.77 |
| one month | 4 | 12.87 | -16.7 | 1.94 |
| one month | 6 | 22.08 | 0.0 | 2.28 |
| one month | 10 | 44.92 | 33.3 | 2.95 |
| one month | 20 | 114.78 | 116.0 | 4.63 |
