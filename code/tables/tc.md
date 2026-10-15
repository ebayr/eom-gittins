# Critical horizons

Source: `results/tc_refined.json`. The numerical horizon uses the fitted-boundary estimate on the finest saved price/time grid (6401 price nodes and 4000 time steps in the supplied data). It does not use a Richardson estimate. The approximation is the saved value of `((r/q - 1)/(C*sigma))**2`, with `C = 0.63883322`.

Both horizons are in years and are rounded to four significant figures using decimal half-up rounding. `tc.csv` contains the unrounded values and grid sizes. Values are generated from the data, so the last displayed digit can differ from the paper's manually entered table.

| r (%) | sigma (%) | q/r | Numerical T_c (years) | Asymptotic T_c (years) |
| --- | --- | --- | --- | --- |
| 5.0 | 15 | 0.98 | 0.04442 | 0.04536 |
| 5.0 | 15 | 0.96 | 0.1811 | 0.1891 |
| 5.0 | 15 | 0.90 | 1.195 | 1.344 |
| 5.0 | 15 | 0.80 | 5.174 | 6.806 |
| 5.0 | 15 | 0.70 | 12.44 | 20.00 |
| 5.0 | 15 | 0.60 | 23.45 | 48.40 |
| 5.0 | 15 | 0.50 | 38.86 | 108.9 |
| 5.0 | 15 | 0.40 | 60.29 | 245.0 |
| 5.0 | 30 | 0.98 | 0.01111 | 0.01134 |
| 5.0 | 30 | 0.96 | 0.04535 | 0.04727 |
| 5.0 | 30 | 0.90 | 0.3015 | 0.3361 |
| 5.0 | 30 | 0.80 | 1.342 | 1.702 |
| 5.0 | 30 | 0.70 | 3.384 | 5.001 |
| 5.0 | 30 | 0.60 | 6.811 | 12.10 |
| 5.0 | 30 | 0.50 | 12.24 | 27.23 |
| 5.0 | 30 | 0.40 | 20.80 | 61.26 |
| 5.0 | 50 | 0.98 | 0.004000 | 0.004082 |
| 5.0 | 50 | 0.96 | 0.01633 | 0.01702 |
| 5.0 | 50 | 0.90 | 0.1087 | 0.1210 |
| 5.0 | 50 | 0.80 | 0.4871 | 0.6126 |
| 5.0 | 50 | 0.70 | 1.242 | 1.800 |
| 5.0 | 50 | 0.60 | 2.541 | 4.356 |
| 5.0 | 50 | 0.50 | 4.669 | 9.801 |
| 5.0 | 50 | 0.40 | 8.166 | 22.05 |
