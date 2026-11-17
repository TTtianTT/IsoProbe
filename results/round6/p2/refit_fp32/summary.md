# Round 6 phase 2 T5: reduced-head refit (gamma, beta, W_U) on frozen bodies (fp32 body, precision check)

| model | rev | arm | seed | complete | extended | loss orig. head | loss refit (step) | σ₁²/Σσ² start → end | σ₁/σ₂ end | |cos(ū, v₁)| end |
|---|---|---|---|---|---|---|---|---|---|---|
| pythia-70m | step50000 | random | 0 | True | False | 4.3521 | 3.7109 (10000) | 0.002 → 0.301 | 4.73 | 0.225 |
| pythia-70m | step143000 | random | 0 | True | False | 4.4011 | 3.8722 (10000) | 0.002 → 0.179 | 1.91 | 0.621 |
| pythia-160m | step80000 | random | 0 | True | False | 3.7683 | 3.3046 (10000) | 0.002 → 0.253 | 4.96 | 0.246 |
| pythia-160m | step143000 | random | 0 | True | False | 3.8558 | 3.4251 (10000) | 0.002 → 0.161 | 2.12 | 0.730 |
| pythia-410m | step80000 | random | 0 | True | False | 3.3929 | 3.0036 (10000) | 0.001 → 0.222 | 5.00 | 0.246 |
| pythia-410m | step143000 | random | 0 | True | False | 3.3379 | 2.9751 (10000) | 0.001 → 0.205 | 4.63 | 0.367 |

| model | pre → post | D_raw | D_refit (random head) | per-seed D_refit | D_refit / D_raw | reading |
|---|---|---|---|---|---|---|
| pythia-70m | step50000 → step143000 | +0.0490 | +0.1613 | +0.1613 | 3.29 | body side |
| pythia-160m | step80000 → step143000 | +0.0874 | +0.1206 | +0.1206 | 1.38 | body side |
| pythia-410m | step80000 → step143000 | -0.0549 | -0.0285 | -0.0285 | – | not interpreted (D_raw ≤ 0) |

- last-interval eval drop (10000 steps: 9000 → 10000; extended: 19000 → 20000): 160m 143000 random s0 +0.0012; 160m 80000 random s0 +0.0013; 410m 143000 random s0 +0.0019; 410m 80000 random s0 +0.0017; 70m 143000 random s0 +0.0013; 70m 50000 random s0 +0.0011

- **P7-6**: not evaluable yet (runs incomplete)
