# Round 6 phase 2 T5: reduced-head refit (gamma, beta, W_U) on frozen bodies (bf16 body)

| model | rev | arm | seed | complete | extended | loss orig. head | loss refit (step) | σ₁²/Σσ² start → end | σ₁/σ₂ end | |cos(ū, v₁)| end |
|---|---|---|---|---|---|---|---|---|---|---|
| pythia-70m | step50000 | random | 0 | True | False | 4.3681 | 3.7241 (10000) | 0.002 → 0.301 | 4.75 | 0.225 |
| pythia-70m | step50000 | random | 1 | True | False | 4.3681 | 3.7242 (10000) | 0.002 → 0.301 | 4.76 | 0.226 |
| pythia-70m | step50000 | random | 2 | True | False | 4.3681 | 3.7255 (10000) | 0.002 → 0.305 | 4.78 | 0.223 |
| pythia-70m | step110000 | random | 0 | True | False | 4.5057 | 3.8999 (10000) | 0.002 → 0.198 | 2.37 | 0.609 |
| pythia-70m | step110000 | random | 1 | True | False | 4.5057 | 3.9016 (10000) | 0.002 → 0.199 | 2.37 | 0.607 |
| pythia-70m | step110000 | random | 2 | True | False | 4.5057 | 3.9014 (10000) | 0.002 → 0.200 | 2.39 | 0.609 |
| pythia-70m | step143000 | own | 0 | True | True | 4.5355 | 4.2023 (20000) | 0.937 → 0.967 | 20.96 | 0.996 |
| pythia-70m | step143000 | random | 0 | True | False | 4.5355 | 3.9664 (10000) | 0.002 → 0.181 | 1.89 | 0.614 |
| pythia-70m | step143000 | random | 1 | True | False | 4.5355 | 3.9678 (10000) | 0.002 → 0.181 | 1.90 | 0.613 |
| pythia-70m | step143000 | random | 2 | True | False | 4.5355 | 3.9672 (10000) | 0.002 → 0.182 | 1.91 | 0.615 |
| pythia-70m | step143000 | trunc | 0 | True | True | 4.5355 | 4.3189 (20000) | 0.033 → 0.877 | 10.67 | 0.985 |
| pythia-160m | step80000 | random | 0 | True | False | 3.7780 | 3.3137 (10000) | 0.002 → 0.253 | 4.95 | 0.247 |
| pythia-160m | step80000 | random | 1 | True | False | 3.7780 | 3.3138 (10000) | 0.002 → 0.253 | 4.96 | 0.247 |
| pythia-160m | step80000 | random | 2 | True | False | 3.7780 | 3.3147 (10000) | 0.002 → 0.255 | 5.00 | 0.246 |
| pythia-160m | step143000 | random | 0 | True | False | 3.8967 | 3.4614 (10000) | 0.002 → 0.162 | 2.10 | 0.725 |
| pythia-160m | step143000 | random | 1 | True | False | 3.8967 | 3.4616 (10000) | 0.002 → 0.161 | 2.09 | 0.729 |
| pythia-160m | step143000 | random | 2 | True | False | 3.8967 | 3.4626 (10000) | 0.002 → 0.162 | 2.09 | 0.729 |
| pythia-410m | step80000 | random | 0 | True | False | 3.3940 | 3.0042 (10000) | 0.001 → 0.222 | 5.00 | 0.245 |
| pythia-410m | step80000 | random | 1 | True | False | 3.3940 | 3.0038 (10000) | 0.001 → 0.221 | 4.97 | 0.247 |
| pythia-410m | step80000 | random | 2 | True | False | 3.3940 | 3.0048 (10000) | 0.001 → 0.222 | 4.94 | 0.246 |
| pythia-410m | step143000 | random | 0 | True | False | 3.3652 | 2.9926 (10000) | 0.001 → 0.202 | 4.62 | 0.372 |
| pythia-410m | step143000 | random | 1 | True | False | 3.3652 | 2.9915 (10000) | 0.001 → 0.204 | 4.61 | 0.371 |
| pythia-410m | step143000 | random | 2 | True | False | 3.3652 | 2.9929 (10000) | 0.001 → 0.203 | 4.62 | 0.372 |

| model | pre → post | D_raw | D_refit (random head) | per-seed D_refit | D_refit / D_raw | reading |
|---|---|---|---|---|---|---|
| pythia-70m | step50000 → step143000 | +0.1674 | +0.2426 | +0.2423, +0.2437, +0.2417 | 1.45 | body side |
| pythia-160m | step80000 → step143000 | +0.1188 | +0.1478 | +0.1478, +0.1478, +0.1479 | 1.24 | body side |
| pythia-410m | step80000 → step143000 | -0.0289 | -0.0119 | -0.0116, -0.0123, -0.0119 | – | not interpreted (D_raw ≤ 0) |

- last-interval eval drop (10000 steps: 9000 → 10000; extended: 19000 → 20000): 160m 143000 random s0 +0.0012; 160m 143000 random s1 +0.0016; 160m 143000 random s2 +0.0013; 160m 80000 random s0 +0.0013; 160m 80000 random s1 +0.0016; 160m 80000 random s2 +0.0013; 410m 143000 random s0 +0.0018; 410m 143000 random s1 +0.0023; 410m 143000 random s2 +0.0017; 410m 80000 random s0 +0.0017; 410m 80000 random s1 +0.0020; 410m 80000 random s2 +0.0014; 70m 110000 random s0 +0.0010; 70m 110000 random s1 +0.0012; 70m 110000 random s2 +0.0008; 70m 143000 own s0 +0.0047; 70m 143000 random s0 +0.0013; 70m 143000 random s1 +0.0013; 70m 143000 random s2 +0.0010; 70m 143000 trunc s0 +0.0053; 70m 50000 random s0 +0.0011; 70m 50000 random s1 +0.0013; 70m 50000 random s2 +0.0008

- **P7-6**: 70m step143000 random-init refit σ₁²/Σσ² = 0.181, 0.181, 0.182 -> PASS (all < 0.5)
