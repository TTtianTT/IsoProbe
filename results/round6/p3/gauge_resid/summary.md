# U1 residual-stream gauge check (fp64)

| case | arch | transform | expected invariant | logits max rel err | argmax agree | max abs diff A / Bc / RankMe (judged) | raw-x diff A / RankMe (LN only) |
|---|---|---|---|---|---|---|---|
| gpt2_haar_ctrl | ln | haar | False | 6.88e-01 | 0.92364 | 5.1e-03 / 7.4e-05 / 2.0e+00 | 0.00489 / 2.04 |
| gpt2_q0 | ln | keep1 | True | 2.35e-12 | 1.00000 | 2.2e-14 / 5.2e-16 / 3.2e-12 | 2.21e-14 / 2.84e-12 |
| gpt2_q1 | ln | keep1 | True | 2.79e-12 | 1.00000 | 1.9e-14 / 8.3e-16 / 6.3e-12 | 1.92e-14 / 6.25e-12 |
| gpt2_q2 | ln | keep1 | True | 3.69e-12 | 1.00000 | 2.1e-14 / 2.2e-16 / 5.7e-12 | 2.08e-14 / 5.68e-12 |
| gpt2_shear | ln | shear | True | 2.68e-13 | 1.00000 | 4.0e-15 / 7.0e-17 / 3.4e-12 | 0.286 / 79.9 |
| olmo2-1b_haar_ctrl | post | haar_post | False | 1.52e+00 | 0.00611 | 3.3e-01 / 1.6e-03 / 1.6e+02 | – |
| olmo2-1b_perm0 | post | perm | True | 2.31e-14 | 1.00000 | 1.1e-16 / 1.1e-17 / 3.0e-12 | – |
| olmo2-1b_perm1 | post | perm | True | 2.34e-14 | 1.00000 | 1.1e-16 / 3.3e-17 / 3.2e-12 | – |
| olmo2-1b_perm2 | post | perm | True | 1.80e-14 | 1.00000 | 1.1e-16 / 8.0e-18 / 3.2e-12 | – |
| qwen2.5-7b_q0 | rms | haar | True | 4.18e-13 | 1.00000 | 4.4e-16 / 1.0e-15 / 5.0e-12 | – |

P8-3 (logits rel err < 1e-6 on all 8 in-group cases): pass
P8-4 (per-layer metric diffs < 1e-6): pass
