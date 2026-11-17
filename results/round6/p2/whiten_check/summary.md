# Round 6 phase 2 T3 numerical check (prereg T3 amendment): eigh vs SVD vs SVD with numerical-rank tolerance

| model | route | 7-rep range | dropped (V1, V3, D_Amax, D_Amin, lin0-2) | max |cos − cos_V1| |
|---|---|---|---|---|
| Llama-3.1-8B | eigh | 0.00e+00 | [0, 0, 0, 0, 0, 0, 0] | 1.46e-11 |
| Llama-3.1-8B | svd | 0.00e+00 | [0, 0, 0, 0, 0, 0, 0] | 3.67e-14 |
| Llama-3.1-8B | svd_ranktol | 0.00e+00 | [0, 0, 0, 0, 0, 0, 0] | 3.67e-14 |
| Qwen3-8B-Base | eigh | 1.96e-02 | [0, 1, 0, 0, 0, 0, 0] | 4.54e-03 |
| Qwen3-8B-Base | svd | 1.96e-02 | [0, 1, 0, 0, 0, 0, 0] | 4.54e-03 |
| Qwen3-8B-Base | svd_ranktol | 0.00e+00 | [0, 0, 0, 0, 0, 0, 0] | 3.30e-14 |
| Qwen3-8B | eigh | 7.96e-03 | [0, 1, 0, 0, 0, 0, 0] | 1.08e-02 |
| Qwen3-8B | svd | 7.96e-03 | [0, 1, 0, 0, 0, 0, 0] | 1.08e-02 |
| Qwen3-8B | svd_ranktol | 0.00e+00 | [0, 0, 0, 0, 0, 0, 0] | 3.53e-14 |
| gpt2 | eigh | 0.00e+00 | [1, 1, 1, 1, 1, 1, 1] | 1.70e-08 |
| gpt2 | svd | 0.00e+00 | [1, 1, 1, 1, 1, 1, 1] | 1.70e-08 |
| gpt2 | svd_ranktol | 0.00e+00 | [0, 0, 0, 0, 0, 0, 0] | 1.09e-08 |

- route (c) svd_ranktol range < 1e-6 in 4/4 checked models: ['Llama-3.1-8B', 'Qwen3-8B-Base', 'Qwen3-8B', 'gpt2']
- reading (prereg amendment): (c) < 1e-6 for Qwen3-8B and Qwen3-8B-Base -> numerical/threshold issue, stop condition not triggered; otherwise stop condition
