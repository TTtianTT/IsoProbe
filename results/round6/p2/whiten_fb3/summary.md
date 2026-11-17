# Round 6 phase 2 fallback 3: T3 on SICK-R (mean) and STS-B (PromptEOL), descriptive


## sickr_mean (12/12 models)

| model | whiten range (7 reps, eigh 1e-12) | whiten range (svd rank-tol) | zscore range (3 G_lin) | raw | zscore | whiten | whiten − zscore [95% CI] |
|---|---|---|---|---|---|---|---|
| gpt2 | 0.00e+00 | 0.00e+00 | 0.98 | 52.4 | 53.4 | 55.6 | +2.2 [+1.8, +2.5] |
| gpt2-medium | 0.00e+00 | 0.00e+00 | 0.89 | 51.5 | 54.6 | 56.5 | +1.8 [+1.5, +2.2] |
| gpt2-large | 6.34e-06 | 0.00e+00 | 0.31 | 53.7 | 55.1 | 57.1 | +2.0 [+1.7, +2.3] |
| gpt2-xl | 1.23e-06 | 0.00e+00 | 0.19 | 53.6 | 55.5 | 57.4 | +1.8 [+1.5, +2.1] |
| Qwen2.5-7B-Instruct | 0.00e+00 | 0.00e+00 | 0.60 | 46.2 | 52.8 | 53.7 | +0.9 [+0.6, +1.3] |
| Llama-3.1-8B | 0.00e+00 | 0.00e+00 | 0.60 | 49.9 | 52.0 | 56.7 | +4.7 [+4.2, +5.2] |
| Llama-3.1-8B-Instruct | 0.00e+00 | 0.00e+00 | 0.55 | 51.2 | 54.3 | 58.7 | +4.4 [+4.0, +4.9] |
| Qwen3-8B-Base | 8.36e-03 | 0.00e+00 | 1.52 | 37.4 | 48.5 | 51.1 | +2.6 [+2.2, +3.0] |
| Qwen3-8B | 3.51e-03 | 0.00e+00 | 0.60 | 41.1 | 53.6 | 54.5 | +0.9 [+0.7, +1.2] |
| Mistral-Small-24B-Instruct-2501 | 0.00e+00 | 0.00e+00 | 0.29 | 52.6 | 54.5 | 56.7 | +2.3 [+1.9, +2.7] |
| Olmo-3.1-32B-Instruct | 0.00e+00 | 0.00e+00 | 0.16 | 53.5 | 55.2 | 56.7 | +1.5 [+1.3, +1.8] |
| Qwen3-32B | 0.00e+00 | 0.00e+00 | 0.51 | 40.1 | 48.0 | 52.0 | +4.0 [+3.4, +4.5] |

- whitened-score range over 7 representations: eigh path max 0.00836 (> 0 in 4/12); svd rank-tol path max 0 (> 0 in 0/12)
- zscore range over 3 G_lin samples > 0.5 in 8/12 (range 0.16 – 1.52)
- V1 whiten − zscore: CI above 0 in 12/12, below 0 in 0, includes 0 in 0; delta range +0.93 – +4.67
- dropped eigenvalues (eigh path) per representation: gpt2 [1, 1, 1, 1, 1, 1, 1]; gpt2-medium [1, 1, 1, 1, 1, 1, 1]; gpt2-large [1, 1, 1, 1, 1, 1, 1]; gpt2-xl [1, 1, 1, 1, 1, 1, 1]; Qwen3-8B-Base [0, 1, 0, 0, 0, 0, 0]; Qwen3-8B [0, 1, 0, 0, 0, 0, 0]

## stsb_eol (12/12 models)

| model | whiten range (7 reps, eigh 1e-12) | whiten range (svd rank-tol) | zscore range (3 G_lin) | raw | zscore | whiten | whiten − zscore [95% CI] |
|---|---|---|---|---|---|---|---|
| gpt2 | 0.00e+00 | 0.00e+00 | 0.98 | 67.3 | 71.2 | 65.9 | -5.4 [-6.5, -4.2] |
| gpt2-medium | 0.00e+00 | 0.00e+00 | 1.41 | 56.2 | 63.3 | 64.7 | +1.4 [+0.1, +2.8] |
| gpt2-large | 0.00e+00 | 0.00e+00 | 0.44 | 69.2 | 72.1 | 71.6 | -0.5 [-1.4, +0.3] |
| gpt2-xl | 0.00e+00 | 0.00e+00 | 1.51 | 65.1 | 69.2 | 71.2 | +2.0 [+0.9, +2.9] |
| Qwen2.5-7B-Instruct | 0.00e+00 | 0.00e+00 | 0.20 | 75.1 | 75.1 | 75.1 | +0.0 [-0.8, +0.8] |
| Llama-3.1-8B | 0.00e+00 | 0.00e+00 | 0.18 | 70.9 | 72.3 | 73.9 | +1.6 [+0.8, +2.5] |
| Llama-3.1-8B-Instruct | 0.00e+00 | 0.00e+00 | 0.85 | 73.4 | 74.3 | 75.5 | +1.1 [+0.3, +2.0] |
| Qwen3-8B-Base | 7.73e-04 | 0.00e+00 | 0.37 | 72.1 | 70.4 | 70.4 | -0.0 [-0.9, +0.9] |
| Qwen3-8B | 3.88e-03 | 0.00e+00 | 0.75 | 72.7 | 71.8 | 72.8 | +0.9 [-0.0, +1.9] |
| Mistral-Small-24B-Instruct-2501 | 0.00e+00 | 0.00e+00 | 0.20 | 73.3 | 73.5 | 73.6 | +0.2 [-0.5, +0.9] |
| Olmo-3.1-32B-Instruct | 0.00e+00 | 0.00e+00 | 0.72 | 68.6 | 67.7 | 67.8 | +0.2 [-0.4, +0.9] |
| Qwen3-32B | 0.00e+00 | 0.00e+00 | 0.49 | 71.5 | 73.0 | 72.7 | -0.4 [-1.2, +0.3] |

- whitened-score range over 7 representations: eigh path max 0.00388 (> 0 in 2/12); svd rank-tol path max 0 (> 0 in 0/12)
- zscore range over 3 G_lin samples > 0.5 in 6/12 (range 0.18 – 1.51)
- V1 whiten − zscore: CI above 0 in 4/12, below 0 in 1, includes 0 in 7; delta range -5.37 – +1.99
- dropped eigenvalues (eigh path) per representation: gpt2 [1, 1, 1, 1, 1, 1, 1]; gpt2-medium [1, 1, 1, 1, 1, 1, 1]; gpt2-large [1, 1, 1, 1, 1, 1, 1]; gpt2-xl [1, 1, 1, 1, 1, 1, 1]; Qwen3-8B-Base [0, 1, 0, 0, 0, 0, 0]; Qwen3-8B [0, 1, 0, 0, 0, 0, 0]
