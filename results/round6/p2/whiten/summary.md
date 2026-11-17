# Round 6 phase 2 T3: whitening invariance and STS-B (12/12 models)

| model | whiten range (7 reps) | zscore range (3 G_lin) | zscore range (V1,V3,D) | raw | zscore | zscore (old fit) | whiten | abtt-10 | whiten − zscore [95% CI] | dropped eig / cond kept (V1) |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2 | 0.00e+00 | 2.48 | 0.00e+00 | 42.6 | 47.5 | 47.3 | 50.1 | 47.5 | +2.6 [+1.5, +3.7] | 1 / 1.53e+03 |
| gpt2-medium | 0.00e+00 | 2.50 | 0.00e+00 | 38.0 | 50.6 | 50.6 | 53.3 | 51.1 | +2.7 [+1.5, +4.0] | 1 / 7.51e+03 |
| gpt2-large | 0.00e+00 | 0.65 | 0.00e+00 | 43.9 | 51.8 | 51.7 | 54.4 | 52.0 | +2.6 [+1.6, +3.7] | 1 / 2.30e+03 |
| gpt2-xl | 0.00e+00 | 0.97 | 0.00e+00 | 42.8 | 52.1 | 52.1 | 54.7 | 52.4 | +2.6 [+1.4, +3.7] | 1 / 2.57e+03 |
| Qwen2.5-7B-Instruct | 0.00e+00 | 0.67 | 0.00e+00 | 43.6 | 52.2 | 52.1 | 52.1 | 50.8 | -0.0 [-1.4, +1.3] | 0 / 5.28e+04 |
| Llama-3.1-8B | 0.00e+00 | 0.71 | 0.00e+00 | 43.6 | 50.0 | 50.0 | 55.0 | 48.6 | +5.1 [+3.6, +6.6] | 0 / 4.54e+04 |
| Llama-3.1-8B-Instruct | 0.00e+00 | 0.48 | 0.00e+00 | 45.4 | 52.9 | 52.9 | 57.0 | 51.3 | +4.1 [+2.8, +5.5] | 0 / 3.88e+04 |
| Qwen3-8B-Base | 1.96e-02 | 1.03 | 0.00e+00 | 34.7 | 44.4 | 44.3 | 48.8 | 47.5 | +4.4 [+2.8, +6.0] | 0 / 1.00e+05 |
| Qwen3-8B | 7.96e-03 | 0.75 | 0.00e+00 | 35.4 | 49.2 | 49.1 | 51.9 | 50.2 | +2.7 [+1.3, +3.9] | 0 / 7.77e+04 |
| Mistral-Small-24B-Instruct-2501 | 0.00e+00 | 0.43 | 0.00e+00 | 46.1 | 51.9 | 51.9 | 54.3 | 50.2 | +2.4 [+1.2, +3.8] | 0 / 5.17e+04 |
| Olmo-3.1-32B-Instruct | 0.00e+00 | 0.27 | 0.00e+00 | 45.9 | 56.7 | 56.8 | 57.4 | 54.4 | +0.7 [-0.2, +1.7] | 0 / 1.73e+04 |
| Qwen3-32B | 0.00e+00 | 0.33 | 0.00e+00 | 40.3 | 48.6 | 48.4 | 51.7 | 48.0 | +3.2 [+1.6, +4.8] | 0 / 7.58e+04 |

- **P7-3**: max whitened-STS range over 7 representations = 0.0196 -> FAIL (< 1e-6)
- **P7-4**: models with zscore range over 3 G_lin samples > 0.5: 8/12 -> FAIL (≥ 10)
- zscore range over V1/V3/D_Amax/D_Amin (new fit / old fit): max 0.00e+00 / 0.00e+00
- V1 whiten − zscore: 95% CI above 0 in 10/12 models, includes 0 in 2, below 0 in 0; delta range -0.03 – +5.09
- V1 zscore (new fit) − zscore (old fit): range -0.05 – +0.17; CI excludes 0 in 2/12
- models with zscore G_lin range ≤ 0.5: Llama-3.1-8B-Instruct 0.48, Mistral-Small-24B-Instruct-2501 0.43, Olmo-3.1-32B-Instruct 0.27, Qwen3-32B 0.33
- dropped eigenvalues per representation (V1, V3, D_Amax, D_Amin, lin0-2): gpt2 [1, 1, 1, 1, 1, 1, 1]; gpt2-medium [1, 1, 1, 1, 1, 1, 1]; gpt2-large [1, 1, 1, 1, 1, 1, 1]; gpt2-xl [1, 1, 1, 1, 1, 1, 1]; Qwen2.5-7B-Instruct [0, 0, 0, 0, 0, 0, 0]; Llama-3.1-8B [0, 0, 0, 0, 0, 0, 0]; Llama-3.1-8B-Instruct [0, 0, 0, 0, 0, 0, 0]; Qwen3-8B-Base [0, 1, 0, 0, 0, 0, 0]; Qwen3-8B [0, 1, 0, 0, 0, 0, 0]; Mistral-Small-24B-Instruct-2501 [0, 0, 0, 0, 0, 0, 0]; Olmo-3.1-32B-Instruct [0, 0, 0, 0, 0, 0, 0]; Qwen3-32B [0, 0, 0, 0, 0, 0, 0]
- max condition number of the kept spectrum over the 7 representations: gpt2 3.4e+09; gpt2-medium 9.8e+06; gpt2-large 2.5e+06; gpt2-xl 7.2e+06; Qwen2.5-7B-Instruct 1.6e+08; Llama-3.1-8B 3.0e+07; Llama-3.1-8B-Instruct 3.3e+07; Qwen3-8B-Base 2.9e+10; Qwen3-8B 6.4e+10; Mistral-Small-24B-Instruct-2501 3.6e+07; Olmo-3.1-32B-Instruct 4.8e+09; Qwen3-32B 2.2e+08
