# Round 6 phase 2 T2: saturation split and the high-deviation tokens

| model | Δloss (mean) | q | k | share of Σd in H | share in complement [95% CI] | mean d in H / complement | class |
|---|---|---|---|---|---|---|---|
| pythia-70m | +0.1674 | 0.1% | 507 | 0.0007 | 0.9993 [0.9986, 0.9999] | +0.119 / +0.167 | diffuse |
| pythia-70m | +0.1674 | 0.5% | 2535 | 0.0028 | 0.9972 [0.9954, 0.9988] | +0.093 / +0.168 | diffuse |
| pythia-70m | +0.1674 | 1% | 5070 | 0.0048 | 0.9952 [0.9927, 0.9976] | +0.080 / +0.168 | diffuse |
| pythia-160m | +0.1188 | 0.1% | 507 | 0.0060 | 0.9940 [0.9894, 0.9979] | +0.710 / +0.118 | diffuse |
| pythia-160m | +0.1188 | 0.5% | 2535 | 0.0142 | 0.9858 [0.9799, 0.9907] | +0.338 / +0.118 | diffuse |
| pythia-160m | +0.1188 | 1% | 5070 | 0.0170 | 0.9830 [0.9770, 0.9880] | +0.202 / +0.118 | diffuse |
| pythia-410m | -0.0289 | 0.1% | 507 | 0.0022 | 0.9978 [0.9962, 0.9994] | -0.064 / -0.029 | diffuse |
| pythia-410m | -0.0289 | 0.5% | 2535 | 0.0173 | 0.9827 [0.9783, 0.9870] | -0.100 / -0.029 | diffuse |
| pythia-410m | -0.0289 | 1% | 5070 | 0.0381 | 0.9619 [0.9551, 0.9685] | -0.110 / -0.028 | diffuse |

| model | ρ(dev, −log f input) | ρ(dev, −log f target) | ρ(dev, entropy) | loss H / compl. | entropy H / compl. | incomplete-UTF-8 input: H / all | share of all such inputs in H |
|---|---|---|---|---|---|---|---|
| pythia-70m | +0.133 | -0.015 | -0.217 | 2.309 / 4.538 | 1.499 / 4.568 | 0.785 / 0.0014 | 0.556 |
| pythia-160m | +0.235 | -0.087 | -0.340 | 3.173 / 3.897 | 2.236 / 4.097 | 0.811 / 0.0014 | 0.574 |
| pythia-410m | +0.161 | +0.023 | -0.213 | 1.580 / 3.367 | 1.670 / 3.302 | 0.047 / 0.0014 | 0.034 |

- **P7-2**: ρ(dev, −log f input) = 70m +0.133, 160m +0.235 -> FAIL (both > 0.2 required)

## 2c t50 (report only)

- pythia-70m (window from 50000): kurtosis 113000, q999_over_median 83000, ubar_sq_V1 87000, s1sq_over_sum 93000
- pythia-160m (window from 80000): kurtosis 121000, q999_over_median 107000, ubar_sq_V1 96000, s1sq_over_sum 104000
- pythia-410m (window from 80000): kurtosis 90000, q999_over_median 100000, ubar_sq_V1 98000, s1sq_over_sum 100000
- pythia-1.4b (window from 80000): kurtosis 100000, q999_over_median 100000, ubar_sq_V1 100000, s1sq_over_sum 110000
