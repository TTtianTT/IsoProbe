# Round 6 phase 2 T1: collapse in the gauge-invariant geometry h*

| model | from → to | top1 V1 | top1 V3 | top1 h* (uniform) | top1 h* (unigram) | A V1 | A V3 | A h* (uniform) | A h* (unigram) | ‖ū‖² h*u |
|---|---|---|---|---|---|---|---|---|---|---|
| pythia-70m | 50000 → 143000 | 0.119 → 0.041 | 0.152 → 0.040 | 0.187 → 0.303 | 0.172 → 0.185 | 0.271 → 0.030 | 0.487 → 0.011 | 0.402 → 0.002 | 0.275 → 0.013 | 0.484 → 0.998 |
| pythia-160m | 80000 → 143000 | 0.177 → 0.060 | 0.067 → 0.058 | 0.172 → 0.545 | 0.170 → 0.278 | 0.412 → 0.037 | 0.544 → 0.018 | 0.425 → 0.002 | 0.283 → 0.010 | 0.468 → 0.998 |
| pythia-410m | 80000 → 143000 | 0.097 → 0.081 | 0.078 → 0.073 | 0.118 → 0.116 | 0.170 → 0.170 | 0.453 → 0.522 | 0.660 → 0.555 | 0.476 → 0.469 | 0.298 → 0.298 | 0.409 → 0.411 |
| pythia-1.4b | 80000 → 143000 | 0.090 → 0.086 | 0.083 → 0.084 | 0.114 → 0.108 | 0.176 → 0.174 | 0.442 → 0.536 | 0.645 → 0.591 | 0.481 → 0.491 | 0.289 → 0.295 | 0.405 → 0.389 |

- **P7-1** (uniform h*): pythia-70m Δtop1 = +0.115 (not ok); pythia-160m Δtop1 = +0.373 (ok); pythia-410m Δtop1 = -0.002 (ok); pythia-1.4b Δtop1 = -0.006 (ok) -> FAIL
- unigram h*_π Δtop1 (report only): pythia-70m +0.013; pythia-160m +0.108; pythia-410m +0.000; pythia-1.4b -0.003

## Seed variation at step 143000 (original seed + 9 PolyPythias)

| size | n | A V1 mean ± sd (CV) | A V3 mean ± sd (CV) | A h* (uniform) mean ± sd (CV) | A h* (unigram) mean ± sd (CV) | top1 h*u mean ± sd |
|---|---|---|---|---|---|---|
| pythia-70m | 10 | 0.041 ± 0.018 (0.45) | 0.014 ± 0.006 (0.43) | 0.013 ± 0.035 (2.68) | 0.054 ± 0.083 (1.53) | 0.322 ± 0.072 |
| pythia-160m | 10 | 0.126 ± 0.064 (0.51) | 0.079 ± 0.050 (0.63) | 0.060 ± 0.073 (1.21) | 0.156 ± 0.098 (0.63) | 0.327 ± 0.094 |

## Trajectories (min over training / final), A

| model | V1 min (step) / final | V3 min (step) / final | h* (uniform) min (step) / final | h* (unigram) min (step) / final |
|---|---|---|---|---|
| pythia-70m | 0.030 (step143000) / 0.030 | 0.011 (step143000) / 0.011 | 0.002 (step143000) / 0.002 | 0.013 (step143000) / 0.013 |
| pythia-160m | 0.037 (step143000) / 0.037 | 0.018 (step143000) / 0.018 | 0.002 (step143000) / 0.002 | 0.010 (step143000) / 0.010 |
| pythia-410m | 0.402 (step50000) / 0.522 | 0.503 (step118000) / 0.555 | 0.454 (step5000) / 0.469 | 0.291 (step13000) / 0.298 |
| pythia-1.4b | 0.385 (step50000) / 0.536 | 0.586 (step130000) / 0.591 | 0.407 (step5000) / 0.491 | 0.282 (step5000) / 0.295 |
| OLMo-2-0425-1B | 0.410 (stage1-step0-tokens0B) / 0.593 | 0.410 (stage1-step0-tokens0B) / 0.596 | 0.169 (stage1-step300-tokens1B) / 0.351 | 0.177 (stage1-step300-tokens1B) / 0.266 |

(min taken over checkpoints with step ≥ 1000 for Pythia, i.e. excluding the early transient)
