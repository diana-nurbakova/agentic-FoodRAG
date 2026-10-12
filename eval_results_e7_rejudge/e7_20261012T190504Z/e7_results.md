# E-7 — follow-up re-judging of the 421 empty-plan items (`e7_20261012T190504Z`)

One session; Judges A and B; frozen templates; pipeline parser. Means with bootstrap 95% CIs over items (10,000 resamples, seed 0). Descriptive only; no significance claims across judges.

## correctness

| Judge | C1 q+orig | C2 q+msg | C3 −q+orig | C4 −q+msg |
|---|---|---|---|---|
| judgeA | 0.053 [0.034, 0.074] (NaN 3, stray 0) | 0.588 [0.540, 0.636] (NaN 1, stray 0) | 0.060 [0.038, 0.083] (NaN 1, stray 0) | 0.962 [0.943, 0.979] (NaN 0, stray 0) |
| judgeB | 0.160 [0.138, 0.182] (NaN 0, stray 0) | 0.159 [0.125, 0.194] (NaN 0, stray 0) | 0.056 [0.040, 0.074] (NaN 0, stray 0) | 0.109 [0.081, 0.140] (NaN 0, stray 0) |

| Judge | Question effect C2−C4 | Answer effect C2−C1 | C4−C3 | Interaction | Drift C1 / C4 (today − stored) |
|---|---|---|---|---|---|
| judgeA | -0.374 [-0.426, -0.321] | 0.537 [0.486, 0.587] | 0.902 [0.873, 0.930] | -0.366 [-0.419, -0.314] | +0.008 / +0.019 |
| judgeB | 0.049 [0.005, 0.094] | -0.001 [-0.042, 0.040] | 0.053 [0.020, 0.089] | -0.055 [-0.102, -0.005] | -0.004 / -0.007 |

## relevancy

| Judge | C1 q+orig | C2 q+msg | C3 −q+orig | C4 −q+msg |
|---|---|---|---|---|
| judgeA | 0.067 [0.043, 0.090] (NaN 0, stray 0) | 1.000 [1.000, 1.000] (NaN 0, stray 0) | 0.074 [0.050, 0.100] (NaN 0, stray 0) | 1.000 [1.000, 1.000] (NaN 0, stray 0) |
| judgeB | 0.036 [0.019, 0.055] (NaN 0, stray 0) | 1.000 [1.000, 1.000] (NaN 0, stray 0) | 0.026 [0.012, 0.043] (NaN 0, stray 0) | 1.000 [1.000, 1.000] (NaN 0, stray 0) |

| Judge | Question effect C2−C4 | Answer effect C2−C1 | C4−C3 | Interaction | Drift C1 / C4 (today − stored) |
|---|---|---|---|---|---|
| judgeA | 0.000 [0.000, 0.000] | 0.933 [0.910, 0.957] | 0.926 [0.900, 0.950] | 0.007 [-0.007, 0.021] | +0.007 / +0.000 |
| judgeB | 0.000 [0.000, 0.000] | 0.964 [0.945, 0.981] | 0.974 [0.957, 0.988] | -0.010 [-0.019, -0.002] | +0.000 / +0.000 |
