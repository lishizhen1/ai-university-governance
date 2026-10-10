# Independent methods and result audit

The frozen additional-analysis source and aggregate outputs were independently reviewed on 2026-10-10. No original analysis file or patient record was changed during this review.

## Calibration error

The implementation computes admission-weighted absolute calibration deviations at all observed predictions. Four-quantile-knot logistic B-spline smoothing is re-estimated in each bootstrap sample. eICU resamples whole people and carries all included admissions; identical draws pair models. Independent refits reproduced all 26 point estimates and the first bootstrap draw for each model/cohort to less than 1e-12. All percentile interval endpoints and 72 paired metric contrasts were reconstructed from 500 replicates. The existing 44 numeric/model artifacts retained their frozen hashes. These conditional intervals do not include repeated fitting of the risk-model development pipeline.

## Formal latent-class sensitivity

The posterior-column ordering, neutral labels, APP, normalized entropy, likelihood-repeat counts and fixed-model external allocation were checked against source logic and lcmm documentation. Each analysis compares candidate models on its own unchanged population; no cross-population BIC comparison is made.

All 55 source/model files in the original private freeze matched their hashes. Aggregate checks independently reconstructed AIC/BIC, best-likelihood repetition, class-assignment totals, posterior expected totals, weighted APP and posterior probability thresholds. Each analysis contains 139 initialization diagnostics, four selected models, eight classification-quality records and 20 class-summary records. The final report and tables agree. Eight selected models have conv=1 and convergence criteria below 1e-5; their saved-model checks reproduce likelihoods and posterior probabilities with zero difference. All 36 full restarts were reloaded and agreed with their recorded likelihoods to below 4.8e-11.

Both original fit processes saved completed checkpoints before a trailing source-stream parsing error caused exit status 1. The execution record discloses this incident. The current R sources parse, and a subsequent independent reload and fixed-parameter likelihood/posterior verification completed successfully. No result was accepted solely on the interrupted processes' status, and no clean original exit is claimed.

K4 is the BIC minimum only within K1–K4. Small classes, moderate posterior certainty, the Gaussian approximation to discrete SIC, informative observation and only three time windows preclude a claim of four natural biological populations. eICU results are descriptive posterior allocations, not mortality prediction validation.

## Public-export boundary

An explicit file allowlist selected source, methods, environment and aggregate diagnostics. Patient-level data, source-person maps, risk predictions, individual class probabilities/assignments, fitted models/parameters, centers, fitted class trajectories, private paths, runtime binaries and the controlled-deposit preparation were excluded. The original private hash inventory is retained locally; only source provenance and aggregate validation records are distributed. Public-source edits concern path portability and historical status wording and do not change numerical algorithms or reference results.
