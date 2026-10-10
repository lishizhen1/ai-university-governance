# Integrated calibration error: additional descriptive analysis

The integrated calibration index (ICI) is the empirical mean of the absolute differences between each original risk prediction and the corresponding smoothed calibration probability. E50 and E90 are the median and 90th percentile of those same absolute differences. The calculation averages over admissions, preserving their empirical prediction distribution; it is not the unweighted area between curves on a uniform probability grid. This follows the metric definition of Austin and Steyerberg (2019), DOI: [10.1002/sim.8281](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8281).

For prediction p_i and estimated calibration function c_hat, the implemented definitions are:

- ICI = mean_i |c_hat(p_i) − p_i|.
- E50 = empirical 50th percentile of |c_hat(p_i) − p_i|.
- E90 = empirical 90th percentile of |c_hat(p_i) − p_i|.

All values are on the probability scale. Multiplication by 100 expresses them in percentage points. Lower values indicate smaller estimated calibration deviations under the stated smoother; no threshold for clinical adequacy is inferred.

## Fixed smoothing specification

The estimator was fixed before examining these new calibration-error results and exactly follows the flexible curves already used in Figure 2: cubic logistic B-spline regression on logit(predicted probability), with four empirical quantile knots, `include_bias=False`, an intercept, and L2 logistic regression with C=1, `solver='lbfgs'`, `tol=1e-4` and `max_iter=2000`. No comparison among smoothers, knot counts or penalty strengths was used to choose the reported estimator. It is a B-spline estimator, not the loess or restricted-cubic-spline implementations illustrated in the original paper.

For finite logit transformation only, probabilities are clipped to [10^-6, 1−10^-6]. Absolute deviations use the original, unclipped model probabilities. ICI/E50/E90 are evaluated at every observed prediction, including the tails. The earlier plot shows the curve only between the 1st and 99th prediction percentiles; that display restriction is not used to truncate the ICI calculation. Calibration probabilities remain between zero and one because the smoother uses a logistic link.

Numerical policies are specified in advance: constant predictions or a single observed outcome class produce a constant empirical-event-rate curve. If four empirical quantile knots coincide while predictions vary, the same degree and penalty are retained with four equally spaced knots over the observed logit range. Every occurrence is counted. Convergence failures cause an error rather than silently dropping bootstrap replicates. Input NaNs, nonbinary outcomes and probabilities outside [0,1] are rejected. No degenerate fallback is expected for the supplied large-cohort continuous predictions; actual diagnostic counts are reported in `ici_verification.json`.

## Conditional uncertainty and comparisons

The existing frozen outputs contain 10,810 held-out MIMIC-IV predictions and 4,355 eICU predictions. Original model fitting, tuning, imputation, clustering and internal-fold generation are not rerun. Five hundred bootstrap replicates use seed 20261010. MIMIC-IV resamples first-stay patients with replacement. eICU resamples its 4,220 unique people with replacement and includes all hospitalizations belonging to each selected person. Within each resample, quantile knots, spline basis and logistic calibration coefficients are estimated again, and the error metrics are calculated at the resampled predictions. The same resampling draw is shared across all 13 models, permitting paired differences.

Reported 95% confidence limits are the empirical 2.5th and 97.5th percentiles. E50/E90 and interval endpoints use NumPy's linear quantile interpolation. Each eICU admission has equal weight in the point estimate; person-level resampling addresses repeated-admission dependence without changing the estimand to an equally weighted mean across people. Paired differences are model A minus model B; negative ICI/E50/E90 differences favor model A. Comparisons are post hoc and unadjusted for multiplicity.

These intervals incorporate sampling and calibration-smoother estimation uncertainty conditional on the saved risk predictions. They do not include full model-development uncertainty, dependency from shared internal-fold training samples, or hospital-level sampling variation. The original paper's illustrative bootstrap refitted the prediction method using resampled development and validation samples; the present conditional procedure is deliberately identified as different. Finite-sample smoothing error and the choice of fixed regularization influence these descriptive estimates, particularly near perfect calibration. The calibration fit is an evaluation step and does not replace or update the reported risk predictions.

## Verification and files

`ici_verification.json` records hand-calculated metric tests, probability-boundary tests, constant and single-class cases, whole-person resampling checks, calibrator convergence, and pointwise agreement with the previously saved Figure 2 curves. The eight applicable model/cohort curve series must agree within 10^-6. Input-file hashes are checked against the frozen snapshot before calculation and again afterward.

- `ici_analysis.py`: executable calculation and boundary tests.
- `ici_analysis_plan.json`: estimator, bootstrap, contrasts and source fixed before result calculation.
- `ici_point_estimates.csv`: point estimates for all 13 models in both cohorts.
- `ici_e50_e90_with_95CI.csv`: estimates and conditional confidence intervals.
- `ici_e50_e90_paired_differences.csv`: paired differences with conditional intervals.
- `ici_bootstrap_metric_replicates.csv`: aggregate metric replicates, without participant identifiers or predictions.
- `ici_verification.json` and `ici_environment.json`: validation and execution metadata.

Run `python ici_analysis.py --bootstrap 500` from an authorized local analysis checkout after the existing private prediction files have been generated. The script reads those files but does not write participant-level data, calibration curves, fitted calibrator coefficients or model objects. It writes only to this `additional_revision` directory. Existing models and main result files remain unchanged.

Reference: Austin PC, Steyerberg EW (2019). The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. Statistics in Medicine, 38(21):4051–4065. https://doi.org/10.1002/sim.8281
