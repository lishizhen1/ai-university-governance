# Public-export scope

This methods record also describes artifacts generated during the authorized local analysis. File names below identify generated local outputs; they are not a claim that restricted models, fitted parameters, or all output tables are distributed on GitHub. PUBLIC_MANIFEST.json lists the actual public file set.

# Principal reanalysis methods retained from revision v1

This record describes the primary prediction and K-means analyses distributed in v1. Revision v2 additionally **completed** ICI/E50/E90 with conditional bootstrap intervals and formal Gaussian longitudinal latent-class models (K1-K4, observed-value and complete-three-window analyses). Their distinct specifications and completed results are documented in [additional_revision](../additional_revision/README.md). The formal latent-class analysis uses observed SIC without carry-forward and does not replace or validate the earlier multivariable K-means labels. Primary prediction outputs remain unchanged.

This is a post-review reanalysis, not a prospectively registered analysis. All model fitting, imputation parameters, scaling, K-means centroids and logistic regularization selection use MIMIC-IV only. The eICU cohort was examined in the original analysis; therefore the revised evaluation is an external re-evaluation, not a never-inspected external test set.

## Findings in the supplied implementation

- The supplied original code and processed data reproduce the original scope: 29,308 source MIMIC-IV stays, 10,983 old W1-eligible 72-hour stays, and 4,710 old eICU external stays. Original outputs were retained privately, unmodified; they are not included in this public repository.
- Original prediction code imputes from full-cohort medians before cross-validation. K-means imputation, standardization and clustering also precede all prediction folds. Thus the old internal results do not represent validation of the complete fitted pipeline.
- Original K=4 naming identifies the highest-mortality cluster using whole-cohort outcomes. Revised cluster labels are neutral and ordered by centroid mean SIC; mortality is an outcome used only after unsupervised assignment.
- Original “nested models” are nested predictor sets, not nested resampling. Revised regularized logistic models use 5 outer folds and 3 inner folds.
- The manuscript's 0–5/6–9/>=10 SIC organ-component mapping is a textual error. Both supplied derivation scripts map the four-system SOFA sum 0/1/>=2 to SIC organ points 0/1/2. The four systems are respiratory, hepatic, cardiovascular and renal.
- MIMIC used INR >=1.2 for one SIC point, whereas eICU used INR >1.2. Revised totals use <=1.2:0, >1.2 and <=1.4:1, >1.4:2 consistently. Revised positivity additionally requires coagulation points >=3 and total SIC >=4. The original MIMIC positivity flag omitted the coagulation requirement. No model eligibility depends on positivity.
- Original MIMIC eligibility uses ICU LOS >=3 days alone, leaving patients with recorded death/discharge before the landmark. The revised pipeline explicitly excludes incompatible timestamps and date-only deaths before the landmark calendar date.
- Original suspected-infection pairing could use an antibiotic prescribed after the landmark. Revised main MIMIC eligibility requires both timestamps to be available by ICU admission +72 hours.
- Original eICU text-sepsis scans diagnoses recorded at any time. Revised main eligibility restricts diagnostic records to observed numeric diagnosisoffset <=4,320 minutes, while retaining baseline APACHE admission diagnosis. Negative offsets are retained as pre-ICU documentation.
- Original eICU hospital mortality uses status ==Expired and consequently treats missing/unknown status as survival. Revised external evaluation requires a recorded hospital discharge status of Alive or Expired.
- No original frozen .joblib or .pkl objects were located. New frozen pipelines and their preprocessing parameters, coefficients and centroids are provided locally. The public code export is identified by the repository commit and PUBLIC_MANIFEST.json. Trained models and fitted parameters remain excluded from GitHub under the source-data sharing requirements. No preregistered protocol is claimed.

## Cohort and endpoint definitions

`cohort_audit.json` is the authoritative finalized flow. MIMIC uses the provided first-ICU-per-person adult extraction, suspected infection near ICU admission (first pair time from -48 to +24 hours) and W1 SOFA >=2 under a missing-normal convention. Baseline SOFA before infection was unavailable, so this is an operational EHR definition rather than an adjudicated acute SOFA increase. An antibiotic-first pair permits culture within 24 hours; a culture-first pair permits antibiotics within 72 hours. Both observed times must be <=the 72-hour landmark in the revision.

eICU includes first ICU units per hospital encounter (`unitvisitnumber==1`, missing unitvisitnumber treated as 1 in the supplied source extraction), not necessarily the first hospitalization per person. Repeated people remain in the external population and are resampled as person clusters. Exact text matching is case-insensitive regex `sepsis|septic` in `apacheadmissiondx`, OR in `diagnosisstring + ' ' + icd9code` for diagnosisoffset <=4320. Missing text is empty. Missing or nonnumeric diagnosisoffset does not qualify a diagnostic record. No code-based ICD expansion, negation detection or manual ambiguity adjudication is implied. A word-boundary sensitivity was checked and yields the same final eligible set. Admission-APACHE-only evaluation is also provided without refitting.

The prediction time origin is ICU admission +72 hours. Predictors end at that landmark. The primary outcome is death during the index hospitalization after the landmark, with variable follow-up to hospital discharge. It is neither mortality predicted at ICU admission nor a fixed 28-day post-landmark outcome. The historical MIMIC 28-day flag starts at ICU admission; only 25 days remain after the landmark. eICU does not provide comparable post-discharge all-cause follow-up in these files. ICU and hospital disposition flags are recorded independently; discordances are disclosed in `verification.json` and not silently reconciled.

## Missingness and predictor construction

W1=0–24h, W2=24–48h, W3=48–72h, using worst values within each half-open 24-hour interval. Observed platelet and INR are required in W1. Platelet SIC points are 0 for >=150, 1 for >=100 and <150, and 2 for <100 (10^9/L). Laboratory INR and platelet counts remain continuous in the coagulation-only model.

For SIC, selected laboratory and individual SOFA-component predictor series, missing W2 and W3 values are each replaced by that same person's W1 value when available. This is W1 carry-forward, not recursive last-observation carry-forward: missing W3 is filled from W1 even when W2 is observed. It can attenuate change and does not correct informative measurement. SIC mean/max/min and slope are recalculated after that carry-forward; slope=(W3-W1)/2, in SIC points per 24 hours. Remaining missing predictor values use training-fold medians, then training-fold z-score scaling. The corresponding frozen derivation parameters are applied externally. Missing indicators are constructed before carry-forward.

The source SOFA total and four-system sum use zero for missing components at score derivation. That convention is not equivalent to saying that individual hepatic/other SOFA measurements were observed as normal. Individual component predictors can subsequently be carried forward/imputed for expanded models, while the source total remains the missing-normal score. This distinction and the source respiratory window-level PaO2/FiO2 approximation limit physiological interpretation. eICU APACHE fallback is restricted to W1 by the source window key.

`missingness_before_carry_forward.csv` reports missingness in the supplied window features before the new carry-forward operation; eICU W1 values may already include the documented APACHE fallback. Missingness rates do not prove missing-at-random. The optional missing-indicator model, complete-window and single-iterative-imputation analyses, and observation-probability sensitivity must be labeled as sensitivities; none establishes noninformative observation.

MIMIC `anchor_age` is the supplied deidentified anchor-age variable, not a reconstruction from anchor year. eICU age '>89' is coded 90 in the supplied derivation. Sex is the supplied binary male indicator; any nonmale category would otherwise be 0 and is checked in the verification report. These source-variable limitations should not be hidden behind a claim of bedside-ready measurement.

## Fitting, evaluation and uncertainty

All logistic models use a pipeline with median imputation, scaling, optional K-means + one-hot assignment, and L2 logistic regression. Each outer training set selects C in {0.01,0.1,1,10} by 3-fold stratified inner negative log-loss. Preprocessing and centroids are refitted independently for each inner and outer training split. A 5-fold stratified outer split with seed 42 generates one held-out prediction per MIMIC stay. Full-data tuning uses MIMIC only to produce the external frozen model. K=2 is the primary clustering analysis; K=4 is explicitly exploratory. K=2..6 full-derivation silhouette scores describe cluster structure and do not select an outer-test predictor pipeline.

The expanded XGBoost comparison retains fixed historical hyperparameters: 200 estimators, max_depth 4, learning_rate 0.1, subsample 0.8, colsample_bytree 0.8, log-loss evaluation, seed 42. It has no new data-dependent hyperparameter search. Expanded LR and XGBoost share an identical feature set. `feature_sets.json` defines every comparison; the revised expanded set also includes all six SOFA components across windows. It is not numerically identical to the original five-model experiment.

Main estimates pool the held-out outer-fold predictions, not the average of fold-specific metrics. AUROC, average precision (AUPRC), Brier score, logistic calibration intercept/slope, and calibration-in-the-large are reported. Calibration intercept/slope jointly regress outcome on logit(predicted probability); calibration-in-the-large fits an intercept with the original logit as an offset. Ideal intercept=0 and slope=1.

The 95% intervals are percentile intervals from 1,000 bootstrap resamples of fixed cross-fitted/frozen prediction-outcome pairs; MIMIC resamples people and eICU resamples unique people with all their stays. Paired differences use the same resampled observations for both models. These intervals are conditional on the fitted/cross-fitted predictions: they do not repeat the complete model search/fitting pipeline, and eICU intervals do not resample hospitals. They should not be described as full-pipeline bootstrap optimism correction. No multiplicity-adjusted confirmatory claims are made.

Cluster diagnostics use 200 bootstrap resamples with imputation, scaling and clustering refitted in each bootstrap. Centroids are aligned by Hungarian matching in reference-standardized space, without outcomes. ARIs compare reference and bootstrap assignments for all original derivation stays. Assignment confidence reports nearest-centroid distance margin and repeated assignment agreement; these are descriptive, not calibrated posterior class probabilities. Cluster profile mean intervals condition on reference membership, whereas centroid intervals include refitting. External distance summaries compare observations with fixed derivation centroids; cluster-specific derivation 95th-percentile distances identify extrapolation.

External recalibration is explicitly secondary: 5-fold StratifiedGroupKFold over unique people fits intercept and slope on four external folds, then evaluates on the held-out fold. The main frozen models remain unchanged. Apparent full-external recalibration is also shown separately and is not a validation estimate. An APACHE-only subgroup evaluation freezes all original fitted parameters and reports its own denominator.

## Outputs generated locally, not all distributed here

- `cohort_audit.json`: finalized cohort flow, endpoints, follow-up and counts.
- `performance_with_95CI.csv` and `paired_differences_with_95CI.csv`: replacement main results.
- `calibration_deciles.csv`: mean predicted/observed probabilities and descriptive Wilson bin intervals.
- `feature_dictionary.csv`, `feature_sets.json`, `tuning.csv`, `logistic_coefficients_standardized.csv`, `models/*_preprocessing.json`: predictor and fitting appendix.
- `cluster_selection.json`, `cluster_profiles.csv`, `centroids_with_bootstrap_CI.csv`, `cluster_stability_distribution.csv`, `cluster_assignment_confidence.csv`, `cluster_transport_distances.csv`: clustering appendix.
- `external_recalibration.csv`, `external_recalibration_coefficients.csv`, `external_APACHE_subset.csv`: secondary external analyses.
- `cohort_exclusion_characteristics_*.csv`, `missingness_before_carry_forward.csv`, `subgroup_event_counts.csv`: selection and observation appendix.
- `verification.json`, `environment.json`, `provenance.json`, `FINAL_SNAPSHOT.json`: verification and provenance.

Files with `_RESTRICTED` in their names contain participant-level source identifiers and predictions. They are for authorized local verification and must not be placed in an unrestricted public repository. Data access remains governed by the original database agreements. `legacy_before_time_ascertainment` contains superseded intermediate audit outputs and must not be used in the revised paper.
