# Reproducing the reviewer-requested analysis

The finalized analysis has 10,810 MIMIC-IV stays (2,174 hospital deaths) and 4,355 eICU stays (979 hospital deaths). The public export is identified by its Git commit and `../PUBLIC_MANIFEST.json`. The privately retained FINAL_SNAPSHOT identifies the frozen analysis outputs; restricted models are not included in this repository.

The software environment was 64-bit Windows, Python 3.12.14, with package versions in `requirements_revision.txt` and `environment.json`. A seed of 42 supports repeatability; cross-platform bit-for-bit agreement is not claimed.

## Revision scope

The steps below reproduce the principal analysis retained from v1. Revision v2 has also completed integrated calibration error (ICI/E50/E90) and the formal longitudinal latent-class sensitivity. After finishing the principal workflow and freezing its local outputs, follow [the additional-analysis workflow](../additional_revision/README.md). These additions have their own methods, environment records, aggregate reference tables and verification; they are not unimplemented suggestions.

## Installation and data layout

Create a local virtual environment and install the pinned requirements with `python -m pip install -r requirements_revision.txt`. No global installation is required. The analysis optionally prepends a local `_deps` directory if present. No `_deps`, virtual environment, database, or installed library is included in this public export.

Set `SIC_DATA_ROOT` to the directory corresponding to the original `03_数据源` tree. Set `SIC_PROJECT_ROOT` to the original project root only if you also want automatic comparison against the original outputs; those outputs are optional. Patient-level data are not bundled for public redistribution and must be obtained through the database access procedures. Keep the unchanged original scripts below `source_definitions/` (with their numbered subdirectories) or `source_definitions/02_可复现代码/`; alternatively set `SIC_SOURCE_CODE_ROOT` to their original code directory. `supplemental_audit.py` supports all three layouts when recording source hashes.

Required processed MIMIC files below `SIC_DATA_ROOT/MIMIC-IV`:

- `07_full_sofa_sepsis3/mimiciv_sepsis3_icu_admit_cohort.csv`
- `07_full_sofa_sepsis3/mimiciv_sepsis3_icu_admit_full_sofa_scores_0_72h.csv`
- `05_outcomes/mimiciv_outcomes.csv`
- `02_window_features/mimiciv_lab_window_features_0_72h_wide.csv`

Required processed eICU files below `SIC_DATA_ROOT/eICU/08_full_sofa_sepsis3_eicu`:

- `eicu_sic_trajectory_features_full_sofa.csv`
- `eicu_full_sofa_sic_scores_0_72h.csv`

Required raw eICU files below `SIC_DATA_ROOT/eicu-collaborative-research-database-2.0`:

- `patient.csv.gz` (to verify hospital-discharge status and exclude unknown outcomes)
- `diagnosis.csv.gz` (to enforce diagnosisoffset <=4,320 minutes)

The original cohort derivation scripts are separately retained in the supplied reproducible-code folder. They define laboratory mappings, systemic-antibiotic/culture pairing, SOFA component calculations and window aggregation; the reviewer-requested scripts apply the corrected eligibility rules, score harmonization and validation workflow. Public source SHA-256 identifiers are recorded in `../PUBLIC_MANIFEST.json`. Recreating those processed files from raw databases requires the database setup and table schemas documented in the original scripts; the revised model script does not silently rebuild or modify a local database.

## Execution order

1. `python revision_analysis.py --audit-only` writes cohort and missingness checks. Confirm finalized counts above.
2. `python revision_analysis.py` fits all 13 model specifications, computes held-out and external predictions, 1,000-resample conditional bootstrap intervals, and freezes models.
3. `python supplemental_audit.py` fits the descriptive K=2..6 clustering diagnostics, 200-resample K=2/K=4 diagnostics, secondary external recalibration, subgroup counts, and observation tables. The source-hash step resolves the packaged `source_definitions` tree or the `SIC_SOURCE_CODE_ROOT` override.
4. Run the separate `sensitivity_revision.py` according to its local instructions for complete-window, single iterative-imputation, observation-probability and outcome-definition sensitivities. Its outputs are written below `sensitivity_revision/`.
5. `python verify_revision.py` reloads frozen models, checks cohort and endpoint alignment, and reproduces all external probabilities within 1e-7 without refitting.
6. For a fresh reproduction, `python freeze_snapshot.py` validates table denominators and writes file hashes to a new `FINAL_SNAPSHOT.json`. The supplied public summary tables are the manuscript reference outputs; preserve a separate checkout before rerunning. The public export does not contain the local frozen snapshot or models. Packaging changes are documented in `../PUBLIC_EXPORT_NOTES.md`.

The original source scripts retain their original path configuration and database setup requirements; `SIC_DATA_ROOT` configures the revised model scripts, not an unmodified legacy `config.py`. Raw re-derivation therefore requires the documented sibling `01_论文文档`, `02_可复现代码`, `03_数据源` layout or an explicit local configuration of the legacy scripts, plus PostgreSQL schemas for the MIMIC extraction. No claim of a one-command raw-database reconstruction is made.

`refresh_external.py` and `supplemental_audit.py --external-only` were used only to update external results after correcting the original source's unknown-outcome coding, retaining the already fitted MIMIC models. They are not needed after a fresh complete run of the current final script. Their historical role is recorded to explain the privately retained execution logs (not part of this export).

## Loading a frozen model

Use ordinary Python imports from the analysis directory:

```python
import revision_analysis
import joblib
model = joblib.load('models/M3_continuous_SIC.joblib')
raw_mimic, raw_eicu, mimic, eicu = revision_analysis.load_data()
probability = model.predict_proba(eicu)[:, 1]
```

`MatrixBuilder` is exported as `revision_analysis.MatrixBuilder`, including when the fitting script is executed directly. The verified local objects do not require an ad hoc `__main__` alias. This public repository contains no pretrained model objects. Run the analysis using independently authorized database access to create models locally; load only trusted local files.

Predictions, cohort IDs and assignment files ending `_RESTRICTED.csv` are for authorized local verification. The public repository distributes source code, environment specifications, feature definitions, and selected non-participant-level result summaries. Patient-level files, fitted model objects and fitted parameter files are excluded. See `../DATA_ACCESS.md` for the source-database sharing requirements.
