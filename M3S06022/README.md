# M3S06022: dynamic SIC measures and post-72-hour mortality prediction

Code and reproducibility documentation for **Dynamic Sepsis-Induced Coagulopathy Measures for Post-72-Hour Mortality Prediction and External Evaluation**.

This directory is a separate medical research project within the repository maintained by `lishizhen1`. The university-governance files at the repository root are unrelated to this manuscript.

## Version

Public code export: **M3S06022-revision-v1**, 2026-10-10. Cite the full Git commit identifier and the manuscript title when referring to this code. `PUBLIC_MANIFEST.json` records SHA-256 hashes of the distributed files. A GitHub commit URL is the fixed version identifier; no archival DOI is claimed.

## Start here

1. Read [data access and sharing](DATA_ACCESS.md). Obtain MIMIC-IV v3.1 and eICU v2.0 through their access procedures.
2. Create a Python 3.12 virtual environment and install `requirements_all.txt`.
3. Follow [the reproduction instructions](audit_analysis/REPRODUCTION_README.md), including the required processed-file layout and source database setup.
4. Set `SIC_DATA_ROOT` to your authorized local `03_数据源` directory, then run the revised scripts from `audit_analysis` in the documented order. The source definitions are provided, but reconstruction from raw databases is not a one-command workflow.
5. After generating the restricted prediction files locally, run `python figures/make_revision_figures.py all` to regenerate the manuscript figures.

The analysis scripts write outputs beside the scripts. Work in a private local checkout or copy and do not commit generated data or model files. This export does not include raw data, processed patient records, patient predictions, trained models, fitted parameters, database passwords, or author manuscripts.

## Reference results

- MIMIC-IV: 10,810 admissions, 2,174 hospital deaths.
- eICU: 4,355 admissions, 979 hospital deaths.
- `audit_analysis/performance_with_95CI.csv`: principal performance estimates and intervals.
- `audit_analysis/paired_differences_with_95CI.csv`: paired performance differences.
- `audit_analysis/sensitivity_revision/`: selected sensitivity-analysis summaries and methods.

These summary tables refer to the revised cohort and supersede the original manuscript results. Intervals condition on fixed cross-fitted/external predictions and do not represent bootstrap refitting of the entire development pipeline. Clustering and additional sensitivity analyses are exploratory. The models are not established for clinical deployment.

## Contents

- `audit_analysis/*.py`: revised eligibility, harmonized SIC scoring, model fitting, fold-contained preprocessing, nested logistic tuning, clustering diagnostics, sensitivities and verification.
- `audit_analysis/source_definitions/`: historical extraction and derivation scripts; original modelling scripts are retained for provenance, not as the source of revised results.
- `feature_dictionary.csv` and `feature_sets.json` inside `audit_analysis`: variables and explicit model specifications.
- `figures/make_revision_figures.py`: portable figure-generation source.
- [Public export notes](PUBLIC_EXPORT_NOTES.md): packaging changes and excluded artifact categories.

Source-database licenses continue to apply to all database access and derived resources. Public availability of code does not grant access to the underlying clinical data or a separate license to redistribute trained models.
