# Data access and trained models

MIMIC-IV v3.1 and eICU Collaborative Research Database v2.0 require credentialed access and the applicable data use agreement. Each researcher must obtain their own authorization.

- MIMIC-IV: https://physionet.org/content/mimiciv/3.1/
- eICU: https://physionet.org/content/eicu-crd/2.0/
- PhysioNet access FAQ: https://physionet.org/about/faqs/
- MIMIC-IV license: https://physionet.org/content/mimiciv/view-license/3.1/

The MIMIC-IV project guidance states that derived datasets and models should be shared through PhysioNet under the same agreement as the source data. Accordingly this GitHub export excludes trained joblib/pickle objects, fitted coefficient/centroid/preprocessing files, patient-level predictions and assignments, and raw or processed participant records. No controlled-access model deposit or model DOI is claimed here.

The source code regenerates these artifacts in an authorized local environment. Local verification and frozen-model code remain included for this purpose. Selected manuscript-level aggregate performance summaries are provided; they do not provide access to patient records.

The repository owner has not supplied an additional source-code license. Database licenses and third-party software licenses are not replaced by this repository. Dependency packages are installed from their distributions and are not redistributed here.
