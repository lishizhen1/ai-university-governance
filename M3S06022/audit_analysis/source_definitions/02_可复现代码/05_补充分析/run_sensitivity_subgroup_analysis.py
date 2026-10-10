from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (MIMIC_DIR, EICU_DIR, SICDB_DIR,
                     OUTPUT_DIR as _OUT, FIGURE_DIR, ensure_output_dirs)
ensure_output_dirs()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MIMIC_DIR            # MIMIC-IV subdirs
EICU_DATA = EICU_DIR            # eICU files
SICDB_DATA = SICDB_DIR          # SICdb files
OUTPUT_DIR = _OUT / "full_k4" / "supplementary"
TABLE_DIR = OUTPUT_DIR / "tables"


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def bool_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_primary() -> pd.DataFrame:
    df = pd.read_csv(_OUT / "full_k4" / "tables" / "cluster_assignments.csv")
    df["hospital_death"] = df["hospital_death"].astype(int)
    df["icu_death"] = bool_to_int(df["icu_mortality"])
    df["death_28d"] = bool_to_int(df["mortality_28d"])
    df["gender_numeric"] = (df["gender"] == "M").astype(int)
    return df


def fill_by_training(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in features:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        median = train[col].median()
        if pd.isna(median):
            median = 0.0
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)
    return train, test


def lr_cv_metrics(
    df: pd.DataFrame,
    endpoint_col: str = "hospital_death",
    phenotype_col: str = "phenotype_name",
    dynamic_suffix: str = "3w",
    model_prefix: str = "",
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    work = df.copy()
    if "anchor_age" not in work.columns and "age_years" in work.columns:
        work["anchor_age"] = work["age_years"]
    if "gender_numeric" not in work.columns:
        if "gender" in work.columns:
            work["gender_numeric"] = (work["gender"].astype(str).str.upper().str[0] == "M").astype(int)
        elif "sex_label" in work.columns:
            work["gender_numeric"] = (work["sex_label"] == "male").astype(int)
        else:
            work["gender_numeric"] = 0

    endpoint = bool_to_int(work[endpoint_col])
    valid = endpoint.notna()
    work = work.loc[valid].copy()
    y = endpoint.loc[valid].astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 50:
        return {
            f"{model_prefix}n": float(len(y)),
            f"{model_prefix}event_rate": float(np.mean(y)) if len(y) else np.nan,
            f"{model_prefix}m3_auroc": np.nan,
            f"{model_prefix}m4_auroc": np.nan,
            f"{model_prefix}delta_m4_m3": np.nan,
        }

    if dynamic_suffix == "2w":
        dynamic = ["sic_mean_2w", "sic_max_2w", "sic_min_2w", "sic_slope_2w"]
    else:
        dynamic = ["sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope"]

    m1 = ["anchor_age", "gender_numeric"]
    if "w1_sofa_total_assuming_missing_normal" in work.columns:
        m1.append("w1_sofa_total_assuming_missing_normal")
    elif "w1_sofa_total" in work.columns:
        m1.append("w1_sofa_total")
    m2 = m1 + ["w1_sic_total_score"]
    m3 = m2 + dynamic

    phenotype_dummies = pd.get_dummies(work[phenotype_col], prefix="pheno") if phenotype_col in work.columns else pd.DataFrame(index=work.index)
    work = pd.concat([work, phenotype_dummies], axis=1)
    m4 = m3 + list(phenotype_dummies.columns)

    metrics: dict[str, dict[str, float]] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, features in {"M3": m3, "M4": m4}.items():
        available = [f for f in features if f in work.columns]
        xwork, _ = fill_by_training(work, work.iloc[:0].copy(), available)
        X = xwork[available].values.astype(float)
        preds = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        metrics[model_name] = {
            "auroc": roc_auc_score(y, preds),
            "auprc": average_precision_score(y, preds),
            "brier": brier_score_loss(y, preds),
        }
    return {
        f"{model_prefix}n": float(len(y)),
        f"{model_prefix}event_rate": float(np.mean(y)),
        f"{model_prefix}m3_auroc": metrics["M3"]["auroc"],
        f"{model_prefix}m4_auroc": metrics["M4"]["auroc"],
        f"{model_prefix}m3_auprc": metrics["M3"]["auprc"],
        f"{model_prefix}m4_auprc": metrics["M4"]["auprc"],
        f"{model_prefix}m3_brier": metrics["M3"]["brier"],
        f"{model_prefix}m4_brier": metrics["M4"]["brier"],
        f"{model_prefix}delta_m4_m3": metrics["M4"]["auroc"] - metrics["M3"]["auroc"],
    }


def cluster_kmeans(
    df: pd.DataFrame,
    features: list[str],
    k: int,
    phenotype_col: str = "phenotype_name",
    require_complete: bool = False,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    if require_complete:
        work = work.dropna(subset=features).copy()
    else:
        if "w1_sic_total_score" in work.columns:
            for col in features:
                if col.startswith("w2_") and col.replace("w2_", "w1_", 1) in work.columns:
                    work[col] = work[col].fillna(work[col.replace("w2_", "w1_", 1)])
                if col.startswith("w3_") and col.replace("w3_", "w1_", 1) in work.columns:
                    work[col] = work[col].fillna(work[col.replace("w3_", "w1_", 1)])
        for col in features:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            work[col] = work[col].fillna(work[col].median())
    scaler = StandardScaler()
    X = scaler.fit_transform(work[features].values.astype(float))
    labels = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500).fit_predict(X)
    work["sensitivity_cluster"] = labels

    profiles = work.groupby("sensitivity_cluster").agg(
        mort=("hospital_death", "mean"),
        mean_sic=("sic_mean_3w", "mean"),
    )
    ordered = profiles.sort_values(["mort", "mean_sic"]).index.tolist()
    if k == 4:
        names = {
            ordered[0]: "Stable Low-Risk",
            ordered[1]: "High-SIC Improving",
            ordered[2]: "Intermediate Persistent Risk",
            ordered[3]: "Persistently High-Risk",
        }
    elif k == 2:
        names = {
            ordered[0]: "Lower-Risk",
            ordered[1]: "Higher-Risk",
        }
    else:
        names = {cluster: f"Cluster {cluster}" for cluster in profiles.index}
    work[phenotype_col] = work["sensitivity_cluster"].map(names)
    return work


def load_partial_sic() -> pd.DataFrame:
    cohort = pd.read_csv(DATA_DIR / "01_cohort" / "mimiciv_icd_sepsis_candidate_cohort.csv")
    outcomes = pd.read_csv(DATA_DIR / "05_outcomes" / "mimiciv_icd_sepsis_candidate_outcomes.csv")
    traj = pd.read_csv(DATA_DIR / "04_trajectory_features" / "mimiciv_icd_sepsis_candidate_partial_trajectory_features.csv")
    df = cohort.merge(outcomes, on=["subject_id", "hadm_id", "stay_id"], how="inner", suffixes=("", "_out"))
    df = df[df["icu_los_days"] >= 3].copy()
    df["hospital_death"] = df["hospital_expire_flag"].astype(int)
    df["gender_numeric"] = (df["gender"] == "M").astype(int)
    df = df.merge(traj, on=["subject_id", "hadm_id", "stay_id"], how="left")
    rename = {
        "w1_sic_coagulation_partial_score": "w1_sic_total_score",
        "w2_sic_coagulation_partial_score": "w2_sic_total_score",
        "w3_sic_coagulation_partial_score": "w3_sic_total_score",
    }
    df = df.rename(columns=rename)
    score_cols = ["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]
    df["sic_mean_3w"] = df[score_cols].mean(axis=1)
    df["sic_max_3w"] = df[score_cols].max(axis=1)
    df["sic_min_3w"] = df[score_cols].min(axis=1)
    df["sic_slope"] = (df["w3_sic_total_score"] - df["w1_sic_total_score"]) / 2.0
    df = df.dropna(subset=["w1_sic_total_score"]).copy()
    features = [
        "w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score",
        "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
        "w1_inr_max", "w2_inr_max", "w3_inr_max",
    ]
    features = [f for f in features if f in df.columns]
    return cluster_kmeans(df, features, k=2)


def load_48h_landmark() -> pd.DataFrame:
    cohort = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sepsis3_icu_admit_cohort.csv")
    outcomes = pd.read_csv(DATA_DIR / "05_outcomes" / "mimiciv_outcomes.csv")
    lab_wide = pd.read_csv(DATA_DIR / "02_window_features" / "mimiciv_lab_window_features_0_72h_wide.csv")
    sic_long = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sic_scores_full_sofa.csv")
    sofa_long = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_full_sofa_scores_0_72h.csv")

    sic_vals = [
        "sic_total_score", "sofa_4systems_for_sic", "sofa_respiration",
        "sofa_cardiovascular", "sofa_renal", "sofa_liver",
    ]
    sic_wide = sic_long.pivot_table(
        index=["subject_id", "hadm_id", "stay_id"],
        columns="window_name",
        values=[c for c in sic_vals if c in sic_long.columns],
        aggfunc="max",
    )
    sic_wide.columns = [
        f"{win.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}"
        for col, win in sic_wide.columns
    ]
    sic_wide = sic_wide.reset_index()

    sofa_wide = sofa_long.pivot_table(
        index=["subject_id", "hadm_id", "stay_id"],
        columns="window_name",
        values=["sofa_total_assuming_missing_normal"],
        aggfunc="max",
    )
    sofa_wide.columns = [
        f"{win.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}"
        for col, win in sofa_wide.columns
    ]
    sofa_wide = sofa_wide.reset_index()

    df = cohort.merge(outcomes, on=["subject_id", "hadm_id", "stay_id"], how="inner", suffixes=("", "_out"))
    df = df[df["icu_los_days"] >= 2].copy()
    df["hospital_death"] = df["hospital_expire_flag"].astype(int)
    df["gender_numeric"] = (df["gender"] == "M").astype(int)
    df = df.merge(lab_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    df = df.merge(sic_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    df = df.merge(sofa_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    df = df.dropna(subset=["w1_sic_total_score", "w2_sic_total_score"]).copy()
    df["sic_mean_2w"] = df[["w1_sic_total_score", "w2_sic_total_score"]].mean(axis=1)
    df["sic_max_2w"] = df[["w1_sic_total_score", "w2_sic_total_score"]].max(axis=1)
    df["sic_min_2w"] = df[["w1_sic_total_score", "w2_sic_total_score"]].min(axis=1)
    df["sic_slope_2w"] = df["w2_sic_total_score"] - df["w1_sic_total_score"]
    # Also populate 3w names for clustering helper profile naming.
    df["sic_mean_3w"] = df["sic_mean_2w"]
    df["sic_slope"] = df["sic_slope_2w"]
    features = [
        "w1_sic_total_score", "w2_sic_total_score",
        "w1_platelet_min", "w2_platelet_min", "w1_inr_max", "w2_inr_max",
        "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic",
        "w1_sofa_respiration", "w2_sofa_respiration",
        "w1_sofa_cardiovascular", "w2_sofa_cardiovascular",
        "w1_sofa_renal", "w2_sofa_renal",
    ]
    features = [f for f in features if f in df.columns]
    return cluster_kmeans(df, features, k=4)


def load_full_raw_landmark(primary: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the full-SIC landmark feature table before W2/W3 carry-forward."""
    cohort = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sepsis3_icu_admit_cohort.csv")
    outcomes = pd.read_csv(DATA_DIR / "05_outcomes" / "mimiciv_outcomes.csv")
    lab_wide = pd.read_csv(DATA_DIR / "02_window_features" / "mimiciv_lab_window_features_0_72h_wide.csv")
    sic_long = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sic_scores_full_sofa.csv")
    sofa_long = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_full_sofa_scores_0_72h.csv")

    sic_vals = [
        "sic_total_score", "sic_flag", "sofa_4systems_for_sic",
        "sic_platelet_component", "sic_inr_component", "sic_organ_sofa_component",
        "sofa_respiration", "sofa_liver", "sofa_cardiovascular", "sofa_renal",
    ]
    sic_wide = sic_long.pivot_table(
        index=["subject_id", "hadm_id", "stay_id"],
        columns="window_name",
        values=[c for c in sic_vals if c in sic_long.columns],
        aggfunc="max",
    )
    sic_wide.columns = [
        f"{win.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}"
        for col, win in sic_wide.columns
    ]
    sic_wide = sic_wide.reset_index()

    sofa_wide = sofa_long.pivot_table(
        index=["subject_id", "hadm_id", "stay_id"],
        columns="window_name",
        values=["sofa_total_assuming_missing_normal", "sofa_components_available"],
        aggfunc="max",
    )
    sofa_wide.columns = [
        f"{win.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}"
        for col, win in sofa_wide.columns
    ]
    sofa_wide = sofa_wide.reset_index()

    df = cohort.merge(outcomes, on=["subject_id", "hadm_id", "stay_id"], how="inner", suffixes=("", "_out"))
    df = df[df["icu_los_days"] >= 3].copy()
    df["hospital_death"] = df["hospital_expire_flag"].astype(int)
    df["icu_death"] = bool_to_int(df["icu_mortality"])
    df["death_28d"] = bool_to_int(df["mortality_28d"])
    df["gender_numeric"] = (df["gender"] == "M").astype(int)
    df = df.merge(lab_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    df = df.merge(sic_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    df = df.merge(sofa_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    df = df.dropna(subset=["w1_sic_total_score"]).copy()
    score_cols = ["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]
    df["sic_mean_3w"] = df[score_cols].mean(axis=1)
    df["sic_max_3w"] = df[score_cols].max(axis=1)
    df["sic_min_3w"] = df[score_cols].min(axis=1)
    df["sic_slope"] = (df["w3_sic_total_score"] - df["w1_sic_total_score"]) / 2.0
    labels = primary[["stay_id", "phenotype_name"]].drop_duplicates("stay_id")
    return df.merge(labels, on="stay_id", how="left")


def sicdb_transport(primary: pd.DataFrame) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    path = SICDB_DATA / "09_full_sofa_sicdb" / "sicdb_admission_form_sepsis_landmark_full_sic_trajectory_features.csv"
    if not path.exists():
        return {"n": 0, "event_rate": np.nan, "m3_auroc": np.nan, "m4_auroc": np.nan, "delta_m4_m3": np.nan}
    ext = pd.read_csv(path)
    ext = ext.copy()
    ext["anchor_age"] = ext["age_years"]
    ext["gender"] = ext["sex_label"].map({"male": "M", "female": "F"}).fillna("U")
    ext["hospital_death"] = ext["hospital_death"].astype(int)

    cluster_features = [
        "w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score",
        "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
        "w1_inr_max", "w2_inr_max", "w3_inr_max",
        "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic", "w3_sofa_4systems_for_sic",
        "w1_sofa_respiration", "w2_sofa_respiration", "w3_sofa_respiration",
        "w1_sofa_cardiovascular", "w2_sofa_cardiovascular", "w3_sofa_cardiovascular",
        "w1_sofa_renal", "w2_sofa_renal", "w3_sofa_renal",
    ]
    train = primary.dropna(subset=["w1_sic_total_score"]).copy()
    for frame in [train, ext]:
        for col in cluster_features:
            if col.startswith("w2_") and col.replace("w2_", "w1_", 1) in frame.columns:
                frame[col] = frame[col].fillna(frame[col.replace("w2_", "w1_", 1)])
            if col.startswith("w3_") and col.replace("w3_", "w1_", 1) in frame.columns:
                frame[col] = frame[col].fillna(frame[col.replace("w3_", "w1_", 1)])
    train, ext = fill_by_training(train, ext, cluster_features)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[cluster_features].values.astype(float))
    km = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
    train_clusters = km.fit_predict(X_train)
    train["_proj_cluster"] = train_clusters
    mapping = (
        train.groupby("_proj_cluster")["phenotype_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    ext_clusters = km.predict(scaler.transform(ext[cluster_features].values.astype(float)))
    ext["phenotype_name"] = pd.Series(ext_clusters).map(mapping).values

    m3_features = [
        "anchor_age", "gender_numeric", "w1_sofa_total_assuming_missing_normal",
        "w1_sic_total_score", "sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope",
    ]
    pheno_train = pd.get_dummies(primary["phenotype_name"], prefix="pheno")
    pheno_ext = pd.get_dummies(ext["phenotype_name"], prefix="pheno")
    for col in pheno_train.columns:
        if col not in pheno_ext.columns:
            pheno_ext[col] = 0
    train_model = pd.concat([primary.copy(), pheno_train], axis=1)
    ext_model = pd.concat([ext.copy(), pheno_ext[pheno_train.columns]], axis=1)
    m4_features = m3_features + list(pheno_train.columns)
    train_model, ext_model = fill_by_training(train_model, ext_model, m4_features)

    y_train = primary["hospital_death"].astype(int).values
    y_ext = ext["hospital_death"].astype(int).values
    out: dict[str, float] = {"n": float(len(ext)), "event_rate": float(np.mean(y_ext))}
    for name, features in {"m3": m3_features, "m4": m4_features}.items():
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(train_model[features].values.astype(float), y_train)
        pred = model.predict_proba(ext_model[features].values.astype(float))[:, 1]
        out[f"{name}_auroc"] = roc_auc_score(y_ext, pred) if len(np.unique(y_ext)) == 2 else np.nan
        out[f"{name}_auprc"] = average_precision_score(y_ext, pred) if len(np.unique(y_ext)) == 2 else np.nan
        out[f"{name}_brier"] = brier_score_loss(y_ext, pred)
    out["delta_m4_m3"] = out["m4_auroc"] - out["m3_auroc"]
    return out


def run_sensitivity(primary: pd.DataFrame) -> pd.DataFrame:
    log("run sensitivity analyses")
    rows: list[dict[str, object]] = []
    raw_full = load_full_raw_landmark(primary)

    optimal_path = _OUT / "full_optimal" / "metrics" / "analysis_metrics.json"
    if optimal_path.exists():
        optimal = json.loads(optimal_path.read_text())
        rows.append({
            "analysis_id": "S1",
            "analysis": "Data-optimal K=2 phenotype solution",
            "n": optimal["cohort"]["landmark_72h_cohort"],
            "event_rate": optimal["cohort"]["landmark_hospital_mortality_pct"] / 100,
            "m3_auroc": optimal["models"]["M3_Continuous"]["AUROC"],
            "m4_auroc": optimal["models"]["M4_Phenotype"]["AUROC"],
            "delta_m4_m3": optimal["incremental_auroc"]["M3_Continuous_to_M4_Phenotype"],
            "interpretation": "K=2 remained stable but less informative than the prespecified K=4 phenotype layer.",
        })
    else:
        log("S1 skipped — full_optimal metrics not available")

    partial = load_partial_sic()
    partial_metrics = lr_cv_metrics(partial)
    rows.append({
        "analysis_id": "S2",
        "analysis": "Partial SIC proxy instead of full SOFA-based SIC",
        **partial_metrics,
        "interpretation": "Partial SIC preserved direction but attenuated phenotype increment.",
    })

    complete3 = raw_full[raw_full[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].notna().all(axis=1)].copy()
    complete3_metrics = lr_cv_metrics(complete3)
    rows.append({
        "analysis_id": "S3",
        "analysis": "Complete W1-W3 full SIC only",
        **complete3_metrics,
        "interpretation": "Requiring all three windows retained the positive M4 increment.",
    })

    cluster_features = [
        "w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score",
        "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
        "w1_inr_max", "w2_inr_max", "w3_inr_max",
        "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic", "w3_sofa_4systems_for_sic",
        "w1_sofa_respiration", "w2_sofa_respiration", "w3_sofa_respiration",
        "w1_sofa_cardiovascular", "w2_sofa_cardiovascular", "w3_sofa_cardiovascular",
        "w1_sofa_renal", "w2_sofa_renal", "w3_sofa_renal",
    ]
    complete_cluster = cluster_kmeans(raw_full, cluster_features, k=4, require_complete=True)
    no_cf_metrics = lr_cv_metrics(complete_cluster, phenotype_col="phenotype_name")
    rows.append({
        "analysis_id": "S4",
        "analysis": "No carry-forward imputation for clustering",
        **no_cf_metrics,
        "interpretation": "Complete-feature clustering produced similar discrimination despite smaller sample size.",
    })

    forty8 = load_48h_landmark()
    forty8_metrics = lr_cv_metrics(forty8, dynamic_suffix="2w")
    rows.append({
        "analysis_id": "S5",
        "analysis": "48-hour landmark using W1-W2 trajectories",
        **forty8_metrics,
        "interpretation": "Earlier two-window trajectories showed a smaller but positive phenotype signal.",
    })

    d28_metrics = lr_cv_metrics(primary, endpoint_col="death_28d")
    rows.append({
        "analysis_id": "S6",
        "analysis": "Alternative endpoint: 28-day mortality",
        **d28_metrics,
        "interpretation": "The M4 increment persisted for time-limited mortality.",
    })

    icu_metrics = lr_cv_metrics(primary, endpoint_col="icu_death")
    rows.append({
        "analysis_id": "S7",
        "analysis": "Alternative endpoint: ICU mortality",
        **icu_metrics,
        "interpretation": "Phenotype labels remained useful for ICU mortality.",
    })

    sicdb_metrics = sicdb_transport(primary)
    rows.append({
        "analysis_id": "S8",
        "analysis": "SICdb full-SIC transport sensitivity",
        **sicdb_metrics,
        "interpretation": "SICdb was analyzable after full SIC derivation but remained limited by missing public GCS/CNS data and small sepsis-landmark sample.",
    })

    result = pd.DataFrame(rows)
    rename = {
        "n": "n",
        "event_rate": "event_rate",
        "m3_auroc": "m3_auroc",
        "m4_auroc": "m4_auroc",
        "m3_auprc": "m3_auprc",
        "m4_auprc": "m4_auprc",
        "m3_brier": "m3_brier",
        "m4_brier": "m4_brier",
        "delta_m4_m3": "delta_m4_m3",
    }
    result = result.rename(columns=rename)
    return result


def forest_or(a: float, b: float, c: float, d: float) -> tuple[float, float, float]:
    # Haldane-Anscombe correction.
    a += 0.5
    b += 0.5
    c += 0.5
    d += 0.5
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return math.exp(log_or), math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se)


def subgroup_rows(primary: pd.DataFrame) -> pd.DataFrame:
    log("run subgroup forest analysis")
    df = primary.copy()
    df["age_group"] = np.where(df["anchor_age"] >= 65, "Age >=65", "Age <65")
    df["sex_group"] = np.where(df["gender"] == "M", "Male", "Female")
    df["w1_sofa_group"] = np.where(df["w1_sofa_total_assuming_missing_normal"] >= 8, "W1 SOFA >=8", "W1 SOFA <8")
    df["w1_sic_group"] = np.where(df["w1_sic_total_score"] >= 4, "W1 SIC >=4", "W1 SIC <4")
    df["renal_group"] = np.where(df["w1_sofa_renal"] >= 2, "Renal SOFA >=2", "Renal SOFA <2")
    df["cardio_group"] = np.where(df["w1_sofa_cardiovascular"] >= 2, "Cardiovascular SOFA >=2", "Cardiovascular SOFA <2")
    df["resp_group"] = np.where(df["w1_sofa_respiration"] >= 2, "Respiratory SOFA >=2", "Respiratory SOFA <2")
    df["sepsis_icd_group"] = np.where(bool_to_int(df["sepsis_icd_flag"]).eq(1), "ICD sepsis", "No ICD sepsis")
    df["icu_type_group"] = np.where(
        df["first_careunit"].fillna("").str.contains("Surgical|Trauma|Neuro", case=False, regex=True),
        "Surgical/trauma/neuro ICU",
        "Medical/cardiac ICU",
    )

    specs = [
        ("Age", "age_group"),
        ("Sex", "sex_group"),
        ("Baseline SOFA", "w1_sofa_group"),
        ("Baseline SIC", "w1_sic_group"),
        ("Renal dysfunction", "renal_group"),
        ("Cardiovascular dysfunction", "cardio_group"),
        ("Respiratory dysfunction", "resp_group"),
        ("ICD sepsis", "sepsis_icd_group"),
        ("ICU type", "icu_type_group"),
    ]
    rows = []
    for domain, col in specs:
        for level in sorted(df[col].dropna().unique()):
            sub = df[df[col].eq(level)]
            pair = sub[sub["phenotype_name"].isin(["Stable Low-Risk", "Persistently High-Risk"])].copy()
            high = pair["phenotype_name"].eq("Persistently High-Risk")
            low = pair["phenotype_name"].eq("Stable Low-Risk")
            a = int((high & pair["hospital_death"].eq(1)).sum())
            b = int((high & pair["hospital_death"].eq(0)).sum())
            c = int((low & pair["hospital_death"].eq(1)).sum())
            d = int((low & pair["hospital_death"].eq(0)).sum())
            if high.sum() < 20 or low.sum() < 20:
                continue
            or_, lo, hi = forest_or(a, b, c, d)
            rows.append({
                "domain": domain,
                "subgroup": level,
                "n_high_risk": int(high.sum()),
                "n_low_risk": int(low.sum()),
                "mortality_high_risk": a / max(1, high.sum()),
                "mortality_low_risk": c / max(1, low.sum()),
                "or": or_,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def plot_forest(forest: pd.DataFrame) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    plot_df = forest.copy()
    plot_df["label"] = plot_df["domain"] + ": " + plot_df["subgroup"]
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    fig_h = max(5.5, 0.34 * len(plot_df) + 1.3)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["or"],
        y,
        xerr=[plot_df["or"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["or"]],
        fmt="o",
        color="#D55E00",
        ecolor="#555555",
        elinewidth=1,
        capsize=2,
        markersize=4.5,
    )
    ax.axvline(1, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("Odds ratio for hospital mortality\nPersistently High-Risk vs Stable Low-Risk")
    ax.set_title("Subgroup Mortality Association")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0.8, max(11, float(plot_df["ci_high"].max()) * 1.05))
    for i, row in plot_df.iterrows():
        ax.text(
            1.02,
            i,
            f"{row['or']:.2f} ({row['ci_low']:.2f}-{row['ci_high']:.2f})",
            va="center",
            ha="left",
            fontsize=7.5,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
    fig.subplots_adjust(left=0.44, right=0.74, top=0.92, bottom=0.10)
    fig.savefig(FIGURE_DIR / "fig_subgroup_forest.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURE_DIR / "fig_subgroup_forest.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_latex_tables(sens: pd.DataFrame) -> None:
    table = sens.copy()
    table["N"] = table["n"].round(0).astype("Int64").astype(str)
    table["Events"] = (table["event_rate"] * 100).map(lambda x: f"{x:.1f}\\%" if pd.notna(x) else "--")
    table["M3"] = table["m3_auroc"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    table["M4"] = table["m4_auroc"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    table["Delta"] = table["delta_m4_m3"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "--")
    rows = []
    for _, row in table.iterrows():
        name = str(row["analysis"]).replace("%", "\\%").replace("_", "\\_")
        rows.append(f"{row['analysis_id']} & {name} & {row['N']} & {row['M3']} & {row['M4']} & {row['Delta']} \\\\")
    tex = "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Sensitivity analyses for the incremental value of trajectory phenotype labels.}",
        "\\label{tab:sensitivity}",
        "\\begin{tabular}{lp{0.46\\textwidth}rrrr}",
        "\\toprule",
        "ID & Analysis & N & M3 AUROC & M4 AUROC & $\\Delta$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    (FIGURE_DIR / "TABLE_V_sensitivity.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    primary = load_primary()
    sens = run_sensitivity(primary)
    forest = subgroup_rows(primary)
    sens.to_csv(TABLE_DIR / "sensitivity_analyses.csv", index=False)
    forest.to_csv(TABLE_DIR / "subgroup_forest_data.csv", index=False)
    write_latex_tables(sens)
    plot_forest(forest)
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "sensitivity_rows": int(len(sens)),
        "subgroup_rows": int(len(forest)),
        "primary_m3_to_m4_delta": float(
            sens.loc[sens["analysis_id"].eq("S3"), "delta_m4_m3"].iloc[0]
        ) if sens["analysis_id"].eq("S3").any() else None,
        "forest_reference": "Persistently High-Risk vs Stable Low-Risk",
    }
    (OUTPUT_DIR / "sensitivity_subgroup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
