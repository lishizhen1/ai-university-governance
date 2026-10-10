#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIC Trajectory Phenotyping and Mortality Prediction Analysis
============================================================
Runs the full analysis pipeline:
1. Cohort assembly with 72h landmark
2. K-means trajectory clustering (K=2..6)
3. Cluster stability assessment (bootstrap)
4. Nested prediction models (M1-M5)
5. External validation on eICU
6. Output tables and figures for the paper

Usage:
    python run_sic_trajectory_analysis.py [--full|--partial] [--skip-external]

    --full          Use full Sepsis-3 cohort and full SIC scores derived from SOFA.
    --partial       Use partial SIC scores (platelet + INR + bilirubin/creatinine proxy)
                    for the earlier sensitivity run.
    --skip-external Skip external validation (eICU/SICdb)
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# Configuration
# ==============================================================================

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (MIMIC_DIR, EICU_DIR, SICDB_DIR, OUTPUT_DIR as _OUT,
                     FIGURE_DIR as _FIG, ensure_output_dirs)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MIMIC_DIR            # MIMIC-IV subdirs: 01_cohort .. 07_full_sofa_sepsis3
EICU_DATA = EICU_DIR            # eICU files at root + 08_full_sofa_sepsis3_eicu/
SICDB_DATA = SICDB_DIR          # SICdb files at root + 09_full_sofa_sicdb/
OUTPUT_DIR = _OUT
FIGURE_DIR = _FIG

ensure_output_dirs()

ANALYSIS_MODE = "partial"
FORCE_K = None


def bool_series_to_int(series):
    """Convert mixed CSV boolean encodings to 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)


def sic_score_columns():
    if ANALYSIS_MODE == "full":
        return ["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]
    return [
        "w1_sic_coagulation_partial_score",
        "w2_sic_coagulation_partial_score",
        "w3_sic_coagulation_partial_score",
    ]


def sic_score_label():
    return "full SIC" if ANALYSIS_MODE == "full" else "partial SIC"


# ==============================================================================
# Step 1: Cohort Assembly
# ==============================================================================

def load_mimiciv_cohort(mode="partial"):
    """Load MIMIC-IV cohort and apply inclusion/exclusion."""
    print("\n" + "="*70)
    print("STEP 1: Cohort Assembly")
    print("="*70)

    if mode == "full":
        cohort_path = DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sepsis3_icu_admit_cohort.csv"
        if not cohort_path.exists():
            raise FileNotFoundError(
                f"Full Sepsis-3 cohort not found: {cohort_path}. "
                "Run derive_full_sofa_sepsis3_mimiciv.py first."
            )
        cohort = pd.read_csv(cohort_path)
        print(f"  MIMIC-IV Sepsis-3 ICU-admit cohort: {len(cohort):,}")
        outcomes_path = DATA_DIR / "05_outcomes" / "mimiciv_outcomes.csv"
    else:
        cohort = pd.read_csv(DATA_DIR / "01_cohort" / "mimiciv_icd_sepsis_candidate_cohort.csv")
        print(f"  MIMIC-IV ICD sepsis candidates: {len(cohort):,}")
        outcomes_path = DATA_DIR / "05_outcomes" / "mimiciv_icd_sepsis_candidate_outcomes.csv"

    # Load outcomes
    outcomes = pd.read_csv(outcomes_path)
    print(f"  Outcomes loaded: {len(outcomes):,}")

    # Merge
    df = cohort.merge(outcomes, on=["subject_id", "hadm_id", "stay_id"], how="inner", suffixes=("", "_out"))
    print(f"  After merge: {len(df):,}")

    # Parse mortality flags
    df["hospital_death"] = df["hospital_expire_flag"].astype(int)
    df["icu_death"] = bool_series_to_int(df["icu_mortality"])
    df["death_28d"] = bool_series_to_int(df["mortality_28d"])

    # Apply 72-hour landmark
    early_death = df[df["icu_los_days"] < 3]
    landmark_cohort = df[df["icu_los_days"] >= 3].copy()
    print(f"  Early deaths/discharges (<72h): {len(early_death):,} (mortality: {early_death['hospital_death'].mean()*100:.1f}%)")
    print(f"  72-hour landmark cohort: {len(landmark_cohort):,}")
    print(f"  Hospital mortality (landmark): {landmark_cohort['hospital_death'].mean()*100:.1f}%")

    return df, landmark_cohort, early_death


# ==============================================================================
# Step 2: Feature Engineering
# ==============================================================================

def build_features(landmark_cohort, mode="partial"):
    """Build trajectory features for clustering."""
    print("\n" + "="*70)
    print("STEP 2: Feature Engineering")
    print("="*70)

    if mode == "full":
        wide = pd.read_csv(DATA_DIR / "02_window_features" / "mimiciv_lab_window_features_0_72h_wide.csv")
        print(f"  Wide laboratory features: {len(wide):,} rows, {len(wide.columns)} columns")

        sic_long = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sic_scores_full_sofa.csv")
        sofa_long = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_full_sofa_scores_0_72h.csv")
        traj = pd.read_csv(DATA_DIR / "07_full_sofa_sepsis3" / "mimiciv_sepsis3_icu_admit_sic_trajectory_features.csv")
        print(f"  Full SIC scores: {len(sic_long):,} rows")
        print(f"  Full SOFA scores: {len(sofa_long):,} rows")
        print(f"  Full trajectory features: {len(traj):,} rows")

        sic_pivot_cols = [
            "sic_total_score", "sic_flag", "sofa_4systems_for_sic",
            "sic_platelet_component", "sic_inr_component", "sic_organ_sofa_component",
            "sofa_respiration", "sofa_liver", "sofa_cardiovascular", "sofa_renal",
        ]
        sic_wide = sic_long.pivot_table(
            index=["subject_id", "hadm_id", "stay_id"],
            columns="window_name",
            values=[c for c in sic_pivot_cols if c in sic_long.columns],
            aggfunc="max",
        )
        sic_wide.columns = [f"{win.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}" for col, win in sic_wide.columns]
        sic_wide = sic_wide.reset_index()
        sic_wide = sic_wide.rename(columns={
            "w1_sic_total_score": "w1_sic_total_score",
            "w2_sic_total_score": "w2_sic_total_score",
            "w3_sic_total_score": "w3_sic_total_score",
        })

        sofa_wide = sofa_long.pivot_table(
            index=["subject_id", "hadm_id", "stay_id"],
            columns="window_name",
            values=["sofa_total_assuming_missing_normal", "sofa_components_available"],
            aggfunc="max",
        )
        sofa_wide.columns = [f"{win.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}" for col, win in sofa_wide.columns]
        sofa_wide = sofa_wide.reset_index()

        df = landmark_cohort.merge(wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
        df = df.merge(sic_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
        df = df.merge(sofa_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
        keep_cols = ["stay_id", "sic_w3_minus_w1", "sofa_w3_minus_w1", "sic_trajectory_group"]
        df = df.merge(traj[[c for c in keep_cols if c in traj.columns]], on="stay_id", how="left")
        sic_cols = ["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]
    else:
        # Load wide-form window features
        wide = pd.read_csv(DATA_DIR / "02_window_features" / "mimiciv_icd_sepsis_candidate_lab_window_features_0_72h_wide.csv")
        print(f"  Wide features: {len(wide):,} rows, {len(wide.columns)} columns")

        # Load trajectory features
        traj = pd.read_csv(DATA_DIR / "04_trajectory_features" / "mimiciv_icd_sepsis_candidate_partial_trajectory_features.csv")
        print(f"  Trajectory features: {len(traj):,} rows")

        sic_wide = traj.copy()

        # Merge with landmark cohort
        df = landmark_cohort.merge(wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
        df = df.merge(sic_wide[[
            "stay_id",
            "w1_sic_coagulation_partial_score", "w2_sic_coagulation_partial_score", "w3_sic_coagulation_partial_score",
            "partial_sic_w2_minus_w1", "partial_sic_w3_minus_w1",
            "platelet_min_w3_minus_w1", "inr_max_w3_minus_w1",
        ]], on="stay_id", how="left")
        sic_cols = [
            "w1_sic_coagulation_partial_score",
            "w2_sic_coagulation_partial_score",
            "w3_sic_coagulation_partial_score",
        ]

    print(f"  Final feature matrix: {len(df):,} rows")

    # Compute additional trajectory features
    sic_matrix = df[sic_cols].copy()
    df["sic_mean_3w"] = sic_matrix.mean(axis=1, skipna=True)
    df["sic_max_3w"] = sic_matrix.max(axis=1, skipna=True)
    df["sic_min_3w"] = sic_matrix.min(axis=1, skipna=True)
    df["n_windows_available"] = sic_matrix.notna().sum(axis=1)
    df["sic_slope"] = (sic_matrix.iloc[:, -1] - sic_matrix.iloc[:, 0]) / 2.0

    return df


# ==============================================================================
# Step 3: K-means Trajectory Clustering
# ==============================================================================

def run_clustering(df):
    """K-means clustering on SIC trajectory features."""
    print("\n" + "="*70)
    print("STEP 3: K-means Trajectory Clustering")
    print("="*70)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler

    score_cols = sic_score_columns()
    score_w1, score_w2, score_w3 = score_cols

    # Clustering features: SIC scores per window + key components.
    cluster_features = score_cols + [
        "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
        "w1_inr_max", "w2_inr_max", "w3_inr_max",
    ]
    if ANALYSIS_MODE == "full":
        cluster_features += [
            "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic", "w3_sofa_4systems_for_sic",
            "w1_sofa_respiration", "w2_sofa_respiration", "w3_sofa_respiration",
            "w1_sofa_cardiovascular", "w2_sofa_cardiovascular", "w3_sofa_cardiovascular",
            "w1_sofa_renal", "w2_sofa_renal", "w3_sofa_renal",
        ]
    cluster_features = [c for c in cluster_features if c in df.columns]

    # Require W1 data at minimum
    df_cluster = df.dropna(subset=[score_w1]).copy()
    print(f"  Patients with W1 {sic_score_label()} data: {len(df_cluster):,}")

    # Impute missing W2/W3 with W1 (conservative: assume no change)
    for col in [score_w2, score_w3]:
        mask = df_cluster[col].isna()
        df_cluster.loc[mask, col] = df_cluster.loc[mask, score_w1]

    for col in cluster_features:
        if col.startswith("w2_") and col.replace("w2_", "w1_", 1) in df_cluster.columns:
            w1_col = col.replace("w2_", "w1_", 1)
            mask = df_cluster[col].isna() & df_cluster[w1_col].notna()
            df_cluster.loc[mask, col] = df_cluster.loc[mask, w1_col]
        if col.startswith("w3_") and col.replace("w3_", "w1_", 1) in df_cluster.columns:
            w1_col = col.replace("w3_", "w1_", 1)
            mask = df_cluster[col].isna() & df_cluster[w1_col].notna()
            df_cluster.loc[mask, col] = df_cluster.loc[mask, w1_col]
        if df_cluster[col].isna().any():
            df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())

    X = df_cluster[cluster_features].copy()
    print(f"  Clustering feature matrix: {X.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate K=2..6
    results = {}
    print(f"\n  {'K':>3} | {'Silhouette':>10} | {'Calinski-H':>12} | {'Davies-B':>10} | {'Min Size':>9}")
    print("  " + "-"*55)

    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        sizes = pd.Series(labels).value_counts()
        min_size_pct = sizes.min() / len(labels) * 100

        results[k] = {
            "labels": labels,
            "model": km,
            "scaler": scaler,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "min_cluster_pct": min_size_pct,
            "cluster_sizes": sizes.to_dict(),
        }
        print(f"  {k:>3} | {sil:>10.3f} | {ch:>12.1f} | {db:>10.3f} | {min_size_pct:>8.1f}%")

    # Select optimal K (highest silhouette with min 5% per cluster), unless forced.
    valid_ks = {k: v for k, v in results.items() if v["min_cluster_pct"] >= 5}
    if not valid_ks:
        valid_ks = results  # fallback

    if FORCE_K is not None:
        best_k = FORCE_K
        if best_k not in results:
            raise ValueError(f"Forced K={best_k} was not evaluated.")
        print(f"\n  Forced K = {best_k} (silhouette = {results[best_k]['silhouette']:.3f})")
    else:
        best_k = max(valid_ks, key=lambda k: valid_ks[k]["silhouette"])
        print(f"\n  Optimal K = {best_k} (silhouette = {results[best_k]['silhouette']:.3f})")

    # Assign labels
    df_cluster["phenotype_cluster"] = results[best_k]["labels"]

    # Name phenotypes by trajectory pattern
    cluster_profiles = df_cluster.groupby("phenotype_cluster").agg({
        score_w1: "mean",
        score_w2: "mean",
        score_w3: "mean",
        "hospital_death": "mean",
    }).round(2)
    print(f"\n  Cluster profiles ({sic_score_label()}):")
    print(cluster_profiles.to_string())

    # Auto-name based on trajectory pattern
    phenotype_names = {}
    if ANALYSIS_MODE == "full" and best_k == 4:
        profiles = cluster_profiles.copy()
        profiles["mean_sic"] = profiles[[score_w1, score_w2, score_w3]].mean(axis=1)
        low_cluster = profiles["mean_sic"].idxmin()
        high_mort_cluster = profiles["hospital_death"].idxmax()
        phenotype_names[low_cluster] = "Stable Low-Risk"
        phenotype_names[high_mort_cluster] = "Persistently High-Risk"
        remaining = [c for c in profiles.index if c not in phenotype_names]
        remaining_sorted = sorted(remaining, key=lambda c: profiles.loc[c, "mean_sic"], reverse=True)
        if remaining_sorted:
            c = remaining_sorted[0]
            phenotype_names[c] = "High-SIC Improving" if profiles.loc[c, score_w3] < profiles.loc[c, score_w1] else "High-SIC Persistent"
        if len(remaining_sorted) > 1:
            phenotype_names[remaining_sorted[1]] = "Intermediate Persistent Risk"
    for c in range(best_k):
        row = cluster_profiles.loc[c]
        w1, w2, w3 = row[score_w1], row[score_w2], row[score_w3]
        mort = row["hospital_death"]
        if c not in phenotype_names:
            high_threshold = 4 if ANALYSIS_MODE == "full" else 2
            low_threshold = 2 if ANALYSIS_MODE == "full" else 1

            if w1 <= low_threshold and w2 <= low_threshold and w3 <= low_threshold:
                name = "Stable Low-Risk"
            elif w1 > w3 + 0.5 and w3 < high_threshold:
                name = "Early Improving"
            elif w3 > w1 + 0.5:
                name = "New-Onset Progressive"
            elif w1 >= high_threshold and w2 >= high_threshold and w3 >= high_threshold:
                name = "Persistently High-Risk"
            else:
                name = f"Phenotype-{c+1}"
            phenotype_names[c] = name
        print(f"  Cluster {c} → {phenotype_names[c]}: W1={w1:.1f}, W2={w2:.1f}, W3={w3:.1f}, Mortality={mort*100:.1f}%")

    seen_names = {}
    for c, name in list(phenotype_names.items()):
        seen_names[name] = seen_names.get(name, 0) + 1
        if seen_names[name] > 1:
            phenotype_names[c] = f"{name} (Cluster {c})"

    df_cluster["phenotype_name"] = df_cluster["phenotype_cluster"].map(phenotype_names)

    # Mortality by phenotype
    mort_by_pheno = df_cluster.groupby("phenotype_name").agg(
        n=("stay_id", "count"),
        hospital_mortality=("hospital_death", "mean"),
        icu_mortality=("icu_death", "mean"),
    ).round(4)
    mort_by_pheno["hospital_mortality_pct"] = (mort_by_pheno["hospital_mortality"] * 100).round(1)
    mort_by_pheno["icu_mortality_pct"] = (mort_by_pheno["icu_mortality"] * 100).round(1)
    mort_by_pheno["pct_of_cohort"] = (mort_by_pheno["n"] / len(df_cluster) * 100).round(1)
    print(f"\n  Mortality by phenotype:")
    print(mort_by_pheno.to_string())

    return df_cluster, results, best_k, phenotype_names, scaler, cluster_features


# ==============================================================================
# Step 4: Bootstrap Cluster Stability
# ==============================================================================

def bootstrap_stability(df_cluster, cluster_features, best_k, n_iter=200):
    """Bootstrap stability with adjusted Rand index."""
    print("\n" + "="*70)
    print("STEP 4: Bootstrap Cluster Stability")
    print("="*70)

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    X = df_cluster[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reference clustering
    km_ref = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels_ref = km_ref.fit_predict(X_scaled)

    ari_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_iter):
        # Bootstrap sample
        idx = rng.choice(len(X_scaled), size=len(X_scaled), replace=True)
        X_boot = X_scaled[idx]

        km_boot = KMeans(n_clusters=best_k, random_state=i, n_init=10)
        labels_boot = km_boot.fit_predict(X_boot)

        # Map bootstrap labels to reference labels via the shared samples
        # Use ARI on the overlapping indices
        unique_idx, counts = np.unique(idx, return_counts=True)
        # Only compare labels for indices that appear in bootstrap
        ref_sub = labels_ref[idx]
        ari = adjusted_rand_score(ref_sub, labels_boot)
        ari_scores.append(ari)

    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)
    print(f"  Bootstrap iterations: {n_iter}")
    print(f"  Adjusted Rand Index: {ari_mean:.3f} ± {ari_std:.3f}")
    print(f"  Interpretation: {'excellent' if ari_mean > 0.8 else 'good' if ari_mean > 0.6 else 'moderate' if ari_mean > 0.4 else 'poor'} stability")

    return {"ari_mean": ari_mean, "ari_std": ari_std, "n_iter": n_iter}


# ==============================================================================
# Step 5: Nested Prediction Models
# ==============================================================================

def run_nested_models(df_cluster):
    """Run 5 nested prediction models with cross-validation."""
    print("\n" + "="*70)
    print("STEP 5: Nested Prediction Models (M1-M5)")
    print("="*70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler

    y = df_cluster["hospital_death"].values
    score_w1, _, _ = sic_score_columns()

    # Prepare features for each model tier
    # M1: Baseline (age, gender)
    m1_features = ["anchor_age", "gender_numeric"]
    if ANALYSIS_MODE == "full" and "w1_sofa_total_assuming_missing_normal" in df_cluster.columns:
        m1_features += ["w1_sofa_total_assuming_missing_normal"]

    # M2: +Static SIC (W1 score)
    m2_features = m1_features + [score_w1]

    # M3: +Continuous summaries (mean, max, min, delta across W1-W3)
    m3_features = m2_features + ["sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope"]

    # M4: +Phenotype (add cluster label as one-hot)
    # Will be handled separately

    # M5: Extended (all features + XGBoost)
    m5_features = [
        "anchor_age", "gender_numeric",
        "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
        "w1_inr_max", "w2_inr_max", "w3_inr_max",
        "w1_bilirubin_total_max", "w2_bilirubin_total_max", "w3_bilirubin_total_max",
        "w1_creatinine_max", "w2_creatinine_max", "w3_creatinine_max",
        "w1_lactate_max", "w2_lactate_max", "w3_lactate_max",
        "sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope",
    ]
    if ANALYSIS_MODE == "full":
        m5_features += [
            "w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score",
            "w1_sofa_total_assuming_missing_normal", "w2_sofa_total_assuming_missing_normal", "w3_sofa_total_assuming_missing_normal",
            "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic", "w3_sofa_4systems_for_sic",
            "w1_sofa_respiration", "w2_sofa_respiration", "w3_sofa_respiration",
            "w1_sofa_liver", "w2_sofa_liver", "w3_sofa_liver",
            "w1_sofa_cardiovascular", "w2_sofa_cardiovascular", "w3_sofa_cardiovascular",
            "w1_sofa_renal", "w2_sofa_renal", "w3_sofa_renal",
            "w1_sic_platelet_component", "w2_sic_platelet_component", "w3_sic_platelet_component",
            "w1_sic_inr_component", "w2_sic_inr_component", "w3_sic_inr_component",
            "w1_sic_organ_sofa_component", "w2_sic_organ_sofa_component", "w3_sic_organ_sofa_component",
        ]

    # Prepare data
    df_model = df_cluster.copy()
    df_model["gender_numeric"] = (df_model["gender"] == "M").astype(int)

    # Impute missing with median
    all_features = list(set(m1_features + m2_features + m3_features + m5_features))
    for col in all_features:
        if col in df_model.columns and df_model[col].dtype in [np.float64, np.int64, float, int]:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    # Phenotype one-hot for M4
    phenotype_dummies = pd.get_dummies(df_model["phenotype_name"], prefix="pheno")
    df_model = pd.concat([df_model, phenotype_dummies], axis=1)
    m4_features = m3_features + list(phenotype_dummies.columns)

    models_config = {
        "M1_Baseline": m1_features,
        "M2_StaticSIC": m2_features,
        "M3_Continuous": m3_features,
        "M4_Phenotype": m4_features,
        "M5_Extended_XGB": m5_features,
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, features in models_config.items():
        available_features = [f for f in features if f in df_model.columns]
        X = df_model[available_features].values.astype(float)

        # Handle any remaining NaN
        X = np.nan_to_num(X, nan=0.0)

        if model_name == "M5_Extended_XGB":
            try:
                import xgboost as xgb
                use_xgb = True
            except ImportError:
                print(f"  WARNING: xgboost not available, using LogisticRegression for {model_name}")
                use_xgb = False
        else:
            use_xgb = False

        fold_aurocs = []
        fold_auprcs = []
        fold_briers = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if use_xgb:
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, eval_metric="logloss", verbosity=0
                )
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            auroc = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)

            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)
            fold_briers.append(brier)

        results[model_name] = {
            "AUROC_mean": np.mean(fold_aurocs),
            "AUROC_std": np.std(fold_aurocs),
            "AUPRC_mean": np.mean(fold_auprcs),
            "AUPRC_std": np.std(fold_auprcs),
            "Brier_mean": np.mean(fold_briers),
            "Brier_std": np.std(fold_briers),
            "fold_aurocs": fold_aurocs,
            "n_features": len(available_features),
        }
        print(f"  {model_name:25s}: AUROC={np.mean(fold_aurocs):.3f}±{np.std(fold_aurocs):.3f} "
              f"AUPRC={np.mean(fold_auprcs):.3f}±{np.std(fold_auprcs):.3f} "
              f"Brier={np.mean(fold_briers):.3f}")

    # Incremental AUROC
    print(f"\n  Incremental AUROC gains:")
    model_order = ["M1_Baseline", "M2_StaticSIC", "M3_Continuous", "M4_Phenotype", "M5_Extended_XGB"]
    for i in range(1, len(model_order)):
        prev = results[model_order[i-1]]["AUROC_mean"]
        curr = results[model_order[i]]["AUROC_mean"]
        delta = curr - prev
        print(f"  {model_order[i-1]} → {model_order[i]}: ΔAUROC = {delta:+.3f}")

    return results


# ==============================================================================
# Step 6: External Validation (eICU)
# ==============================================================================

def run_external_validation(df_mimic_cluster, cluster_features, best_k, phenotype_names, scaler_params):
    """External validation on eICU."""
    print("\n" + "="*70)
    print("STEP 6: External Validation (eICU)")
    print("="*70)

    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler

    # Load eICU data
    eicu_traj = pd.read_csv(EICU_DATA / "eicu_partial_trajectory_features.csv")
    eicu_outcomes = pd.read_csv(EICU_DATA / "eicu_outcomes.csv")

    print(f"  eICU trajectory features: {len(eicu_traj):,}")
    print(f"  eICU outcomes: {len(eicu_outcomes):,}")

    # Check columns
    print(f"  eICU traj columns: {list(eicu_traj.columns)}")

    # Merge
    eicu = eicu_traj.merge(eicu_outcomes, on=["patientunitstayid"], how="inner", suffixes=("", "_out"))
    print(f"  eICU merged: {len(eicu):,}")

    # Apply 72h landmark
    if "icu_los_days" in eicu.columns:
        eicu = eicu[eicu["icu_los_days"] >= 3].copy()
        print(f"  eICU 72h landmark: {len(eicu):,}")
    else:
        print("  WARNING: icu_los_days not found, skipping landmark")

    # Check mortality
    mort_col = None
    for col in ["hospital_mortality", "hospital_expire_flag", "expire_flag"]:
        if col in eicu.columns:
            mort_col = col
            break
    if mort_col:
        if eicu[mort_col].dtype == object:
            eicu["hospital_death"] = (eicu[mort_col] == "t").astype(int)
        else:
            eicu["hospital_death"] = eicu[mort_col].astype(int)
        print(f"  eICU mortality ({mort_col}): {eicu['hospital_death'].mean()*100:.1f}%")
    else:
        print("  WARNING: No mortality column found in eICU outcomes")
        return None

    # Project eICU patients onto MIMIC-IV cluster centroids
    # Use available trajectory features
    eicu_cluster_feats = []
    mimic_cluster_feats = []
    for f in cluster_features:
        if f in eicu.columns:
            eicu_cluster_feats.append(f)
            mimic_cluster_feats.append(f)
        elif f.replace("w1_", "w1_").replace("w2_", "w2_").replace("w3_", "w3_") in eicu.columns:
            alt = f.replace("w1_", "w1_").replace("w2_", "w2_").replace("w3_", "w3_")
            eicu_cluster_feats.append(alt)
            mimic_cluster_feats.append(f)

    print(f"  Matching cluster features: {len(eicu_cluster_feats)} / {len(cluster_features)}")

    if len(eicu_cluster_feats) < 3:
        print("  WARNING: Too few matching features for external clustering, skipping")
        return None

    eicu_valid = eicu.dropna(subset=eicu_cluster_feats).copy()
    print(f"  eICU with valid cluster features: {len(eicu_valid):,}")

    # Train/evaluate on the feature intersection available in both databases.
    X_eicu = eicu_valid[eicu_cluster_feats].values.astype(float)
    X_eicu = np.nan_to_num(X_eicu, nan=0.0)

    # Fit reference model on MIMIC-IV and predict on eICU
    # For now, train a simple logistic regression
    y_mimic = df_mimic_cluster["hospital_death"].values
    X_mimic = df_mimic_cluster[mimic_cluster_feats].values.astype(float)
    X_mimic = np.nan_to_num(X_mimic, nan=0.0)

    scaler_mimic = StandardScaler()
    X_mimic_scaled = scaler_mimic.fit_transform(X_mimic)
    X_eicu_scaled = scaler_mimic.transform(X_eicu)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_mimic_scaled, y_mimic)

    # Predict on eICU
    y_eicu = eicu_valid["hospital_death"].values
    y_prob_eicu = lr.predict_proba(X_eicu_scaled)[:, 1]

    auroc_eicu = roc_auc_score(y_eicu, y_prob_eicu)
    auprc_eicu = average_precision_score(y_eicu, y_prob_eicu)
    brier_eicu = brier_score_loss(y_eicu, y_prob_eicu)

    print(f"  External validation (frozen MIMIC-IV model on eICU):")
    print(f"    AUROC = {auroc_eicu:.3f}")
    print(f"    AUPRC = {auprc_eicu:.3f}")
    print(f"    Brier = {brier_eicu:.3f}")

    return {
        "AUROC": auroc_eicu,
        "AUPRC": auprc_eicu,
        "Brier": brier_eicu,
        "n_eicu": len(eicu_valid),
        "eicu_mortality_pct": eicu_valid["hospital_death"].mean() * 100,
    }


# ==============================================================================
# Step 7: Generate Output Tables
# ==============================================================================

def generate_tables(df_all, df_landmark, df_cluster, best_k, phenotype_names,
                    cluster_results, stability_results, model_results, external_results,
                    early_death):
    """Generate all tables for the paper."""
    print("\n" + "="*70)
    print("STEP 7: Generate Output Tables")
    print("="*70)

    tables = {}
    score_w1, score_w2, score_w3 = sic_score_columns()

    # --- Table I: Cohort Characteristics ---
    table1_data = {
        "Metric": [
            "MIMIC-IV source cohort",
            "72-hour landmark before W1 SIC filter",
            "72-hour landmark cohort",
            "Early deaths/discharges (<72h)",
            "Hospital mortality (landmark), %",
            "ICU mortality (landmark), %",
            "Median age, years",
            "Female, %",
            "Median ICU LOS, days",
        ],
        "Value": [
            f"{len(df_all):,}",
            f"{len(df_landmark):,}",
            f"{len(df_cluster):,}",
            f"{len(early_death):,}",
            f"{df_cluster['hospital_death'].mean()*100:.1f}",
            f"{df_cluster['icu_death'].mean()*100:.1f}",
            f"{df_cluster['anchor_age'].median():.0f}",
            f"{(df_cluster['gender']=='F').mean()*100:.1f}",
            f"{df_cluster['icu_los_days'].median():.1f}",
        ],
    }
    tables["table_I_cohort"] = pd.DataFrame(table1_data)

    # --- Table II: Phenotype Characteristics ---
    pheno_stats = df_cluster.groupby("phenotype_name").agg(
        n=("stay_id", "count"),
        age_median=("anchor_age", "median"),
        female_pct=("gender", lambda x: (x=="F").mean()*100),
        hospital_mortality=("hospital_death", "mean"),
        icu_mortality=("icu_death", "mean"),
        w1_sic=(score_w1, "mean"),
        w2_sic=(score_w2, "mean"),
        w3_sic=(score_w3, "mean"),
        icu_los=("icu_los_days", "median"),
    ).round(2)
    pheno_stats["pct_of_cohort"] = (pheno_stats["n"] / len(df_cluster) * 100).round(1)
    tables["table_II_phenotypes"] = pheno_stats

    # --- Table III: Mortality OR ---
    from scipy import stats
    ref_pheno = "Stable Low-Risk"
    if ref_pheno not in df_cluster["phenotype_name"].values:
        # Find lowest mortality phenotype
        ref_pheno = df_cluster.groupby("phenotype_name")["hospital_death"].mean().idxmin()

    or_results = []
    ref = df_cluster[df_cluster["phenotype_name"] == ref_pheno]
    for pheno in sorted(df_cluster["phenotype_name"].unique()):
        grp = df_cluster[df_cluster["phenotype_name"] == pheno]
        a = grp["hospital_death"].sum()
        b = len(grp) - a
        c = ref["hospital_death"].sum()
        d = len(ref) - c
        if b > 0 and d > 0 and a > 0:
            odds_ratio = (a/b) / (c/d)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_low = np.exp(np.log(odds_ratio) - 1.96*se)
            ci_high = np.exp(np.log(odds_ratio) + 1.96*se)
            # Chi-square test
            table_2x2 = np.array([[a, b], [c, d]])
            chi2, p_val = stats.chi2_contingency(table_2x2)[:2]
        else:
            odds_ratio, ci_low, ci_high, p_val = np.nan, np.nan, np.nan, np.nan
        or_results.append({
            "Phenotype": pheno,
            "n": len(grp),
            "Deaths": int(a) if not np.isnan(a) else 0,
            "Mortality %": grp["hospital_death"].mean()*100,
            "OR": odds_ratio,
            "95% CI": f"{ci_low:.2f}-{ci_high:.2f}" if not np.isnan(ci_low) else "N/A",
            "p-value": p_val,
        })
    tables["table_III_mortality_OR"] = pd.DataFrame(or_results)

    # --- Table IV: Model Performance ---
    model_table = []
    for name, res in model_results.items():
        model_table.append({
            "Model": name,
            "AUROC": f"{res['AUROC_mean']:.3f} ± {res['AUROC_std']:.3f}",
            "AUPRC": f"{res['AUPRC_mean']:.3f} ± {res['AUPRC_std']:.3f}",
            "Brier": f"{res['Brier_mean']:.3f}",
            "n_features": res["n_features"],
        })
    tables["table_IV_model_performance"] = pd.DataFrame(model_table)

    # Print summary
    for name, tbl in tables.items():
        print(f"\n  --- {name} ---")
        print(tbl.to_string(index=False) if isinstance(tbl, pd.DataFrame) else tbl)

    # Save to CSV
    for name, tbl in tables.items():
        if isinstance(tbl, pd.DataFrame):
            tbl.to_csv(OUTPUT_DIR / "tables" / f"{name}.csv", index=False)

    return tables


# ==============================================================================
# Step 8: Save Metrics JSON
# ==============================================================================

def save_metrics(best_k, cluster_results, stability_results, model_results, external_results,
                 phenotype_names, df_cluster, early_death):
    """Save all metrics as JSON for paper filling."""
    print("\n" + "="*70)
    print("STEP 8: Save Metrics JSON")
    print("="*70)

    metrics = {
        "analysis_date": datetime.now().isoformat(),
        "data_version": f"{ANALYSIS_MODE}_sic",
        "forced_k": FORCE_K,
        "cohort": {
            "total_adult_first_icu": 65366,
            "icd_sepsis_candidates": int(len(df_cluster) + len(early_death)),
            "landmark_72h_cohort": int(len(df_cluster)),
            "early_deaths_under_72h": int(len(early_death)),
            "landmark_hospital_mortality_pct": round(df_cluster["hospital_death"].mean()*100, 1),
            "landmark_icu_mortality_pct": round(df_cluster["icu_death"].mean()*100, 1),
        },
        "clustering": {
            "optimal_k": best_k,
            "silhouette_score": round(cluster_results[best_k]["silhouette"], 3),
            "calinski_harabasz": round(cluster_results[best_k]["calinski_harabasz"], 1),
            "davies_bouldin": round(cluster_results[best_k]["davies_bouldin"], 3),
            "bootstrap_ari_mean": round(stability_results["ari_mean"], 3),
            "bootstrap_ari_std": round(stability_results["ari_std"], 3),
            "phenotype_names": {str(int(k)): v for k, v in phenotype_names.items()},
            "cluster_sizes": {str(k): int(v) for k, v in cluster_results[best_k]["cluster_sizes"].items()},
        },
        "phenotype_mortality": {},
        "models": {},
        "external_validation": external_results if external_results else "skipped",
    }

    # Phenotype mortality
    score_w1, score_w2, score_w3 = sic_score_columns()
    for pheno in df_cluster["phenotype_name"].unique():
        grp = df_cluster[df_cluster["phenotype_name"] == pheno]
        metrics["phenotype_mortality"][pheno] = {
            "n": int(len(grp)),
            "pct_of_cohort": round(len(grp)/len(df_cluster)*100, 1),
            "hospital_mortality_pct": round(grp["hospital_death"].mean()*100, 1),
            "icu_mortality_pct": round(grp["icu_death"].mean()*100, 1),
            "w1_sic_mean": round(grp[score_w1].mean(), 2),
            "w2_sic_mean": round(grp[score_w2].mean(), 2),
            "w3_sic_mean": round(grp[score_w3].mean(), 2),
        }

    # Model results
    for name, res in model_results.items():
        metrics["models"][name] = {
            "AUROC": round(res["AUROC_mean"], 3),
            "AUROC_std": round(res["AUROC_std"], 3),
            "AUPRC": round(res["AUPRC_mean"], 3),
            "Brier": round(res["Brier_mean"], 3),
        }

    # Incremental AUROC
    model_order = ["M1_Baseline", "M2_StaticSIC", "M3_Continuous", "M4_Phenotype", "M5_Extended_XGB"]
    metrics["incremental_auroc"] = {}
    for i in range(1, len(model_order)):
        prev = model_results[model_order[i-1]]["AUROC_mean"]
        curr = model_results[model_order[i]]["AUROC_mean"]
        metrics["incremental_auroc"][f"{model_order[i-1]}_to_{model_order[i]}"] = round(curr - prev, 3)

    # Save
    output_path = OUTPUT_DIR / "metrics" / "analysis_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved to: {output_path}")

    return metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    global ANALYSIS_MODE, FORCE_K, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="SIC Trajectory Analysis")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full", action="store_true", help="Use full Sepsis-3/full SIC derived cohort")
    mode_group.add_argument("--partial", action="store_true", help="Use partial SIC scores")
    parser.add_argument("--force-k", type=int, default=None, help="Force K for the final clustering assignment")
    parser.add_argument("--output-tag", default=None, help="Write outputs to analysis_output_<tag>")
    parser.add_argument("--skip-external", action="store_true", help="Skip external validation")
    args = parser.parse_args()
    ANALYSIS_MODE = "full" if args.full else "partial"
    FORCE_K = args.force_k
    if args.output_tag:
        output_tag = args.output_tag
    elif ANALYSIS_MODE == "full":
        output_tag = f"full_k{FORCE_K}" if FORCE_K is not None else "full_optimal"
    else:
        output_tag = "partial"
    OUTPUT_DIR = _OUT / output_tag
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(exist_ok=True)

    print(f"SIC Trajectory Analysis Pipeline ({ANALYSIS_MODE})")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Step 1: Cohort assembly
    df_all, df_landmark, early_death = load_mimiciv_cohort(ANALYSIS_MODE)

    # Step 2: Feature engineering
    df_features = build_features(df_landmark, ANALYSIS_MODE)

    # Step 3: K-means clustering
    df_cluster, cluster_results, best_k, phenotype_names, scaler, cluster_features = run_clustering(df_features)
    df_cluster.to_csv(OUTPUT_DIR / "tables" / "cluster_assignments.csv", index=False)

    # Step 4: Bootstrap stability
    stability_results = bootstrap_stability(df_cluster, cluster_features, best_k, n_iter=200)

    # Step 5: Nested prediction models
    model_results = run_nested_models(df_cluster)

    # Step 6: External validation
    external_results = None
    if not args.skip_external:
        try:
            scaler_params = {"mean_": scaler.mean_, "scale_": scaler.scale_}
            external_results = run_external_validation(
                df_cluster, cluster_features, best_k, phenotype_names, scaler_params
            )
        except Exception as e:
            print(f"  External validation failed: {e}")
            external_results = None

    # Step 7: Generate tables
    tables = generate_tables(
        df_all, df_landmark, df_cluster, best_k, phenotype_names,
        cluster_results, stability_results, model_results, external_results,
        early_death
    )

    # Step 8: Save metrics
    metrics = save_metrics(
        best_k, cluster_results, stability_results, model_results, external_results,
        phenotype_names, df_cluster, early_death
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tables: {list(tables.keys())}")
    print(f"Metrics JSON: analysis_output/metrics/analysis_metrics.json")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
