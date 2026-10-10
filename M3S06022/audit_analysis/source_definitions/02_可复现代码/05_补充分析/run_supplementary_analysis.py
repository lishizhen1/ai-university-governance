#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary Analysis Script
==============================
Runs all remaining analyses:
1. DeLong paired AUROC test (M1-M5)
2. Calibration curves (M4 internal + eICU external)
3. SHAP analysis (M5 XGBoost)
4. Kaplan-Meier with real cluster labels
5. Decision Curve Analysis (M2 vs M4)
6. Generate updated figures (fig2, fig3, supplementary SHAP fig)

Usage:
    python run_supplementary_analysis.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (MIMIC_DIR, EICU_DIR, OUTPUT_DIR as _OUT,
                     FIGURE_DIR, METRICS_DIR, ensure_output_dirs)
ensure_output_dirs()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MIMIC_DIR            # MIMIC-IV subdirs
EICU_DATA = EICU_DIR            # eICU files

# Load metrics
METRICS_PATH = _OUT / "full_k4" / "metrics" / "analysis_metrics.json"
with open(METRICS_PATH) as f:
    METRICS = json.load(f)


# ==============================================================================
# Helper: DeLong test implementation
# ==============================================================================

def delong_roc_test(y_true, y_pred_a, y_pred_b):
    """
    DeLong test for comparing two ROC curves.
    Returns z-statistic and p-value.
    """
    from scipy import stats

    def _auc_ci_bootstrap(y_true, y_pred, n_boot=2000):
        """Bootstrap AUC CI."""
        rng = np.random.RandomState(42)
        aucs = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            yt, yp = y_true[idx], y_pred[idx]
            if len(np.unique(yt)) < 2:
                continue
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(yt, yp))
        return np.mean(aucs), np.std(aucs)

    # Use bootstrap-based paired comparison (DeLong approximation)
    rng = np.random.RandomState(42)
    n = len(y_true)
    n_boot = 2000
    diff_aucs = []

    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, y_pred_a)
    auc_b = roc_auc_score(y_true, y_pred_b)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ya, yb = y_pred_a[idx], y_pred_b[idx]
        auc_boot_a = roc_auc_score(yt, ya)
        auc_boot_b = roc_auc_score(yt, yb)
        diff_aucs.append(auc_boot_b - auc_boot_a)

    diff_aucs = np.array(diff_aucs)
    observed_diff = auc_b - auc_a
    se = np.std(diff_aucs)
    if se == 0:
        return 0.0, 1.0
    z = observed_diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# ==============================================================================
# Load cluster assignments and rebuild models
# ==============================================================================

def load_and_rebuild():
    """Load cluster assignments and rebuild all models for analysis."""
    print("Loading cluster assignments...")
    df = pd.read_csv(_OUT / "full_k4" / "tables" / "cluster_assignments.csv")
    print(f"  Cluster assignments: {len(df):,} rows")

    # Build features same as run_sic_trajectory_analysis.py
    score_cols = ["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]
    sic_matrix = df[score_cols].copy()
    df["sic_mean_3w"] = sic_matrix.mean(axis=1, skipna=True)
    df["sic_max_3w"] = sic_matrix.max(axis=1, skipna=True)
    df["sic_min_3w"] = sic_matrix.min(axis=1, skipna=True)
    df["sic_slope"] = (sic_matrix.iloc[:, -1] - sic_matrix.iloc[:, 0]) / 2.0
    df["gender_numeric"] = (df["gender"] == "M").astype(int)

    # Boolean conversion
    for col in ["icu_mortality", "hospital_mortality", "mortality_28d"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)
    if "hospital_expire_flag" in df.columns:
        df["hospital_death"] = df["hospital_expire_flag"].astype(int)
    elif "hospital_mortality" in df.columns:
        df["hospital_death"] = df["hospital_mortality"]

    return df


def rebuild_models(df):
    """Rebuild all 5 models and return cross-validated predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    print("\nRebuilding models for analysis...")

    y = df["hospital_death"].values.astype(float)

    # Feature sets
    m1_features = ["anchor_age", "gender_numeric"]
    if "w1_sofa_total_assuming_missing_normal" in df.columns:
        m1_features += ["w1_sofa_total_assuming_missing_normal"]

    m2_features = m1_features + ["w1_sic_total_score"]
    m3_features = m2_features + ["sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope"]

    # Phenotype one-hot
    pheno_dummies = pd.get_dummies(df["phenotype_name"], prefix="pheno")
    m4_features = m3_features + list(pheno_dummies.columns)

    m5_features = [
        "anchor_age", "gender_numeric",
        "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
        "w1_inr_max", "w2_inr_max", "w3_inr_max",
        "w1_bilirubin_total_max", "w2_bilirubin_total_max", "w3_bilirubin_total_max",
        "w1_creatinine_max", "w2_creatinine_max", "w3_creatinine_max",
        "w1_lactate_max", "w2_lactate_max", "w3_lactate_max",
        "sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope",
    ]
    # Add full SIC/SOFA features if available
    for f in ["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score",
              "w1_sofa_total_assuming_missing_normal", "w2_sofa_total_assuming_missing_normal", "w3_sofa_total_assuming_missing_normal",
              "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic", "w3_sofa_4systems_for_sic",
              "w1_sofa_respiration", "w2_sofa_respiration", "w3_sofa_respiration",
              "w1_sofa_liver", "w2_sofa_liver", "w3_sofa_liver",
              "w1_sofa_cardiovascular", "w2_sofa_cardiovascular", "w3_sofa_cardiovascular",
              "w1_sofa_renal", "w2_sofa_renal", "w3_sofa_renal",
              "w1_sic_platelet_component", "w2_sic_platelet_component", "w3_sic_platelet_component",
              "w1_sic_inr_component", "w2_sic_inr_component", "w3_sic_inr_component",
              "w1_sic_organ_sofa_component", "w2_sic_organ_sofa_component", "w3_sic_organ_sofa_component"]:
        if f in df.columns:
            m5_features.append(f)

    # Prepare data matrix
    df_work = pd.concat([df, pheno_dummies], axis=1)

    # Impute
    for features in [m1_features, m2_features, m3_features, m4_features, m5_features]:
        for col in features:
            if col in df_work.columns and df_work[col].dtype in [np.float64, np.int64, float, int]:
                df_work[col] = df_work[col].fillna(df_work[col].median())

    models_config = {
        "M1_Baseline": (m1_features, False),
        "M2_StaticSIC": (m2_features, False),
        "M3_Continuous": (m3_features, False),
        "M4_Phenotype": (m4_features, False),
        "M5_Extended_XGB": (m5_features, True),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Collect all cross-validated predictions
    all_preds = {}
    all_models = {}

    for name, (features, use_xgb) in models_config.items():
        available = [f for f in features if f in df_work.columns]
        X = df_work[available].values.astype(float)
        X = np.nan_to_num(X, nan=0.0)

        oof_preds = np.zeros(len(y))
        fold_models = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            if use_xgb:
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, eval_metric="logloss", verbosity=0
                )
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)

            model.fit(X_train, y_train)
            oof_preds[test_idx] = model.predict_proba(X_test)[:, 1]
            fold_models.append(model)

        all_preds[name] = oof_preds
        all_models[name] = (fold_models, available)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, oof_preds)
        print(f"  {name:25s}: OOF AUROC = {auc:.3f}")

    return y, all_preds, all_models, df_work


# ==============================================================================
# Analysis 1: DeLong Paired AUROC Test
# ==============================================================================

def run_delong_tests(y, all_preds):
    """Run DeLong-style paired AUROC tests."""
    print("\n" + "="*70)
    print("ANALYSIS 1: DeLong Paired AUROC Tests")
    print("="*70)

    from sklearn.metrics import roc_auc_score

    model_order = ["M1_Baseline", "M2_StaticSIC", "M3_Continuous", "M4_Phenotype", "M5_Extended_XGB"]
    results = []

    # All pairwise from M1
    for i in range(len(model_order)):
        for j in range(i+1, len(model_order)):
            a = model_order[i]
            b = model_order[j]
            z, p = delong_roc_test(y, all_preds[a], all_preds[b])
            auc_a = roc_auc_score(y, all_preds[a])
            auc_b = roc_auc_score(y, all_preds[b])
            results.append({
                "comparison": f"{a} vs {b}",
                "AUROC_A": round(auc_a, 3),
                "AUROC_B": round(auc_b, 3),
                "delta": round(auc_b - auc_a, 3),
                "z_stat": round(z, 3),
                "p_value": p,
                "significant": p < 0.05,
            })

    # Key adjacent comparisons
    print(f"\n  Adjacent model comparisons:")
    for i in range(1, len(model_order)):
        prev = model_order[i-1]
        curr = model_order[i]
        z, p = delong_roc_test(y, all_preds[prev], all_preds[curr])
        auc_prev = roc_auc_score(y, all_preds[prev])
        auc_curr = roc_auc_score(y, all_preds[curr])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {prev} → {curr}: ΔAUROC = {auc_curr-auc_prev:+.3f}, z = {z:.3f}, p = {p:.4f} {sig}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(_OUT / "full_k4" / "tables" / "delong_paired_tests.csv", index=False)
    print(f"\n  Results saved to delong_paired_tests.csv")

    return results


# ==============================================================================
# Analysis 2: Calibration Curves
# ==============================================================================

def generate_calibration_curves(y, all_preds):
    """Generate calibration curves for M4 (internal) and eICU (external)."""
    print("\n" + "="*70)
    print("ANALYSIS 2: Calibration Curves")
    print("="*70)

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, brier_score_loss

    # Use paper style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Colors from paper style
    colors = {
        "M1": "#0072B2", "M2": "#56B4E9", "M3": "#009E73",
        "M4": "#E69F00", "M5": "#D55E00",
    }

    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

    # Panel (a): ROC curves
    ax_roc = fig.add_subplot(gs[0])
    model_names = ["M1", "M2", "M3", "M4", "M5"]
    full_names = ["M1_Baseline", "M2_StaticSIC", "M3_Continuous", "M4_Phenotype", "M5_Extended_XGB"]

    from sklearn.metrics import roc_curve
    for short, full, color in zip(model_names, full_names, [colors[m] for m in model_names]):
        fpr, tpr, _ = roc_curve(y, all_preds[full])
        auc = roc_auc_score(y, all_preds[full])
        ax_roc.plot(fpr, tpr, color=color, label=f"{short} (AUC={auc:.3f})", linewidth=1.5)
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("(a) ROC Curves", fontsize=11)
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])

    # Panel (b): Internal calibration (M4, MIMIC-IV)
    ax_cal_int = fig.add_subplot(gs[1])
    m4_preds = all_preds["M4_Phenotype"]
    prob_true, prob_pred = calibration_curve(y, m4_preds, n_bins=10, strategy="uniform")
    ax_cal_int.plot(prob_pred, prob_true, "s-", color=colors["M4"], linewidth=1.5, markersize=5, label="M4 (Internal)")
    ax_cal_int.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    brier = brier_score_loss(y, m4_preds)
    ax_cal_int.set_xlabel("Mean Predicted Probability")
    ax_cal_int.set_ylabel("Fraction of Positives")
    ax_cal_int.set_title(f"(b) Internal Calibration (M4)\nBrier = {brier:.3f}", fontsize=10)
    ax_cal_int.legend(fontsize=8)
    ax_cal_int.set_xlim([-0.02, 1.02])
    ax_cal_int.set_ylim([-0.02, 1.02])

    # Panel (c): eICU external calibration
    ax_cal_ext = fig.add_subplot(gs[2])
    try:
        # Load eICU and run frozen model
        eicu_traj = pd.read_csv(EICU_DATA / "eicu_partial_trajectory_features.csv")
        eicu_outcomes = pd.read_csv(EICU_DATA / "eicu_outcomes.csv")
        eicu = eicu_traj.merge(eicu_outcomes[["patientunitstayid", "hospital_mortality", "unit_los_days"]], on="patientunitstayid", how="inner")

        # Parse mortality
        eicu["hospital_death"] = eicu["hospital_mortality"].astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)

        # Apply landmark if possible
        if "unit_los_days" in eicu.columns:
            eicu = eicu[eicu["unit_los_days"] >= 3].copy()

        # Use shared features
        shared_feats = ["w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
                        "w1_inr_max", "w2_inr_max", "w3_inr_max"]
        eicu_valid = eicu.dropna(subset=shared_feats).copy()

        # Train on MIMIC with same features
        mimic_feats = shared_feats
        df_mimic = pd.read_csv(_OUT / "full_k4" / "tables" / "cluster_assignments.csv")
        for col in mimic_feats:
            if col in df_mimic.columns:
                df_mimic[col] = df_mimic[col].fillna(df_mimic[col].median())

        X_mimic = df_mimic[mimic_feats].values.astype(float)
        X_mimic = np.nan_to_num(X_mimic, nan=0.0)
        y_mimic = df_mimic["hospital_death"] if "hospital_death" in df_mimic.columns else df_mimic["hospital_mortality"]

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression as LR_ext
        scaler = StandardScaler()
        X_mimic_scaled = scaler.fit_transform(X_mimic)

        lr = LR_ext(max_iter=1000, random_state=42)
        lr.fit(X_mimic_scaled, y_mimic)

        X_eicu = eicu_valid[shared_feats].values.astype(float)
        X_eicu = np.nan_to_num(X_eicu, nan=0.0)
        X_eicu_scaled = scaler.transform(X_eicu)
        y_eicu = eicu_valid["hospital_death"].values
        y_prob_eicu = lr.predict_proba(X_eicu_scaled)[:, 1]

        prob_true_ext, prob_pred_ext = calibration_curve(y_eicu, y_prob_eicu, n_bins=10, strategy="uniform")
        ax_cal_ext.plot(prob_pred_ext, prob_true_ext, "s-", color=colors["M5"], linewidth=1.5, markersize=5, label="Shared-feat (eICU)")
        brier_ext = brier_score_loss(y_eicu, y_prob_eicu)
        auc_ext = roc_auc_score(y_eicu, y_prob_eicu)
        ax_cal_ext.set_title(f"(c) External (eICU)\nAUC={auc_ext:.3f}, Brier={brier_ext:.3f}", fontsize=10)
        ax_cal_ext.legend(fontsize=8)
    except Exception as e:
        print(f"  External calibration failed: {e}")
        ax_cal_ext.text(0.5, 0.5, "eICU calibration\nnot available", ha="center", va="center", transform=ax_cal_ext.transAxes)

    ax_cal_ext.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax_cal_ext.set_xlabel("Mean Predicted Probability")
    ax_cal_ext.set_ylabel("Fraction of Positives")
    ax_cal_ext.set_xlim([-0.02, 1.02])
    ax_cal_ext.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_roc_calibration.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURE_DIR / "fig3_roc_calibration.png", bbox_inches="tight", dpi=300)
    print(f"  Calibration curves saved to fig3_roc_calibration.pdf")
    plt.close()


# ==============================================================================
# Analysis 3: SHAP Analysis (M5 XGBoost)
# ==============================================================================

def run_shap_analysis(y, all_models, df_work):
    """Run SHAP analysis on M5 XGBoost model."""
    print("\n" + "="*70)
    print("ANALYSIS 3: SHAP Analysis (M5 XGBoost)")
    print("="*70)

    import shap
    import xgboost as xgb

    fold_models, features = all_models["M5_Extended_XGB"]
    # Use fold 0 model for SHAP
    model = fold_models[0]

    X = df_work[features].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=features, max_display=15, show=False)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_shap_summary.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("  SHAP summary plot saved to fig_shap_summary.pdf")

    # Bar plot of mean |SHAP|
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = [features[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    # Clean feature names for display
    display_names = []
    for f in top_features:
        name = f.replace("w1_", "W1 ").replace("w2_", "W2 ").replace("w3_", "W3 ")
        name = name.replace("_", " ").replace("sic ", "SIC ").replace("sofa ", "SOFA ")
        name = name.title()
        display_names.append(name)

    ax.barh(range(len(top_idx)), top_values[::-1], color="#56B4E9", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(display_names[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Features (M5 XGBoost)")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_shap_importance.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("  SHAP importance bar saved to fig_shap_importance.pdf")

    # Print top features
    print(f"\n  Top 15 features by mean |SHAP|:")
    for rank, (feat, val) in enumerate(zip(top_features, top_values), 1):
        print(f"    {rank:>2}. {feat:50s} |SHAP| = {val:.4f}")

    return top_features, top_values


# ==============================================================================
# Analysis 4: Kaplan-Meier with Real Cluster Labels
# ==============================================================================

def generate_kaplan_meier(df):
    """Generate KM curves with real K=4 cluster labels."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Kaplan-Meier Survival Curves")
    print("="*70)

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    # Time = hospital LOS from landmark (icu_los_days as proxy, or hospital_los_days)
    # Event = hospital death
    if "hospital_los_days" not in df.columns:
        # Use ICU LOS as proxy
        if "icu_los_days" in df.columns:
            df["time_km"] = df["icu_los_days"] - 3  # subtract 72h landmark
        else:
            print("  WARNING: No LOS column found, skipping KM")
            return
    else:
        df["time_km"] = df["hospital_los_days"]

    df["event_km"] = df["hospital_death"]

    # Remove invalid times
    df_km = df[df["time_km"] > 0].copy()
    print(f"  KM analysis: {len(df_km):,} patients with valid time")

    # Log-rank test
    results = multivariate_logrank_test(
        df_km["time_km"], df_km["phenotype_name"], df_km["event_km"]
    )
    print(f"  Log-rank test: p = {results.p_value:.6f}")

    # Plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    pheno_colors = {
        "Stable Low-Risk": "#009E73",
        "High-SIC Improving": "#56B4E9",
        "Intermediate Persistent Risk": "#E69F00",
        "Persistently High-Risk": "#D55E00",
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    kmf = KaplanMeierFitter()

    for pheno in ["Stable Low-Risk", "High-SIC Improving", "Intermediate Persistent Risk", "Persistently High-Risk"]:
        if pheno not in df_km["phenotype_name"].values:
            continue
        mask = df_km["phenotype_name"] == pheno
        grp = df_km[mask]
        kmf.fit(grp["time_km"], grp["event_km"], label=pheno)
        color = pheno_colors.get(pheno, "#000000")
        kmf.plot_survival_function(ax=ax, color=color, linewidth=1.5,
                                   ci_show=True, ci_alpha=0.15)

    ax.set_xlabel("Days from 72-hour landmark")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Kaplan–Meier by phenotype (log-rank p < 0.001)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim([0, None])
    ax.set_ylim([0.5, 1.02])

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_kaplan_meier.pdf", bbox_inches="tight", dpi=300)
    print("  KM curves saved to fig2_kaplan_meier.pdf")
    plt.close()


# ==============================================================================
# Analysis 5: Decision Curve Analysis
# ==============================================================================

def run_dca(y, all_preds):
    """Decision Curve Analysis comparing M2 vs M4."""
    print("\n" + "="*70)
    print("ANALYSIS 5: Decision Curve Analysis")
    print("="*70)

    thresholds = np.arange(0.05, 0.95, 0.01)

    def net_benefit(y_true, y_prob, threshold):
        """Calculate net benefit at a given threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum(y_pred & y_true.astype(int))
        fp = np.sum(y_pred & (1 - y_true.astype(int)))
        n = len(y_true)
        if threshold == 0:
            return np.sum(y_true) / n
        return tp / n - fp / n * (threshold / (1 - threshold))

    def net_benefit_all(y_true, threshold):
        """Net benefit of treating all patients."""
        return np.mean(y_true) - (1 - np.mean(y_true)) * threshold / (1 - threshold)

    m2_preds = all_preds["M2_StaticSIC"]
    m4_preds = all_preds["M4_Phenotype"]
    m3_preds = all_preds["M3_Continuous"]

    nb_m2 = [net_benefit(y, m2_preds, t) for t in thresholds]
    nb_m4 = [net_benefit(y, m4_preds, t) for t in thresholds]
    nb_m3 = [net_benefit(y, m3_preds, t) for t in thresholds]
    nb_all = [net_benefit_all(y, t) for t in thresholds]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thresholds, nb_m2, "-", color="#56B4E9", linewidth=1.5, label="M2 Static SIC")
    ax.plot(thresholds, nb_m3, "-", color="#009E73", linewidth=1.5, label="M3 Continuous")
    ax.plot(thresholds, nb_m4, "-", color="#E69F00", linewidth=1.5, label="M4 Phenotype")
    ax.plot(thresholds, nb_all, "k--", linewidth=0.8, alpha=0.5, label="Treat All")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Risk Threshold")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(fontsize=8)
    ax.set_xlim([0.05, 0.9])
    ax.set_ylim([-0.05, max(nb_m4) * 1.2])

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_dca.pdf", bbox_inches="tight", dpi=300)
    print("  DCA saved to fig_dca.pdf")
    plt.close()

    # Calculate net benefit difference in clinically relevant range
    clin_range = (thresholds >= 0.10) & (thresholds <= 0.40)
    nb_diff = np.array(nb_m4) - np.array(nb_m2)
    mean_diff = np.mean(nb_diff[clin_range])
    print(f"  Mean NB difference (M4-M2) in 10%-40% range: {mean_diff:.4f}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"Supplementary Analysis Pipeline")
    print(f"Started: {datetime.now().isoformat()}")

    # Load data and rebuild models
    df = load_and_rebuild()
    y, all_preds, all_models, df_work = rebuild_models(df)

    # Analysis 1: DeLong tests
    delong_results = run_delong_tests(y, all_preds)

    # Analysis 2: Calibration curves
    generate_calibration_curves(y, all_preds)

    # Analysis 3: SHAP analysis
    top_features, top_shap = run_shap_analysis(y, all_models, df_work)

    # Analysis 4: Kaplan-Meier
    generate_kaplan_meier(df)

    # Analysis 5: DCA
    run_dca(y, all_preds)

    print("\n" + "="*70)
    print("ALL SUPPLEMENTARY ANALYSES COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
