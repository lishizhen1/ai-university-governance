from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MIMIC_DIR, EICU_DIR, OUTPUT_DIR as _OUT, ensure_output_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_PATH = _OUT / "full_k4" / "tables" / "cluster_assignments.csv"
EICU_PATH = EICU_DIR / "08_full_sofa_sepsis3_eicu" / "eicu_sic_trajectory_features_full_sofa.csv"
OUTPUT_DIR = _OUT / "full_k4" / "external_validation_eicu_full_sofa"


CLUSTER_FEATURES = [
    "w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score",
    "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
    "w1_inr_max", "w2_inr_max", "w3_inr_max",
    "w1_sofa_4systems_for_sic", "w2_sofa_4systems_for_sic", "w3_sofa_4systems_for_sic",
    "w1_sofa_respiration", "w2_sofa_respiration", "w3_sofa_respiration",
    "w1_sofa_cardiovascular", "w2_sofa_cardiovascular", "w3_sofa_cardiovascular",
    "w1_sofa_renal", "w2_sofa_renal", "w3_sofa_renal",
]


M5_FEATURES = [
    "anchor_age", "gender_numeric",
    "w1_platelet_min", "w2_platelet_min", "w3_platelet_min",
    "w1_inr_max", "w2_inr_max", "w3_inr_max",
    "w1_bilirubin_total_max", "w2_bilirubin_total_max", "w3_bilirubin_total_max",
    "w1_creatinine_max", "w2_creatinine_max", "w3_creatinine_max",
    "w1_lactate_max", "w2_lactate_max", "w3_lactate_max",
    "sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope",
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


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "t", "1", "yes"])


def prepare_mimic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gender_numeric"] = out["gender"].astype(str).eq("M").astype(int)
    out["hospital_death"] = pd.to_numeric(out["hospital_death"], errors="coerce").fillna(0).astype(int)
    return out


def prepare_eicu(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[bool_series(out["sepsis_text_flag"])].copy()
    out = out[pd.to_numeric(out["unit_los_days"], errors="coerce").ge(3)].copy()
    out = out[out["w1_sic_total_score"].notna()].copy()
    out["anchor_age"] = pd.to_numeric(out["age_years"], errors="coerce")
    out["gender_numeric"] = out["gender"].astype(str).str.lower().eq("male").astype(int)
    out["hospital_death"] = bool_series(out["hospital_mortality"]).astype(int)
    return out


def fill_cluster_features(df: pd.DataFrame, medians: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    for col in CLUSTER_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else np.nan
        if col.startswith("w2_"):
            w1 = col.replace("w2_", "w1_", 1)
            if w1 in out.columns:
                out[col] = out[col].fillna(out[w1])
        if col.startswith("w3_"):
            w1 = col.replace("w3_", "w1_", 1)
            if w1 in out.columns:
                out[col] = out[col].fillna(out[w1])
    if medians is None:
        medians = {col: float(out[col].median()) for col in CLUSTER_FEATURES}
    for col in CLUSTER_FEATURES:
        out[col] = out[col].fillna(medians[col])
    return out, medians


def assign_eicu_phenotypes(mimic: pd.DataFrame, eicu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    mimic_filled, medians = fill_cluster_features(mimic)
    eicu_filled, _ = fill_cluster_features(eicu, medians)

    scaler = StandardScaler()
    x_mimic = scaler.fit_transform(mimic_filled[CLUSTER_FEATURES].values.astype(float))
    x_eicu = scaler.transform(eicu_filled[CLUSTER_FEATURES].values.astype(float))
    km = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
    mimic_labels = km.fit_predict(x_mimic)
    eicu_labels = km.predict(x_eicu)

    cross = pd.crosstab(pd.Series(mimic_labels, name="cluster"), mimic_filled["phenotype_name"])
    label_map = cross.idxmax(axis=1).to_dict()
    mimic_filled["external_cluster"] = mimic_labels
    mimic_filled["external_phenotype_name"] = [label_map[x] for x in mimic_labels]
    eicu_filled["external_cluster"] = eicu_labels
    eicu_filled["external_phenotype_name"] = [label_map[x] for x in eicu_labels]
    return mimic_filled, eicu_filled, {str(k): v for k, v in label_map.items()}


def feature_sets(mimic: pd.DataFrame) -> dict[str, list[str]]:
    dummies = pd.get_dummies(mimic["external_phenotype_name"], prefix="pheno")
    m1 = ["anchor_age", "gender_numeric", "w1_sofa_total_assuming_missing_normal"]
    m2 = m1 + ["w1_sic_total_score"]
    m3 = m2 + ["sic_mean_3w", "sic_max_3w", "sic_min_3w", "sic_slope"]
    m4 = m3 + list(dummies.columns)
    return {
        "M1_Baseline": m1,
        "M2_StaticSIC": m2,
        "M3_Continuous": m3,
        "M4_Phenotype": m4,
        "M5_Extended_XGB": M5_FEATURES,
    }


def add_phenotype_dummies(mimic: pd.DataFrame, eicu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic_d = pd.get_dummies(mimic["external_phenotype_name"], prefix="pheno")
    eicu_d = pd.get_dummies(eicu["external_phenotype_name"], prefix="pheno")
    for col in mimic_d.columns:
        if col not in eicu_d.columns:
            eicu_d[col] = 0
    eicu_d = eicu_d[mimic_d.columns]
    return pd.concat([mimic, mimic_d], axis=1), pd.concat([eicu, eicu_d], axis=1)


def train_and_validate(mimic: pd.DataFrame, eicu: pd.DataFrame, model_features: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import xgboost as xgb

    y_train = mimic["hospital_death"].astype(int).values
    y_test = eicu["hospital_death"].astype(int).values
    rows = []
    pred_out = eicu[["patientunitstayid", "external_cluster", "external_phenotype_name", "hospital_death"]].copy()

    for model_name, features in model_features.items():
        available = [f for f in features if f in mimic.columns and f in eicu.columns]
        train = mimic[available].apply(pd.to_numeric, errors="coerce")
        test = eicu[available].apply(pd.to_numeric, errors="coerce")
        medians = train.median(numeric_only=True)
        train = train.fillna(medians).fillna(0)
        test = test.fillna(medians).fillna(0)

        if model_name == "M5_Extended_XGB":
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(train.values.astype(float), y_train)
        prob = model.predict_proba(test.values.astype(float))[:, 1]
        pred_out[f"{model_name}_prob"] = prob
        rows.append({
            "model": model_name,
            "n_features": len(available),
            "n_eicu": len(eicu),
            "eicu_mortality_pct": float(y_test.mean() * 100),
            "AUROC": float(roc_auc_score(y_test, prob)),
            "AUPRC": float(average_precision_score(y_test, prob)),
            "Brier": float(brier_score_loss(y_test, prob)),
        })
    return pd.DataFrame(rows), pred_out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mimic = prepare_mimic(pd.read_csv(MIMIC_PATH, low_memory=False))
    eicu = prepare_eicu(pd.read_csv(EICU_PATH, low_memory=False))
    mimic, eicu, label_map = assign_eicu_phenotypes(mimic, eicu)
    mimic, eicu = add_phenotype_dummies(mimic, eicu)
    metrics, predictions = train_and_validate(mimic, eicu, feature_sets(mimic))

    phenotype_summary = (
        eicu.groupby("external_phenotype_name")
        .agg(
            n=("patientunitstayid", "size"),
            hospital_mortality=("hospital_death", "mean"),
            w1_sic=("w1_sic_total_score", "mean"),
            w2_sic=("w2_sic_total_score", "mean"),
            w3_sic=("w3_sic_total_score", "mean"),
        )
        .reset_index()
    )
    phenotype_summary["pct"] = phenotype_summary["n"] / len(eicu) * 100
    phenotype_summary["hospital_mortality_pct"] = phenotype_summary["hospital_mortality"] * 100

    eicu.to_csv(OUTPUT_DIR / "eicu_full_sofa_external_cohort.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "eicu_full_sofa_external_model_performance.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "eicu_full_sofa_external_predictions.csv", index=False)
    phenotype_summary.to_csv(OUTPUT_DIR / "eicu_full_sofa_projected_phenotypes.csv", index=False)
    payload = {
        "analysis_date": datetime.now().isoformat(),
        "cohort": {
            "eicu_text_sepsis_landmark_w1_full_sic": int(len(eicu)),
            "eicu_hospital_mortality_pct": float(eicu["hospital_death"].mean() * 100),
        },
        "cluster_label_map": label_map,
        "models": metrics.set_index("model").to_dict(orient="index"),
    }
    (OUTPUT_DIR / "eicu_full_sofa_external_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
