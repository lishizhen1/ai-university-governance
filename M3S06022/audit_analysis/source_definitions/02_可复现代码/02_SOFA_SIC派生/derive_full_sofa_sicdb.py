from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SICDB_RAW, SICDB_DIR, META_DIR, OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT_CANDIDATES = [SICDB_RAW]
OUTPUT_ROOT = OUTPUT_DIR / "data_processed" / "09_full_sofa_sicdb"
METADATA_ROOT = META_DIR

SECONDS_PER_DAY = 24 * 60 * 60
WINDOWS = {
    "w1_0_24h": (0, SECONDS_PER_DAY),
    "w2_24_48h": (SECONDS_PER_DAY, 2 * SECONDS_PER_DAY),
    "w3_48_72h": (2 * SECONDS_PER_DAY, 3 * SECONDS_PER_DAY),
}
CHUNKSIZE = 1_000_000


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def raw_root() -> Path:
    for root in RAW_ROOT_CANDIDATES:
        if (root / "cases.csv.gz").exists():
            return root
    raise FileNotFoundError("Cannot find sicDB-1.0.8 raw CSV directory.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def assign_window_seconds(offset_seconds: pd.Series) -> pd.Series:
    return pd.cut(
        offset_seconds,
        bins=[-0.001, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY, 3 * SECONDS_PER_DAY],
        labels=["w1_0_24h", "w2_24_48h", "w3_48_72h"],
        right=False,
    ).astype("object")


def merge_agg(parts: list[pd.DataFrame], value_col: str, agg: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["caseid", "window_name", value_col])
    df = pd.concat(parts, ignore_index=True)
    grouped = df.groupby(["caseid", "window_name"], as_index=False)[value_col]
    if agg == "min":
        return grouped.min()
    if agg == "max":
        return grouped.max()
    if agg == "sum":
        return grouped.sum()
    raise ValueError(f"Unsupported aggregation: {agg}")


def score_respiration(pf: pd.Series, vent: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pf.index)
    out.loc[pf.notna() & (pf >= 400)] = 0
    out.loc[pf.notna() & (pf < 400)] = 1
    out.loc[pf.notna() & (pf < 300)] = 2
    out.loc[pf.notna() & (pf < 200) & vent.fillna(False)] = 3
    out.loc[pf.notna() & (pf < 100) & vent.fillna(False)] = 4
    return out


def score_platelet(x: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=x.index)
    out.loc[x.notna() & (x >= 150)] = 0
    out.loc[x.notna() & (x < 150)] = 1
    out.loc[x.notna() & (x < 100)] = 2
    out.loc[x.notna() & (x < 50)] = 3
    out.loc[x.notna() & (x < 20)] = 4
    return out


def score_liver(x: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=x.index)
    out.loc[x.notna() & (x < 1.2)] = 0
    out.loc[x.notna() & (x >= 1.2)] = 1
    out.loc[x.notna() & (x >= 2.0)] = 2
    out.loc[x.notna() & (x >= 6.0)] = 3
    out.loc[x.notna() & (x >= 12.0)] = 4
    return out


def score_cardio(map_min: pd.Series, vaso_score: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=map_min.index)
    out.loc[map_min.notna() & (map_min >= 70)] = 0
    out.loc[map_min.notna() & (map_min < 70)] = 1
    out = pd.concat([out, vaso_score], axis=1).max(axis=1, skipna=True)
    return out.where(out.notna(), np.nan)


def score_renal(creatinine: pd.Series, urine_ml: pd.Series) -> pd.Series:
    creat = pd.Series(np.nan, index=creatinine.index)
    creat.loc[creatinine.notna() & (creatinine < 1.2)] = 0
    creat.loc[creatinine.notna() & (creatinine >= 1.2)] = 1
    creat.loc[creatinine.notna() & (creatinine >= 2.0)] = 2
    creat.loc[creatinine.notna() & (creatinine >= 3.5)] = 3
    creat.loc[creatinine.notna() & (creatinine >= 5.0)] = 4

    urine = pd.Series(np.nan, index=urine_ml.index)
    urine.loc[urine_ml.notna() & (urine_ml >= 500)] = 0
    urine.loc[urine_ml.notna() & (urine_ml < 500)] = 3
    urine.loc[urine_ml.notna() & (urine_ml < 200)] = 4
    return pd.concat([creat, urine], axis=1).max(axis=1, skipna=True)


def sic_platelet_component(platelet: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=platelet.index)
    out.loc[platelet.notna() & (platelet >= 150)] = 0
    out.loc[platelet.notna() & (platelet < 150)] = 1
    out.loc[platelet.notna() & (platelet < 100)] = 2
    return out


def sic_inr_component(inr: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=inr.index)
    out.loc[inr.notna() & (inr <= 1.2)] = 0
    out.loc[inr.notna() & (inr > 1.2)] = 1
    out.loc[inr.notna() & (inr > 1.4)] = 2
    return out


def sic_organ_component(sofa4: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=sofa4.index)
    out.loc[sofa4.notna() & (sofa4 == 0)] = 0
    out.loc[sofa4.notna() & (sofa4 == 1)] = 1
    out.loc[sofa4.notna() & (sofa4 >= 2)] = 2
    return out


def load_cohort(root: Path) -> pd.DataFrame:
    log("load sicDB adult cohort")
    c = pd.read_csv(root / "cases.csv.gz", dtype=str)
    c.columns = [col.lower() for col in c.columns]
    c["caseid"] = c["caseid"].astype(str)
    c["patientid"] = c["patientid"].astype(str)
    c["age_years"] = numeric(c["ageonadmission"])
    c["weight_kg"] = numeric(c["weightonadmission"])
    c.loc[c["weight_kg"] > 300, "weight_kg"] = c.loc[c["weight_kg"] > 300, "weight_kg"] / 1000.0
    c["height_cm"] = numeric(c["heightonadmission"])
    c["saps3"] = numeric(c["saps3"])
    c["timeofstay_seconds"] = numeric(c["timeofstay"])
    c["icu_los_days"] = c["timeofstay_seconds"] / SECONDS_PER_DAY
    c["hospitalstaydays"] = numeric(c["hospitalstaydays"])
    c["offsetofdeath_seconds"] = numeric(c["offsetofdeath"])
    c["estimatedsurvivalobservationtime_days"] = numeric(c["estimatedsurvivalobservationtime"])
    c["sex_label"] = c["sex"].map({"735": "male", "736": "female", "737": "unknown"}).fillna("unknown")
    c["gender_numeric"] = c["sex_label"].eq("male").astype(int)
    c["sepsis_admission_form_flag"] = c["admissionformhassepsis"].eq("740")
    text = (
        c["icd10main"].fillna("") + " " + c["icd10maintext"].fillna("") + " " + c["diagnosist2"].fillna("")
    ).str.lower()
    c["sepsis_text_flag"] = text.str.contains("sepsis|septic", regex=True)
    c["sepsis_candidate_flag"] = c["sepsis_admission_form_flag"] | c["sepsis_text_flag"]
    c["hospital_mortality"] = c["hospitaldischargetype"].eq("3130")
    c["observed_mortality"] = c["offsetofdeath_seconds"].notna()
    c["mortality_28d"] = c["offsetofdeath_seconds"].le(28 * SECONDS_PER_DAY).fillna(False)
    c["mortality_90d"] = c["offsetofdeath_seconds"].le(90 * SECONDS_PER_DAY).fillna(False)
    c["hospital_death"] = c["hospital_mortality"].astype(int)
    c["landmark_72h_flag"] = c["timeofstay_seconds"].ge(3 * SECONDS_PER_DAY)

    keep = [
        "caseid", "patientid", "age_years", "sex", "sex_label", "gender_numeric",
        "weight_kg", "height_cm", "hospitalunit", "referringunit", "icd10main",
        "icd10maintext", "diagnosist2", "surgicalsite", "admissionurgency",
        "admissionformhassepsis", "sepsis_admission_form_flag", "sepsis_text_flag",
        "sepsis_candidate_flag", "surgicaladmissiontype", "saps3", "timeofstay_seconds",
        "icu_los_days", "icuoffset", "hospitaldischargetype", "dischargestate",
        "hospitalstaydays", "offsetofdeath_seconds", "estimatedsurvivalobservationtime_days",
        "hospital_mortality", "observed_mortality", "mortality_28d", "mortality_90d",
        "hospital_death", "landmark_72h_flag",
    ]
    return c.loc[c["age_years"].ge(18), keep].drop_duplicates("caseid")


def extract_labs(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan sicDB laboratory")
    mapping: dict[int, tuple[str, str, tuple[float, float]]] = {
        314: ("platelet_min", "min", (1, 2000)),
        315: ("platelet_min", "min", (1, 2000)),
        3128: ("inr_max", "max", (0.5, 20)),
        333: ("bilirubin_total_max", "max", (0, 60)),
        367: ("creatinine_max", "max", (0, 30)),
        368: ("creatinine_max", "max", (0, 30)),
        454: ("lactate_max", "max", (0, 50)),
        465: ("lactate_max", "max", (0, 50)),
        657: ("lactate_max", "max", (0, 50)),
        444: ("pao2_min", "min", (20, 700)),
        664: ("pao2_min", "min", (20, 700)),
        689: ("pao2_min", "min", (20, 700)),
        684: ("fio2_lab_max", "max", (0.21, 100)),
        3135: ("pao2_fio2_ratio_direct_min", "min", (20, 2000)),
    }
    parts: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for col, agg, _ in mapping.values():
        parts.setdefault((col, agg), [])

    usecols = ["CaseID", "LaboratoryID", "Offset", "LaboratoryValue"]
    for chunk in pd.read_csv(root / "laboratory.csv.gz", usecols=usecols, dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["CaseID"].isin(ids)].copy()
        if chunk.empty:
            continue
        chunk["laboratoryid_num"] = numeric(chunk["LaboratoryID"])
        chunk = chunk[chunk["laboratoryid_num"].isin(mapping)]
        if chunk.empty:
            continue
        chunk["offset_seconds"] = numeric(chunk["Offset"])
        chunk["value"] = numeric(chunk["LaboratoryValue"])
        chunk = chunk[chunk["offset_seconds"].between(0, 3 * SECONDS_PER_DAY - 1) & chunk["value"].notna()]
        if chunk.empty:
            continue
        chunk["caseid"] = chunk["CaseID"].astype(str)
        chunk["window_name"] = assign_window_seconds(chunk["offset_seconds"])
        for lab_id, (out_col, agg, bounds) in mapping.items():
            sub = chunk[chunk["laboratoryid_num"].eq(lab_id)].copy()
            if sub.empty:
                continue
            lo, hi = bounds
            sub = sub[sub["value"].between(lo, hi)]
            if sub.empty:
                continue
            sub = sub.rename(columns={"value": out_col})
            parts[(out_col, agg)].append(sub[["caseid", "window_name", out_col]])

    merged = None
    for (col, agg), frames in parts.items():
        part = merge_agg(frames, col, agg)
        merged = part if merged is None else merged.merge(part, on=["caseid", "window_name"], how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["caseid", "window_name"])


def extract_signals(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan sicDB hourly signals")
    relevant = {703, 706, 725, 773, 2278, 2282, 2283, 2284, 2307}
    parts: dict[str, list[pd.DataFrame]] = {
        "map_min": [],
        "urine_output_ml": [],
        "fio2_signal_max": [],
        "mechanical_ventilation_proxy": [],
        "vasopressor_sofa_score_signal": [],
    }
    usecols = ["CaseID", "DataID", "Offset", "Val"]
    for chunk in pd.read_csv(root / "data_float_h.csv.gz", usecols=usecols, dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["CaseID"].isin(ids)].copy()
        if chunk.empty:
            continue
        chunk["dataid_num"] = numeric(chunk["DataID"])
        chunk = chunk[chunk["dataid_num"].isin(relevant)]
        if chunk.empty:
            continue
        chunk["offset_seconds"] = numeric(chunk["Offset"])
        chunk["value"] = numeric(chunk["Val"])
        chunk = chunk[chunk["offset_seconds"].between(0, 3 * SECONDS_PER_DAY - 1) & chunk["value"].notna()]
        if chunk.empty:
            continue
        chunk["caseid"] = chunk["CaseID"].astype(str)
        chunk["window_name"] = assign_window_seconds(chunk["offset_seconds"])

        maps = chunk[chunk["dataid_num"].isin([703, 706]) & chunk["value"].between(20, 200)].copy()
        if not maps.empty:
            maps = maps.rename(columns={"value": "map_min"})
            parts["map_min"].append(maps[["caseid", "window_name", "map_min"]])

        urine = chunk[chunk["dataid_num"].eq(725) & chunk["value"].between(0, 10000)].copy()
        if not urine.empty:
            urine = urine.rename(columns={"value": "urine_output_ml"})
            parts["urine_output_ml"].append(urine[["caseid", "window_name", "urine_output_ml"]])

        fio2 = chunk[chunk["dataid_num"].eq(2283) & chunk["value"].between(0.21, 100)].copy()
        if not fio2.empty:
            fio2 = fio2.rename(columns={"value": "fio2_signal_max"})
            parts["fio2_signal_max"].append(fio2[["caseid", "window_name", "fio2_signal_max"]])

        vent = chunk[
            (
                chunk["dataid_num"].isin([2278, 2284]) & chunk["value"].gt(0)
            )
            | (
                chunk["dataid_num"].eq(2282) & chunk["value"].gt(0)
            )
            | (
                chunk["dataid_num"].eq(2283) & chunk["value"].gt(21)
            )
        ].copy()
        if not vent.empty:
            vent["mechanical_ventilation_proxy"] = True
            parts["mechanical_ventilation_proxy"].append(
                vent[["caseid", "window_name", "mechanical_ventilation_proxy"]]
            )

        vaso = chunk[chunk["dataid_num"].isin([773, 2307]) & chunk["value"].gt(0)].copy()
        if not vaso.empty:
            vaso["dose_mcg_kg_min"] = vaso["value"] * 1_000_000.0 / 60.0
            vaso["vasopressor_sofa_score_signal"] = np.nan
            norepi = vaso["dataid_num"].eq(773)
            dobutamine = vaso["dataid_num"].eq(2307)
            vaso.loc[dobutamine, "vasopressor_sofa_score_signal"] = 2
            vaso.loc[norepi & vaso["dose_mcg_kg_min"].gt(0), "vasopressor_sofa_score_signal"] = 3
            vaso.loc[norepi & vaso["dose_mcg_kg_min"].gt(0.1), "vasopressor_sofa_score_signal"] = 4
            parts["vasopressor_sofa_score_signal"].append(
                vaso[["caseid", "window_name", "vasopressor_sofa_score_signal"]]
            )

    out = None
    specs = {
        "map_min": "min",
        "urine_output_ml": "sum",
        "fio2_signal_max": "max",
        "mechanical_ventilation_proxy": "max",
        "vasopressor_sofa_score_signal": "max",
    }
    for col, agg in specs.items():
        part = merge_agg(parts[col], col, agg)
        out = part if out is None else out.merge(part, on=["caseid", "window_name"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["caseid", "window_name"])


def extract_medication_vasopressors(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan sicDB medication vasopressor intervals")
    official_drugs = {
        1502: "epinephrine",
        1559: "dobutamine",
        1562: "norepinephrine",
        1618: "dopamine",
    }
    records: list[dict[str, object]] = []
    usecols = ["CaseID", "DrugID", "Offset", "OffsetDrugEnd", "IsSingleDose"]
    for chunk in pd.read_csv(root / "medication.csv.gz", usecols=usecols, dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["CaseID"].isin(ids)].copy()
        if chunk.empty:
            continue
        chunk["drugid_num"] = numeric(chunk["DrugID"])
        chunk = chunk[chunk["drugid_num"].isin(official_drugs)]
        if chunk.empty:
            continue
        chunk["start"] = numeric(chunk["Offset"])
        chunk["end"] = numeric(chunk["OffsetDrugEnd"])
        chunk["end"] = chunk["end"].fillna(chunk["start"] + 3600)
        chunk.loc[chunk["end"].le(chunk["start"]), "end"] = chunk["start"] + 3600
        chunk = chunk[chunk["start"].lt(3 * SECONDS_PER_DAY) & chunk["end"].gt(0)]
        if chunk.empty:
            continue
        for caseid, drug_id, start, end, is_single in chunk[
            ["CaseID", "drugid_num", "start", "end", "IsSingleDose"]
        ].itertuples(index=False):
            drug = official_drugs[int(drug_id)]
            for window_name, (w_start, w_end) in WINDOWS.items():
                if start < w_end and end > w_start:
                    if drug == "dobutamine":
                        score = 2
                    elif drug == "dopamine":
                        score = 2
                    else:
                        score = 3
                    records.append({
                        "caseid": str(caseid),
                        "window_name": window_name,
                        "vasopressor_sofa_score_medication": score,
                        "vasopressor_medication_any": True,
                        "vasopressor_single_dose": str(is_single) == "1",
                    })
    if not records:
        return pd.DataFrame(columns=["caseid", "window_name", "vasopressor_sofa_score_medication"])
    df = pd.DataFrame(records)
    return df.groupby(["caseid", "window_name"], as_index=False).agg(
        vasopressor_sofa_score_medication=("vasopressor_sofa_score_medication", "max"),
        vasopressor_medication_any=("vasopressor_medication_any", "max"),
    )


def build_scores(cohort: pd.DataFrame, labs: pd.DataFrame, signals: pd.DataFrame, medication: pd.DataFrame) -> pd.DataFrame:
    log("assemble sicDB SOFA/SIC scores")
    base = pd.MultiIndex.from_product(
        [cohort["caseid"].astype(str).unique(), list(WINDOWS)],
        names=["caseid", "window_name"],
    ).to_frame(index=False)
    df = base.merge(labs, on=["caseid", "window_name"], how="left")
    df = df.merge(signals, on=["caseid", "window_name"], how="left")
    df = df.merge(medication, on=["caseid", "window_name"], how="left")

    fio2 = df.get("fio2_signal_max", pd.Series(np.nan, index=df.index)).combine_first(
        df.get("fio2_lab_max", pd.Series(np.nan, index=df.index))
    )
    df["fio2_fraction"] = fio2.where(fio2 <= 1.0, fio2 / 100.0)
    df.loc[df["fio2_fraction"].lt(0.21) | df["fio2_fraction"].gt(1.0), "fio2_fraction"] = np.nan
    computed_pf = df["pao2_min"] / df["fio2_fraction"]
    direct_pf = df.get("pao2_fio2_ratio_direct_min", pd.Series(np.nan, index=df.index))
    df["pao2_fio2_ratio"] = pd.concat([computed_pf, direct_pf], axis=1).min(axis=1, skipna=True)
    df["pao2_fio2_ratio"] = df["pao2_fio2_ratio"].where(df["pao2_fio2_ratio"].between(20, 2000))

    vent = df.get("mechanical_ventilation_proxy", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    vaso_score = pd.concat(
        [
            df.get("vasopressor_sofa_score_signal", pd.Series(np.nan, index=df.index)),
            df.get("vasopressor_sofa_score_medication", pd.Series(np.nan, index=df.index)),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    df["sofa_respiration"] = score_respiration(df["pao2_fio2_ratio"], vent)
    df["sofa_coagulation"] = score_platelet(df["platelet_min"])
    df["sofa_liver"] = score_liver(df["bilirubin_total_max"])
    df["sofa_cardiovascular"] = score_cardio(df.get("map_min", pd.Series(np.nan, index=df.index)), vaso_score)
    df["sofa_cns"] = np.nan
    df["sofa_renal"] = score_renal(df.get("creatinine_max", pd.Series(np.nan, index=df.index)), df.get("urine_output_ml", pd.Series(np.nan, index=df.index)))

    sofa_cols = [
        "sofa_respiration", "sofa_coagulation", "sofa_liver",
        "sofa_cardiovascular", "sofa_cns", "sofa_renal",
    ]
    df["sofa_components_available"] = df[sofa_cols].notna().sum(axis=1)
    df["sofa_total_assuming_missing_normal"] = df[sofa_cols].fillna(0).sum(axis=1)
    df["sofa_total_complete_only"] = df[sofa_cols].sum(axis=1, min_count=6)
    df["sofa_4systems_for_sic"] = df[["sofa_respiration", "sofa_liver", "sofa_cardiovascular", "sofa_renal"]].fillna(0).sum(axis=1)

    df["sic_platelet_component"] = sic_platelet_component(df["platelet_min"])
    df["sic_inr_component"] = sic_inr_component(df["inr_max"])
    df["sic_organ_sofa_component"] = sic_organ_component(df["sofa_4systems_for_sic"])
    df["sic_total_score"] = df[["sic_platelet_component", "sic_inr_component", "sic_organ_sofa_component"]].sum(axis=1, min_count=3)
    df["sic_flag"] = (df["sic_total_score"] >= 4) & ((df["sic_platelet_component"] + df["sic_inr_component"]) >= 3)
    df["cns_sofa_unavailable"] = True

    return df.merge(cohort, on="caseid", how="left")


def build_trajectory(scores: pd.DataFrame) -> pd.DataFrame:
    log("build sicDB wide trajectory table")
    value_cols = [
        "platelet_min", "inr_max", "bilirubin_total_max", "creatinine_max", "lactate_max",
        "pao2_fio2_ratio", "sic_flag", "sic_platelet_component", "sic_inr_component",
        "sic_organ_sofa_component", "sic_total_score", "sofa_4systems_for_sic",
        "sofa_respiration", "sofa_coagulation", "sofa_liver", "sofa_cardiovascular",
        "sofa_renal", "sofa_cns", "sofa_total_assuming_missing_normal",
        "sofa_total_complete_only", "sofa_components_available",
    ]
    wide = scores.pivot_table(index="caseid", columns="window_name", values=value_cols, aggfunc="first")
    wide.columns = [
        f"{window.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{col}"
        for col, window in wide.columns
    ]
    wide = wide.reset_index()
    cohort_cols = [
        "caseid", "patientid", "age_years", "sex_label", "gender_numeric", "weight_kg", "height_cm",
        "saps3", "icu_los_days", "hospitalstaydays", "sepsis_admission_form_flag",
        "sepsis_text_flag", "sepsis_candidate_flag", "hospital_mortality",
        "observed_mortality", "mortality_28d", "mortality_90d", "hospital_death",
        "landmark_72h_flag",
    ]
    cohort = scores[cohort_cols].drop_duplicates("caseid")
    out = cohort.merge(wide, on="caseid", how="left")
    out["sic_w3_minus_w1"] = out["w3_sic_total_score"] - out["w1_sic_total_score"]
    out["sofa_w3_minus_w1"] = out["w3_sofa_total_assuming_missing_normal"] - out["w1_sofa_total_assuming_missing_normal"]
    out["sic_mean_3w"] = out[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].mean(axis=1)
    out["sic_max_3w"] = out[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].max(axis=1)
    out["sic_min_3w"] = out[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].min(axis=1)
    out["n_windows_available"] = out[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].notna().sum(axis=1)
    out["sic_slope"] = (out["w3_sic_total_score"] - out["w1_sic_total_score"]) / 2.0
    out["cns_sofa_unavailable"] = True
    return out


def export_outputs(cohort: pd.DataFrame, scores: pd.DataFrame, traj: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT_ROOT / "sicdb_adult_cases_cohort.csv", index=False)
    cohort[cohort["sepsis_admission_form_flag"]].to_csv(
        OUTPUT_ROOT / "sicdb_admission_form_sepsis_cohort.csv", index=False
    )
    scores.to_csv(OUTPUT_ROOT / "sicdb_full_sofa_sic_scores_0_72h.csv", index=False)
    traj.to_csv(OUTPUT_ROOT / "sicdb_sic_trajectory_features_full_sofa.csv", index=False)
    traj[traj["sepsis_admission_form_flag"]].to_csv(
        OUTPUT_ROOT / "sicdb_admission_form_sepsis_sic_trajectory_features.csv", index=False
    )
    landmark = traj[traj["landmark_72h_flag"] & traj["w1_sic_total_score"].notna()].copy()
    landmark.to_csv(OUTPUT_ROOT / "sicdb_landmark_full_sic_trajectory_features.csv", index=False)
    landmark[landmark["sepsis_admission_form_flag"]].to_csv(
        OUTPUT_ROOT / "sicdb_admission_form_sepsis_landmark_full_sic_trajectory_features.csv",
        index=False,
    )

    mapping = pd.DataFrame([
        {"source_table": "laboratory", "source_id": "314,315", "variable": "platelet_min", "aggregation": "min"},
        {"source_table": "laboratory", "source_id": "3128", "variable": "inr_max", "aggregation": "max"},
        {"source_table": "laboratory", "source_id": "333", "variable": "bilirubin_total_max", "aggregation": "max"},
        {"source_table": "laboratory", "source_id": "367,368", "variable": "creatinine_max", "aggregation": "max"},
        {"source_table": "laboratory", "source_id": "454,465,657", "variable": "lactate_max", "aggregation": "max"},
        {"source_table": "laboratory", "source_id": "444,664,689", "variable": "pao2_min", "aggregation": "min"},
        {"source_table": "laboratory", "source_id": "684", "variable": "fio2_lab_max", "aggregation": "max"},
        {"source_table": "laboratory", "source_id": "3135", "variable": "pao2_fio2_ratio_direct_min", "aggregation": "min"},
        {"source_table": "data_float_h", "source_id": "703,706", "variable": "map_min", "aggregation": "min"},
        {"source_table": "data_float_h", "source_id": "725", "variable": "urine_output_ml", "aggregation": "sum"},
        {"source_table": "data_float_h", "source_id": "2283", "variable": "fio2_signal_max", "aggregation": "max"},
        {"source_table": "data_float_h", "source_id": "2278,2282,2283,2284", "variable": "mechanical_ventilation_proxy", "aggregation": "any"},
        {"source_table": "data_float_h", "source_id": "773,2307", "variable": "vasopressor_sofa_score_signal", "aggregation": "max"},
        {"source_table": "medication", "source_id": "1502,1559,1562,1618", "variable": "vasopressor_sofa_score_medication", "aggregation": "max"},
        {"source_table": "not available", "source_id": "", "variable": "sofa_cns", "aggregation": "CNS/GCS not available in public sicDB tables"},
    ])
    mapping.to_csv(METADATA_ROOT / "sicdb_full_sofa_variable_mapping.csv", index=False)

    summary = {
        "analysis_date": datetime.now().isoformat(),
        "adult_cases": int(len(cohort)),
        "admission_form_sepsis": int(cohort["sepsis_admission_form_flag"].sum()),
        "text_or_admission_sepsis": int(cohort["sepsis_candidate_flag"].sum()),
        "landmark_72h_with_w1_full_sic": int(len(landmark)),
        "admission_form_sepsis_landmark_72h_with_w1_full_sic": int(
            (landmark["sepsis_admission_form_flag"]).sum()
        ),
        "score_rows": int(len(scores)),
        "w1_full_sic_available": int(traj["w1_sic_total_score"].notna().sum()),
        "w1_sofa_observed_components_median": float(traj["w1_sofa_components_available"].median(skipna=True)),
        "cns_sofa_status": "not available in public sicDB tables; sofa_total_assuming_missing_normal treats CNS as 0 and sofa_total_complete_only remains missing",
        "offset_unit": "seconds",
    }
    (METADATA_ROOT / "sicdb_full_sofa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(METADATA_ROOT / "sicdb_full_sofa_summary.csv", index=False)
    log(json.dumps(summary, indent=2))


def main() -> None:
    root = raw_root()
    log(f"raw root: {root}")
    cohort = load_cohort(root)
    ids = set(cohort["caseid"].astype(str))
    labs = extract_labs(root, ids)
    signals = extract_signals(root, ids)
    medication = extract_medication_vasopressors(root, ids)
    scores = build_scores(cohort, labs, signals, medication)
    traj = build_trajectory(scores)
    export_outputs(cohort, scores, traj)


if __name__ == "__main__":
    main()
