from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EICU_RAW, EICU_DIR, META_DIR, OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT_CANDIDATES = [EICU_RAW]
OUTPUT_ROOT = OUTPUT_DIR / "data_processed" / "08_full_sofa_sepsis3_eicu"
METADATA_ROOT = META_DIR

WINDOWS = {
    "w1_0_24h": (0, 1440),
    "w2_24_48h": (1440, 2880),
    "w3_48_72h": (2880, 4320),
}

CHUNKSIZE = 750_000


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def raw_root() -> Path:
    for root in RAW_ROOT_CANDIDATES:
        if (root / "patient.csv.gz").exists():
            return root
    raise FileNotFoundError("Cannot find eICU raw CSV directory.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def assign_window(offset_min: pd.Series) -> pd.Series:
    return pd.cut(
        offset_min,
        bins=[-0.001, 1440, 2880, 4320],
        labels=["w1_0_24h", "w2_24_48h", "w3_48_72h"],
        right=False,
    ).astype("object")


def merge_agg(parts: list[pd.DataFrame], value_col: str, agg: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["patientunitstayid", "window_name", value_col])
    df = pd.concat(parts, ignore_index=True)
    grouped = df.groupby(["patientunitstayid", "window_name"], as_index=False)[value_col]
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


def score_cns(gcs: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=gcs.index)
    out.loc[gcs.notna() & (gcs >= 15)] = 0
    out.loc[gcs.notna() & (gcs < 15)] = 1
    out.loc[gcs.notna() & (gcs < 13)] = 2
    out.loc[gcs.notna() & (gcs < 10)] = 3
    out.loc[gcs.notna() & (gcs < 6)] = 4
    return out


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
    log("load adult first ICU cohort")
    p = pd.read_csv(root / "patient.csv.gz", dtype=str)
    p["age_years"] = p["age"].replace({"> 89": "90"}).pipe(numeric)
    p["unitvisitnumber_num"] = numeric(p["unitvisitnumber"])
    p["unit_los_days"] = numeric(p["unitdischargeoffset"]) / 1440.0
    p["hospital_los_days"] = numeric(p["hospitaldischargeoffset"]) / 1440.0
    p["admissionweight_num"] = numeric(p["admissionweight"])
    p["hospital_mortality"] = p["hospitaldischargestatus"].eq("Expired")
    p["unit_mortality"] = p["unitdischargestatus"].eq("Expired")
    p["sepsis_text_flag"] = (
        p["apacheadmissiondx"].fillna("").str.lower().str.contains("sepsis|septic", regex=True)
    )
    p = p[(p["age_years"] >= 18) & (p["unitvisitnumber_num"].fillna(1) == 1)].copy()
    keep = [
        "patientunitstayid", "patienthealthsystemstayid", "uniquepid", "gender",
        "age_years", "ethnicity", "hospitalid", "wardid", "apacheadmissiondx",
        "admissionheight", "admissionweight_num", "unitvisitnumber", "unitstaytype",
        "unit_los_days", "hospital_los_days", "unit_mortality", "hospital_mortality",
        "sepsis_text_flag",
    ]
    return p[keep].drop_duplicates("patientunitstayid")


def add_diagnosis_sepsis(root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    ids = set(cohort["patientunitstayid"])
    flags: set[str] = set()
    log("scan diagnosis sepsis text")
    for chunk in pd.read_csv(root / "diagnosis.csv.gz", usecols=["patientunitstayid", "diagnosisstring", "icd9code"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        text = (chunk["diagnosisstring"].fillna("") + " " + chunk["icd9code"].fillna("")).str.lower()
        flags.update(chunk.loc[text.str.contains("sepsis|septic", regex=True), "patientunitstayid"].unique())
    cohort["sepsis_text_flag"] = cohort["sepsis_text_flag"] | cohort["patientunitstayid"].isin(flags)
    return cohort


def extract_labs(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan labs")
    mapping = {
        "platelets x 1000": ("platelet_min", "min"),
        "pt - inr": ("inr_max", "max"),
        "total bilirubin": ("bilirubin_total_max", "max"),
        "creatinine": ("creatinine_max", "max"),
        "lactate": ("lactate_max", "max"),
        "pao2": ("pao2_min", "min"),
        "fio2": ("fio2_lab_max", "max"),
    }
    parts: dict[tuple[str, str], list[pd.DataFrame]] = {(v, agg): [] for v, agg in mapping.values()}
    for chunk in pd.read_csv(root / "lab.csv.gz", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["labresultoffset"])
        chunk["value"] = numeric(chunk["labresult"])
        chunk["lab_lower"] = chunk["labname"].str.lower()
        chunk = chunk[chunk["offset"].between(0, 4319) & chunk["value"].notna() & chunk["lab_lower"].isin(mapping)]
        if chunk.empty:
            continue
        chunk["window_name"] = assign_window(chunk["offset"])
        for lab_lower, (out_col, agg) in mapping.items():
            sub = chunk[chunk["lab_lower"].eq(lab_lower)][["patientunitstayid", "window_name", "value"]].copy()
            if not sub.empty:
                sub = sub.rename(columns={"value": out_col})
                parts[(out_col, agg)].append(sub)

    merged = None
    for (col, agg), frames in parts.items():
        part = merge_agg(frames, col, agg)
        merged = part if merged is None else merged.merge(part, on=["patientunitstayid", "window_name"], how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["patientunitstayid", "window_name"])


def extract_resp(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan respiratory charting")
    fio2_parts: list[pd.DataFrame] = []
    vent_parts: list[pd.DataFrame] = []
    fio2_labels = {"fio2", "fio2 (%)", "o2 percentage", "set fraction of inspired oxygen (fio2)"}
    for chunk in pd.read_csv(root / "respiratoryCharting.csv.gz", usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["respchartoffset"])
        chunk = chunk[chunk["offset"].between(0, 4319)]
        if chunk.empty:
            continue
        chunk["window_name"] = assign_window(chunk["offset"])
        label = chunk["respchartvaluelabel"].fillna("").str.lower()
        value_text = chunk["respchartvalue"].fillna("").str.lower()

        fio2 = chunk[label.isin(fio2_labels)].copy()
        if not fio2.empty:
            fio2["fio2_resp_max"] = numeric(fio2["respchartvalue"].str.extract(r"([-+]?\d*\.?\d+)")[0])
            fio2 = fio2[fio2["fio2_resp_max"].between(0.21, 100)]
            fio2_parts.append(fio2[["patientunitstayid", "window_name", "fio2_resp_max"]])

        vent_mask = (
            label.eq("rt vent on/off") & ~value_text.str.contains("off|discontinued|no", regex=True)
        ) | label.str.contains(
            r"mechanical ventilator|ventilator checks|ventilator support|vent rate|tidal volume \(set\)|peep$|peep/cpap",
            regex=True,
        )
        vent = chunk.loc[vent_mask, ["patientunitstayid", "window_name"]].copy()
        if not vent.empty:
            vent["invasive_ventilation"] = True
            vent_parts.append(vent)

    fio2_df = merge_agg(fio2_parts, "fio2_resp_max", "max")
    vent_df = merge_agg(vent_parts, "invasive_ventilation", "max")
    return fio2_df.merge(vent_df, on=["patientunitstayid", "window_name"], how="outer")


def extract_map(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan MAP sources")
    parts: list[pd.DataFrame] = []
    for filename, offset_col, value_cols in [
        ("vitalPeriodic.csv.gz", "observationoffset", ["systemicmean"]),
        ("vitalAperiodic.csv.gz", "observationoffset", ["noninvasivemean"]),
    ]:
        for chunk in pd.read_csv(root / filename, usecols=["patientunitstayid", offset_col, *value_cols], dtype=str, chunksize=CHUNKSIZE):
            chunk = chunk[chunk["patientunitstayid"].isin(ids)]
            if chunk.empty:
                continue
            chunk["offset"] = numeric(chunk[offset_col])
            chunk["map_min"] = numeric(chunk[value_cols[0]])
            chunk = chunk[chunk["offset"].between(0, 4319) & chunk["map_min"].between(20, 200)]
            if chunk.empty:
                continue
            chunk["window_name"] = assign_window(chunk["offset"])
            parts.append(chunk[["patientunitstayid", "window_name", "map_min"]])

    for chunk in pd.read_csv(root / "nurseCharting.csv.gz", usecols=["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"], dtype=str, chunksize=CHUNKSIZE):
        label = (chunk["nursingchartcelltypevallabel"].fillna("") + " " + chunk["nursingchartcelltypevalname"].fillna("")).str.lower()
        chunk = chunk[chunk["patientunitstayid"].isin(ids) & label.str.contains(r"\bmap\b|arterial line map", regex=True)]
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["nursingchartoffset"])
        chunk["map_min"] = numeric(chunk["nursingchartvalue"].str.extract(r"([-+]?\d*\.?\d+)")[0])
        chunk = chunk[chunk["offset"].between(0, 4319) & chunk["map_min"].between(20, 200)]
        if chunk.empty:
            continue
        chunk["window_name"] = assign_window(chunk["offset"])
        parts.append(chunk[["patientunitstayid", "window_name", "map_min"]])
    return merge_agg(parts, "map_min", "min")


def extract_gcs(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan GCS")
    total_parts: list[pd.DataFrame] = []
    component_parts: dict[str, list[pd.DataFrame]] = {"gcs_eye": [], "gcs_motor": [], "gcs_verbal": []}
    for chunk in pd.read_csv(root / "nurseCharting.csv.gz", usecols=["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        if chunk.empty:
            continue
        label = (chunk["nursingchartcelltypevallabel"].fillna("") + " " + chunk["nursingchartcelltypevalname"].fillna("")).str.lower()
        mask = label.str.contains("glasgow coma score|gcs total|score \\(glasgow coma scale\\)|eyes|motor|verbal", regex=True)
        chunk = chunk[mask].copy()
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["nursingchartoffset"])
        chunk["value_num"] = numeric(chunk["nursingchartvalue"].str.extract(r"([-+]?\d*\.?\d+)")[0])
        chunk = chunk[chunk["offset"].between(0, 4319) & chunk["value_num"].notna()]
        if chunk.empty:
            continue
        chunk["window_name"] = assign_window(chunk["offset"])
        label = (chunk["nursingchartcelltypevallabel"].fillna("") + " " + chunk["nursingchartcelltypevalname"].fillna("")).str.lower()

        total = chunk[label.str.contains("gcs total|score \\(glasgow coma scale\\)", regex=True)].copy()
        total = total[total["value_num"].between(3, 15)]
        if not total.empty:
            total = total.rename(columns={"value_num": "gcs_total"})
            total_parts.append(total[["patientunitstayid", "window_name", "gcs_total"]])

        component_specs = [
            ("gcs_eye", r"\beyes\b|best eye|eye opening", 1, 4),
            ("gcs_motor", r"\bmotor\b|best motor", 1, 6),
            ("gcs_verbal", r"\bverbal\b|best verbal", 1, 5),
        ]
        for col, pat, lo, hi in component_specs:
            sub = chunk[label.str.contains(pat, regex=True) & chunk["value_num"].between(lo, hi)].copy()
            if not sub.empty:
                sub = sub.rename(columns={"value_num": col})
                component_parts[col].append(sub[["patientunitstayid", "window_name", col]])

    total_df = merge_agg(total_parts, "gcs_total", "min")
    comp_df = None
    for col, frames in component_parts.items():
        part = merge_agg(frames, col, "min")
        comp_df = part if comp_df is None else comp_df.merge(part, on=["patientunitstayid", "window_name"], how="outer")
    if comp_df is None:
        comp_df = pd.DataFrame(columns=["patientunitstayid", "window_name", "gcs_eye", "gcs_motor", "gcs_verbal"])
    comp_df["gcs_component_total"] = comp_df[["gcs_eye", "gcs_motor", "gcs_verbal"]].sum(axis=1, min_count=3)
    out = total_df.merge(comp_df[["patientunitstayid", "window_name", "gcs_component_total"]], on=["patientunitstayid", "window_name"], how="outer")
    out["gcs_min"] = out["gcs_total"].combine_first(out["gcs_component_total"])
    return out[["patientunitstayid", "window_name", "gcs_min"]]


def extract_urine(root: Path, ids: set[str]) -> pd.DataFrame:
    log("scan urine output")
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(root / "intakeOutput.csv.gz", usecols=["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel", "cellvaluenumeric"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        if chunk.empty:
            continue
        text = (chunk["cellpath"].fillna("") + " " + chunk["celllabel"].fillna("")).str.lower()
        urine_mask = text.str.contains("urine|foley|urinary catheter|voided amount|urethral catheter|external urinary", regex=True)
        bad_mask = text.str.contains("count|occurrence|bodyweight|stool", regex=True)
        chunk = chunk[urine_mask & ~bad_mask].copy()
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["intakeoutputoffset"])
        chunk["urine_output_ml"] = numeric(chunk["cellvaluenumeric"])
        chunk = chunk[chunk["offset"].between(0, 4319) & chunk["urine_output_ml"].between(0, 10000)]
        if chunk.empty:
            continue
        chunk["window_name"] = assign_window(chunk["offset"])
        parts.append(chunk[["patientunitstayid", "window_name", "urine_output_ml"]])
    return merge_agg(parts, "urine_output_ml", "sum")


def extract_vasopressors(root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    log("scan vasopressors")
    weight = cohort.set_index("patientunitstayid")["admissionweight_num"].to_dict()
    parts: list[pd.DataFrame] = []
    vaso_pat = re.compile(r"norepinephrine|levophed|epinephrine|adrenaline|dopamine|dobutamine", re.I)
    for chunk in pd.read_csv(root / "infusionDrug.csv.gz", usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate", "patientweight"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(weight.keys())]
        if chunk.empty:
            continue
        names = chunk["drugname"].fillna("")
        chunk = chunk[names.str.contains(vaso_pat)].copy()
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["infusionoffset"])
        chunk["rate"] = numeric(chunk["drugrate"].str.extract(r"([-+]?\d*\.?\d+)")[0])
        chunk["weight"] = numeric(chunk["patientweight"]).fillna(chunk["patientunitstayid"].map(weight))
        chunk = chunk[chunk["offset"].between(0, 4319) & chunk["rate"].notna()]
        if chunk.empty:
            continue
        lname = chunk["drugname"].fillna("").str.lower()
        dose = chunk["rate"].copy()
        mcg_min = lname.str.contains("mcg/min", regex=False) & ~lname.str.contains("mcg/kg/min", regex=False)
        dose.loc[mcg_min & chunk["weight"].gt(0)] = dose.loc[mcg_min & chunk["weight"].gt(0)] / chunk.loc[mcg_min & chunk["weight"].gt(0), "weight"]

        score = pd.Series(0, index=chunk.index, dtype=float)
        dopamine = lname.str.contains("dopamine")
        dobutamine = lname.str.contains("dobutamine")
        norepi = lname.str.contains("norepinephrine|levophed", regex=True)
        epi = lname.str.contains("epinephrine|adrenaline", regex=True)
        score.loc[dobutamine] = 2
        score.loc[dopamine & dose.le(5)] = 2
        score.loc[dopamine & dose.gt(5) & dose.le(15)] = 3
        score.loc[dopamine & dose.gt(15)] = 4
        score.loc[(norepi | epi) & dose.le(0.1)] = 3
        score.loc[(norepi | epi) & dose.gt(0.1)] = 4
        chunk["vasopressor_sofa_score"] = score.replace(0, np.nan)
        chunk["window_name"] = assign_window(chunk["offset"])
        parts.append(chunk[["patientunitstayid", "window_name", "vasopressor_sofa_score"]])
    return merge_agg(parts, "vasopressor_sofa_score", "max")


def extract_apache(root: Path, ids: set[str]) -> pd.DataFrame:
    log("load APACHE first-day variables")
    usecols = ["patientunitstayid", "intubated", "vent", "dialysis", "eyes", "motor", "verbal", "urine", "meanbp", "creatinine", "bilirubin", "pao2", "fio2"]
    a = pd.read_csv(root / "apacheApsVar.csv.gz", usecols=usecols, dtype=str)
    a = a[a["patientunitstayid"].isin(ids)].copy()
    for col in usecols[1:]:
        a[col] = numeric(a[col]).replace(-1, np.nan)
    a["window_name"] = "w1_0_24h"
    a["apache_gcs_total"] = a[["eyes", "motor", "verbal"]].sum(axis=1, min_count=3)
    a["apache_vent"] = (a["vent"].fillna(0).astype(float) > 0) | (a["intubated"].fillna(0).astype(float) > 0)
    return a.rename(columns={
        "meanbp": "apache_map_min",
        "creatinine": "apache_creatinine_max",
        "bilirubin": "apache_bilirubin_total_max",
        "pao2": "apache_pao2_min",
        "fio2": "apache_fio2_max",
        "urine": "apache_urine_output_ml",
    })[[
        "patientunitstayid", "window_name", "apache_map_min", "apache_creatinine_max",
        "apache_bilirubin_total_max", "apache_pao2_min", "apache_fio2_max",
        "apache_urine_output_ml", "apache_gcs_total", "apache_vent",
    ]]


def extract_suspected_infection(root: Path, ids: set[str]) -> pd.DataFrame:
    log("derive suspected infection")
    abx_regex = re.compile(
        r"cef|zosyn|piperacillin|tazobactam|vancomycin|vanco|meropenem|imipenem|ertapenem|azithro|"
        r"levofloxacin|ciprofloxacin|moxifloxacin|metronidazole|flagyl|clindamycin|linezolid|"
        r"daptomycin|gentamicin|tobramycin|amikacin|ampicillin|penicillin|nafcillin|oxacillin|"
        r"cefepime|ceftriaxone|ceftazidime|cefazolin|cefuroxime|ceftaroline|bactrim|trimethoprim|"
        r"sulfamethoxazole|doxycycline|tigecycline|fluconazole|micafungin|caspofungin|voriconazole",
        re.I,
    )
    abx: dict[str, list[tuple[float, str]]] = {}
    for chunk in pd.read_csv(root / "medication.csv.gz", usecols=["patientunitstayid", "drugstartoffset", "drugorderoffset", "drugname"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["drugname"].fillna("").str.contains(abx_regex)].copy()
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["drugstartoffset"]).combine_first(numeric(chunk["drugorderoffset"]))
        chunk = chunk[chunk["offset"].between(-2880, 4319)]
        for pid, off, drug in chunk[["patientunitstayid", "offset", "drugname"]].itertuples(index=False):
            abx.setdefault(pid, []).append((float(off), str(drug)))

    cultures: dict[str, list[tuple[float, str]]] = {}
    for chunk in pd.read_csv(root / "microLab.csv.gz", usecols=["patientunitstayid", "culturetakenoffset", "culturesite"], dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        if chunk.empty:
            continue
        chunk["offset"] = numeric(chunk["culturetakenoffset"])
        chunk = chunk[chunk["offset"].between(-2880, 4319)]
        for pid, off, site in chunk[["patientunitstayid", "offset", "culturesite"]].itertuples(index=False):
            cultures.setdefault(pid, []).append((float(off), str(site)))

    records = []
    for pid, abx_events in abx.items():
        best = None
        for a_off, drug in abx_events:
            for c_off, site in cultures.get(pid, []):
                if c_off >= a_off - 4320 and c_off <= a_off + 1440:
                    suspected = min(a_off, c_off)
                    near = -2880 <= suspected <= 1440
                    candidate = (0 if near else 1, suspected, a_off, c_off, drug, site)
                    if best is None or candidate < best:
                        best = candidate
        if best is not None:
            _, suspected, a_off, c_off, drug, site = best
            records.append({
                "patientunitstayid": pid,
                "suspected_infection_offset_min": suspected,
                "suspected_infection_hour_from_icu": suspected / 60.0,
                "suspected_infection_near_icu_admit": -2880 <= suspected <= 1440,
                "antibiotic_offset_min": a_off,
                "culture_offset_min": c_off,
                "antibiotic_drug": drug,
                "culture_site": site,
            })
    return pd.DataFrame(records)


def build_scores(cohort: pd.DataFrame, pieces: list[pd.DataFrame]) -> pd.DataFrame:
    log("assemble SOFA and SIC scores")
    ids = cohort["patientunitstayid"].astype(str).unique()
    base = pd.MultiIndex.from_product([ids, list(WINDOWS)], names=["patientunitstayid", "window_name"]).to_frame(index=False)
    df = base
    for piece in pieces:
        if piece is not None and not piece.empty:
            df = df.merge(piece, on=["patientunitstayid", "window_name"], how="left")

    # APACHE fallback for W1.
    for col, apache_col in [
        ("map_min", "apache_map_min"),
        ("creatinine_max", "apache_creatinine_max"),
        ("bilirubin_total_max", "apache_bilirubin_total_max"),
        ("pao2_min", "apache_pao2_min"),
        ("fio2_lab_max", "apache_fio2_max"),
        ("urine_output_ml", "apache_urine_output_ml"),
        ("gcs_min", "apache_gcs_total"),
    ]:
        if apache_col in df.columns:
            df[col] = df.get(col, pd.Series(np.nan, index=df.index)).combine_first(df[apache_col])
    if "apache_vent" in df.columns:
        df["invasive_ventilation"] = df.get("invasive_ventilation", pd.Series(False, index=df.index)).fillna(False) | df["apache_vent"].fillna(False)

    fio2 = df.get("fio2_resp_max", pd.Series(np.nan, index=df.index)).combine_first(df.get("fio2_lab_max", pd.Series(np.nan, index=df.index)))
    fio2_frac = fio2.where(fio2 <= 1, fio2 / 100.0)
    fio2_frac = fio2_frac.where(fio2_frac >= 0.21)
    df["fio2_fraction"] = fio2_frac
    df["pao2_fio2_ratio"] = df["pao2_min"] / df["fio2_fraction"]

    vent = df.get("invasive_ventilation", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["sofa_respiration"] = score_respiration(df["pao2_fio2_ratio"], vent)
    df["sofa_coagulation"] = score_platelet(df["platelet_min"])
    df["sofa_liver"] = score_liver(df["bilirubin_total_max"])
    df["sofa_cardiovascular"] = score_cardio(df.get("map_min", pd.Series(np.nan, index=df.index)), df.get("vasopressor_sofa_score", pd.Series(np.nan, index=df.index)))
    df["sofa_cns"] = score_cns(df.get("gcs_min", pd.Series(np.nan, index=df.index)))
    df["sofa_renal"] = score_renal(df.get("creatinine_max", pd.Series(np.nan, index=df.index)), df.get("urine_output_ml", pd.Series(np.nan, index=df.index)))

    sofa_cols = ["sofa_respiration", "sofa_coagulation", "sofa_liver", "sofa_cardiovascular", "sofa_cns", "sofa_renal"]
    df["sofa_components_available"] = df[sofa_cols].notna().sum(axis=1)
    df["sofa_total_assuming_missing_normal"] = df[sofa_cols].fillna(0).sum(axis=1)
    df["sofa_total_complete"] = df[sofa_cols].sum(axis=1, min_count=6)
    df["sofa_4systems_for_sic"] = df[["sofa_respiration", "sofa_liver", "sofa_cardiovascular", "sofa_renal"]].fillna(0).sum(axis=1)

    df["sic_platelet_component"] = sic_platelet_component(df["platelet_min"])
    df["sic_inr_component"] = sic_inr_component(df["inr_max"])
    df["sic_organ_sofa_component"] = sic_organ_component(df["sofa_4systems_for_sic"])
    df["sic_total_score"] = df[["sic_platelet_component", "sic_inr_component", "sic_organ_sofa_component"]].sum(axis=1, min_count=3)
    df["sic_flag"] = (df["sic_total_score"] >= 4) & ((df["sic_platelet_component"] + df["sic_inr_component"]) >= 3)

    return df.merge(cohort, on="patientunitstayid", how="left")


def build_sepsis3(cohort: pd.DataFrame, sofa: pd.DataFrame, suspected: pd.DataFrame) -> pd.DataFrame:
    w1 = sofa[sofa["window_name"].eq("w1_0_24h")].copy()
    max_sofa = sofa.groupby("patientunitstayid", as_index=False).agg(
        sofa_total_w1=("sofa_total_assuming_missing_normal", lambda s: np.nan),
    )
    max_sofa = sofa.groupby("patientunitstayid", as_index=False).agg(
        sofa_total_max_0_72h=("sofa_total_assuming_missing_normal", "max"),
        sofa_components_available_max_0_72h=("sofa_components_available", "max"),
    )
    w1_small = w1[["patientunitstayid", "sofa_total_assuming_missing_normal", "sofa_components_available"]].rename(columns={
        "sofa_total_assuming_missing_normal": "sofa_total_w1",
        "sofa_components_available": "sofa_components_available_w1",
    })
    out = cohort.merge(w1_small, on="patientunitstayid", how="left").merge(max_sofa, on="patientunitstayid", how="left")
    out = out.merge(suspected, on="patientunitstayid", how="left")
    out["sepsis3_icu_admit_flag"] = out["suspected_infection_near_icu_admit"].fillna(False) & out["sofa_total_w1"].fillna(0).ge(2)
    out["sepsis3_0_72h_flag"] = out["suspected_infection_offset_min"].notna() & out["sofa_total_max_0_72h"].fillna(0).ge(2)
    return out


def build_trajectory(scores: pd.DataFrame, sepsis3: pd.DataFrame) -> pd.DataFrame:
    log("build wide trajectory table")
    value_cols = [
        "platelet_min", "inr_max", "bilirubin_total_max", "creatinine_max", "lactate_max",
        "pao2_fio2_ratio", "sic_flag", "sic_platelet_component", "sic_inr_component",
        "sic_organ_sofa_component", "sic_total_score", "sofa_4systems_for_sic",
        "sofa_respiration", "sofa_liver", "sofa_cardiovascular", "sofa_renal", "sofa_cns",
        "sofa_total_assuming_missing_normal", "sofa_components_available",
    ]
    wide = scores.pivot_table(index="patientunitstayid", columns="window_name", values=value_cols, aggfunc="first")
    wide.columns = [f"{window.replace('_0_24h', '').replace('_24_48h', '').replace('_48_72h', '')}_{var}" for var, window in wide.columns]
    wide = wide.reset_index()
    rename = {}
    for c in wide.columns:
        rename[c] = c.replace("w1_", "w1_").replace("w2_", "w2_").replace("w3_", "w3_")
    wide = wide.rename(columns=rename)
    merged = sepsis3.merge(wide, on="patientunitstayid", how="left")
    merged["sic_w3_minus_w1"] = merged["w3_sic_total_score"] - merged["w1_sic_total_score"]
    merged["sofa_w3_minus_w1"] = merged["w3_sofa_total_assuming_missing_normal"] - merged["w1_sofa_total_assuming_missing_normal"]
    merged["sic_mean_3w"] = merged[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].mean(axis=1)
    merged["sic_max_3w"] = merged[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].max(axis=1)
    merged["sic_min_3w"] = merged[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].min(axis=1)
    merged["n_windows_available"] = merged[["w1_sic_total_score", "w2_sic_total_score", "w3_sic_total_score"]].notna().sum(axis=1)
    merged["sic_slope"] = (merged["w3_sic_total_score"] - merged["w1_sic_total_score"]) / 2.0
    return merged


def export_outputs(cohort: pd.DataFrame, scores: pd.DataFrame, suspected: pd.DataFrame, sepsis3: pd.DataFrame, traj: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT_ROOT / "eicu_adult_first_icu_cohort.csv", index=False)
    scores.to_csv(OUTPUT_ROOT / "eicu_full_sofa_sic_scores_0_72h.csv", index=False)
    suspected.to_csv(OUTPUT_ROOT / "eicu_suspected_infection.csv", index=False)
    sepsis3.to_csv(OUTPUT_ROOT / "eicu_sepsis3_cohort_all_adult_first_icu.csv", index=False)
    sepsis3[sepsis3["sepsis3_icu_admit_flag"]].to_csv(OUTPUT_ROOT / "eicu_sepsis3_icu_admit_cohort.csv", index=False)
    sepsis3[sepsis3["sepsis3_0_72h_flag"]].to_csv(OUTPUT_ROOT / "eicu_sepsis3_0_72h_cohort.csv", index=False)
    traj.to_csv(OUTPUT_ROOT / "eicu_sic_trajectory_features_full_sofa.csv", index=False)
    traj[traj["sepsis3_icu_admit_flag"]].to_csv(OUTPUT_ROOT / "eicu_sepsis3_icu_admit_sic_trajectory_features.csv", index=False)

    summary = {
        "analysis_date": datetime.now().isoformat(),
        "adult_first_icu": int(len(cohort)),
        "suspected_infection": int(len(suspected)),
        "sepsis3_icu_admit": int(sepsis3["sepsis3_icu_admit_flag"].sum()),
        "sepsis3_0_72h": int(sepsis3["sepsis3_0_72h_flag"].sum()),
        "trajectory_rows": int(len(traj)),
        "sepsis3_icu_admit_trajectory_rows": int(traj["sepsis3_icu_admit_flag"].sum()),
        "w1_full_sic_available": int(traj["w1_sic_total_score"].notna().sum()),
    }
    (METADATA_ROOT / "eicu_full_sofa_sepsis3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(METADATA_ROOT / "eicu_full_sofa_sepsis3_summary.csv", index=False)
    log(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive eICU full SOFA, Sepsis-3, and full SIC trajectories.")
    parser.add_argument("--skip-suspected-infection", action="store_true")
    args = parser.parse_args()

    root = raw_root()
    log(f"raw root: {root}")
    cohort = add_diagnosis_sepsis(root, load_cohort(root))
    ids = set(cohort["patientunitstayid"].astype(str))

    labs = extract_labs(root, ids)
    resp = extract_resp(root, ids)
    maps = extract_map(root, ids)
    gcs = extract_gcs(root, ids)
    urine = extract_urine(root, ids)
    vaso = extract_vasopressors(root, cohort)
    apache = extract_apache(root, ids)

    scores = build_scores(cohort, [labs, resp, maps, gcs, urine, vaso, apache])
    suspected = pd.DataFrame(columns=["patientunitstayid"]) if args.skip_suspected_infection else extract_suspected_infection(root, ids)
    sepsis3 = build_sepsis3(cohort, scores, suspected)
    traj = build_trajectory(scores, sepsis3)
    export_outputs(cohort, scores, suspected, sepsis3, traj)


if __name__ == "__main__":
    main()
