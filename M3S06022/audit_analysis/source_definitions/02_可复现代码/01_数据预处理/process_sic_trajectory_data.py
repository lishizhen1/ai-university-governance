from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import psycopg

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MIMIC_DIR, EICU_DIR, META_DIR, OUTPUT_DIR, PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB_MIMIC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = OUTPUT_DIR / "data_processed"    # Derived data output


def connect(database: str):
    """Connect to PostgreSQL using environment variables or defaults."""
    return psycopg.connect(
        host=PG_HOST,
        port=credentials["port"],
        dbname=database,
        user=credentials["admin user"],
        password=PG_PASSWORD,
        autocommit=True,
    )


def execute(conn, label: str, sql: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {label}", flush=True)
    with conn.cursor() as cur:
        cur.execute(sql)


def query_scalar(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def copy_query(conn, label: str, sql: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%H:%M:%S}] export {label} -> {output_path}", flush=True)
    copy_sql = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"
    with conn.cursor() as cur, cur.copy(copy_sql) as copy, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        while True:
            chunk = copy.read()
            if not chunk:
                break
            if isinstance(chunk, memoryview):
                chunk = chunk.tobytes()
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            fh.write(chunk)


def ensure_directories() -> None:
    folders = [
        "00_metadata",
        "01_cohort",
        "02_window_features",
        "03_sic_scores",
        "04_trajectory_features",
        "05_outcomes",
        "06_external_validation",
        "99_logs",
    ]
    for folder in folders:
        (OUTPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for folder in ["sql", "results/tables", "results/figures", "results/models", "docs"]:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)


def setup_connection(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SET idle_in_transaction_session_timeout = 0")
        cur.execute("SET work_mem = '256MB'")


MIMIC_DROP_SQL = """
CREATE SCHEMA IF NOT EXISTS aris_sic;
DROP TABLE IF EXISTS
    aris_sic.mimiciv_partial_trajectory_features,
    aris_sic.mimiciv_partial_sic_scores,
    aris_sic.mimiciv_lab_window_features_wide,
    aris_sic.mimiciv_lab_window_features_long,
    aris_sic.mimiciv_core_labs_0_72h,
    aris_sic.mimiciv_lab_item_mapping,
    aris_sic.mimiciv_outcomes,
    aris_sic.mimiciv_adult_first_icu_cohort
CASCADE;
"""

MIMIC_PREFLIGHT_SQL = """
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_icustays_subject_hadm
    ON icu.icustays(subject_id, hadm_id);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_icustays_subject_intime
    ON icu.icustays(subject_id, intime);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_admissions_subject_hadm
    ON hosp.admissions(subject_id, hadm_id);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_patients_subject
    ON hosp.patients(subject_id);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_diagnoses_subject_hadm_code
    ON hosp.diagnoses_icd(subject_id, hadm_id, icd_version, icd_code);
ANALYZE icu.icustays;
ANALYZE hosp.admissions;
ANALYZE hosp.patients;
ANALYZE hosp.diagnoses_icd;
"""

MIMIC_COHORT_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_adult_first_icu_cohort AS
WITH icu AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.first_careunit,
        i.last_careunit,
        NULLIF(i.intime, '')::timestamp AS icu_intime,
        NULLIF(i.outtime, '')::timestamp AS icu_outtime,
        CASE WHEN i.los ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN i.los::numeric END AS icu_los_days,
        ROW_NUMBER() OVER (
            PARTITION BY i.subject_id
            ORDER BY NULLIF(i.intime, '')::timestamp, i.stay_id
        ) AS icu_sequence
    FROM icu.icustays i
    WHERE i.intime ~ '^\\d{4}-\\d{2}-\\d{2}'
),
sepsis_dx AS (
    SELECT DISTINCT d.subject_id, d.hadm_id, TRUE AS sepsis_icd_flag
    FROM hosp.diagnoses_icd d
    WHERE
        (
            d.icd_version = '10'
            AND (
                d.icd_code LIKE 'A40%'
                OR d.icd_code LIKE 'A41%'
                OR d.icd_code LIKE 'R652%'
            )
        )
        OR (
            d.icd_version = '9'
            AND d.icd_code IN ('99591', '99592', '78552')
        )
),
base AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.first_careunit,
        i.last_careunit,
        i.icu_intime,
        i.icu_outtime,
        i.icu_los_days,
        i.icu_sequence,
        p.gender,
        CASE WHEN p.anchor_age ~ '^\\d+$' THEN p.anchor_age::int END AS anchor_age,
        CASE WHEN p.dod ~ '^\\d{4}-\\d{2}-\\d{2}' THEN p.dod::timestamp END AS date_of_death,
        NULLIF(a.admittime, '')::timestamp AS hospital_admittime,
        NULLIF(a.dischtime, '')::timestamp AS hospital_dischtime,
        CASE WHEN a.deathtime ~ '^\\d{4}-\\d{2}-\\d{2}' THEN a.deathtime::timestamp END AS hospital_deathtime,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.insurance,
        a.language,
        a.marital_status,
        a.race,
        CASE WHEN a.hospital_expire_flag ~ '^\\d+$' THEN a.hospital_expire_flag::int END AS hospital_expire_flag
    FROM icu i
    INNER JOIN hosp.patients p ON i.subject_id = p.subject_id
    INNER JOIN hosp.admissions a ON i.subject_id = a.subject_id AND i.hadm_id = a.hadm_id
    WHERE i.icu_sequence = 1
)
SELECT
    b.*,
    COALESCE(s.sepsis_icd_flag, FALSE) AS sepsis_icd_flag
FROM base b
LEFT JOIN sepsis_dx s
    ON b.subject_id = s.subject_id AND b.hadm_id = s.hadm_id
WHERE b.anchor_age >= 18;
"""

MIMIC_OUTCOMES_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_outcomes AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    icu_intime,
    icu_outtime,
    hospital_admittime,
    hospital_dischtime,
    icu_los_days,
    EXTRACT(EPOCH FROM (hospital_dischtime - hospital_admittime)) / 86400.0 AS hospital_los_days,
    sepsis_icd_flag,
    hospital_expire_flag,
    CASE
        WHEN COALESCE(hospital_deathtime, date_of_death) IS NULL THEN FALSE
        WHEN COALESCE(hospital_deathtime, date_of_death) >= icu_intime
         AND COALESCE(hospital_deathtime, date_of_death) <= icu_outtime THEN TRUE
        ELSE FALSE
    END AS icu_mortality,
    CASE WHEN hospital_expire_flag = 1 THEN TRUE ELSE FALSE END AS hospital_mortality,
    CASE
        WHEN COALESCE(hospital_deathtime, date_of_death) IS NULL THEN FALSE
        WHEN COALESCE(hospital_deathtime, date_of_death) <= icu_intime + INTERVAL '28 days' THEN TRUE
        ELSE FALSE
    END AS mortality_28d,
    CASE
        WHEN COALESCE(hospital_deathtime, date_of_death) IS NULL THEN FALSE
        WHEN COALESCE(hospital_deathtime, date_of_death) <= icu_intime + INTERVAL '90 days' THEN TRUE
        ELSE FALSE
    END AS mortality_90d
FROM aris_sic.mimiciv_adult_first_icu_cohort;
"""

MIMIC_LAB_MAPPING_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_lab_item_mapping AS
SELECT *
FROM (
    SELECT
        itemid,
        label,
        fluid,
        category,
        CASE
            WHEN label = 'Platelet Count' AND fluid = 'Blood' THEN 'platelet'
            WHEN label = 'INR(PT)' AND fluid = 'Blood' THEN 'inr'
            WHEN label = 'PT' AND fluid = 'Blood' THEN 'pt'
            WHEN label = 'PTT' AND fluid = 'Blood' THEN 'aptt'
            WHEN label IN ('Fibrinogen', 'Fibrinogen, Functional') AND fluid = 'Blood' THEN 'fibrinogen'
            WHEN label = 'D-Dimer' AND fluid = 'Blood' THEN 'd_dimer'
            WHEN label = 'Bilirubin, Total' AND fluid = 'Blood' THEN 'bilirubin_total'
            WHEN label IN ('Creatinine', 'Creatinine, Whole Blood') AND fluid = 'Blood' THEN 'creatinine'
            WHEN label = 'Lactate' AND fluid = 'Blood' THEN 'lactate'
        END AS variable,
        CASE
            WHEN label IN ('Platelet Count', 'Fibrinogen', 'Fibrinogen, Functional') THEN 'min'
            ELSE 'max'
        END AS preferred_window_aggregation
    FROM hosp.d_labitems
) x
WHERE variable IS NOT NULL;
"""

MIMIC_CORE_LABS_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_core_labs_0_72h AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.icu_intime,
    c.sepsis_icd_flag,
    m.variable,
    m.itemid,
    m.label AS source_label,
    m.fluid,
    m.category,
    NULLIF(le.charttime, '')::timestamp AS charttime,
    EXTRACT(EPOCH FROM (NULLIF(le.charttime, '')::timestamp - c.icu_intime)) / 3600.0 AS hours_from_icu,
    CASE
        WHEN EXTRACT(EPOCH FROM (NULLIF(le.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 24 THEN 'w1_0_24h'
        WHEN EXTRACT(EPOCH FROM (NULLIF(le.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 48 THEN 'w2_24_48h'
        ELSE 'w3_48_72h'
    END AS window_name,
    le.valuenum::numeric AS value_num,
    le.valueuom
FROM hosp.labevents le
INNER JOIN aris_sic.mimiciv_lab_item_mapping m ON le.itemid = m.itemid
INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
    ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
WHERE
    le.valuenum ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND le.charttime ~ '^\\d{4}-\\d{2}-\\d{2}'
    AND NULLIF(le.charttime, '')::timestamp >= c.icu_intime
    AND NULLIF(le.charttime, '')::timestamp < c.icu_intime + INTERVAL '72 hours';
"""

WINDOW_LONG_SQL_TEMPLATE = """
CREATE UNLOGGED TABLE aris_sic.{table_prefix}_lab_window_features_long AS
SELECT
    {id_cols},
    window_name,
    variable,
    COUNT(*) AS n_measurements,
    MIN(hours_from_icu) AS first_hour_from_icu,
    MAX(hours_from_icu) AS last_hour_from_icu,
    MIN(value_num) AS min_value,
    MAX(value_num) AS max_value,
    AVG(value_num) AS mean_value,
    (ARRAY_AGG(value_num ORDER BY hours_from_icu ASC))[1] AS first_value,
    (ARRAY_AGG(value_num ORDER BY hours_from_icu DESC))[1] AS last_value
FROM aris_sic.{table_prefix}_core_labs_0_72h
GROUP BY {id_cols}, window_name, variable;
"""

WINDOW_WIDE_SQL_TEMPLATE = """
CREATE UNLOGGED TABLE aris_sic.{table_prefix}_lab_window_features_wide AS
SELECT
    {base_id_cols},
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'platelet' THEN f.min_value END) AS w1_platelet_min,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'platelet' THEN f.min_value END) AS w2_platelet_min,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'platelet' THEN f.min_value END) AS w3_platelet_min,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'inr' THEN f.max_value END) AS w1_inr_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'inr' THEN f.max_value END) AS w2_inr_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'inr' THEN f.max_value END) AS w3_inr_max,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'pt' THEN f.max_value END) AS w1_pt_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'pt' THEN f.max_value END) AS w2_pt_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'pt' THEN f.max_value END) AS w3_pt_max,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'aptt' THEN f.max_value END) AS w1_aptt_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'aptt' THEN f.max_value END) AS w2_aptt_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'aptt' THEN f.max_value END) AS w3_aptt_max,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'fibrinogen' THEN f.min_value END) AS w1_fibrinogen_min,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'fibrinogen' THEN f.min_value END) AS w2_fibrinogen_min,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'fibrinogen' THEN f.min_value END) AS w3_fibrinogen_min,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'd_dimer' THEN f.max_value END) AS w1_d_dimer_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'd_dimer' THEN f.max_value END) AS w2_d_dimer_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'd_dimer' THEN f.max_value END) AS w3_d_dimer_max,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'bilirubin_total' THEN f.max_value END) AS w1_bilirubin_total_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'bilirubin_total' THEN f.max_value END) AS w2_bilirubin_total_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'bilirubin_total' THEN f.max_value END) AS w3_bilirubin_total_max,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'creatinine' THEN f.max_value END) AS w1_creatinine_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'creatinine' THEN f.max_value END) AS w2_creatinine_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'creatinine' THEN f.max_value END) AS w3_creatinine_max,
    MAX(CASE WHEN f.window_name = 'w1_0_24h' AND f.variable = 'lactate' THEN f.max_value END) AS w1_lactate_max,
    MAX(CASE WHEN f.window_name = 'w2_24_48h' AND f.variable = 'lactate' THEN f.max_value END) AS w2_lactate_max,
    MAX(CASE WHEN f.window_name = 'w3_48_72h' AND f.variable = 'lactate' THEN f.max_value END) AS w3_lactate_max
FROM aris_sic.{table_prefix}_{cohort_table} c
LEFT JOIN aris_sic.{table_prefix}_lab_window_features_long f
    ON {join_condition}
GROUP BY {base_id_cols};
"""

PARTIAL_SIC_SQL_TEMPLATE = """
CREATE UNLOGGED TABLE aris_sic.{table_prefix}_partial_sic_scores AS
WITH windowed AS (
    SELECT {id_select}, 'w1_0_24h' AS window_name,
        w1_platelet_min AS platelet_min,
        w1_inr_max AS inr_max,
        w1_bilirubin_total_max AS bilirubin_total_max,
        w1_creatinine_max AS creatinine_max
    FROM aris_sic.{table_prefix}_lab_window_features_wide
    UNION ALL
    SELECT {id_select}, 'w2_24_48h',
        w2_platelet_min,
        w2_inr_max,
        w2_bilirubin_total_max,
        w2_creatinine_max
    FROM aris_sic.{table_prefix}_lab_window_features_wide
    UNION ALL
    SELECT {id_select}, 'w3_48_72h',
        w3_platelet_min,
        w3_inr_max,
        w3_bilirubin_total_max,
        w3_creatinine_max
    FROM aris_sic.{table_prefix}_lab_window_features_wide
),
components AS (
    SELECT
        *,
        CASE
            WHEN platelet_min IS NULL THEN NULL
            WHEN platelet_min < 100 THEN 2
            WHEN platelet_min < 150 THEN 1
            ELSE 0
        END AS sic_platelet_component,
        CASE
            WHEN inr_max IS NULL THEN NULL
            WHEN inr_max > 1.4 THEN 2
            WHEN inr_max > 1.2 THEN 1
            ELSE 0
        END AS sic_inr_component,
        CASE
            WHEN bilirubin_total_max IS NULL THEN NULL
            WHEN bilirubin_total_max >= 12.0 THEN 4
            WHEN bilirubin_total_max >= 6.0 THEN 3
            WHEN bilirubin_total_max >= 2.0 THEN 2
            WHEN bilirubin_total_max >= 1.2 THEN 1
            ELSE 0
        END AS sofa_liver_bilirubin_component,
        CASE
            WHEN creatinine_max IS NULL THEN NULL
            WHEN creatinine_max >= 5.0 THEN 4
            WHEN creatinine_max >= 3.5 THEN 3
            WHEN creatinine_max >= 2.0 THEN 2
            WHEN creatinine_max >= 1.2 THEN 1
            ELSE 0
        END AS sofa_renal_creatinine_component
    FROM windowed
)
SELECT
    *,
    CASE
        WHEN sic_platelet_component IS NULL OR sic_inr_component IS NULL THEN NULL
        ELSE sic_platelet_component + sic_inr_component
    END AS sic_coagulation_partial_score,
    CASE
        WHEN sic_platelet_component IS NULL OR sic_inr_component IS NULL THEN NULL
        ELSE sic_platelet_component + sic_inr_component >= 3
    END AS coagulation_high_risk_flag,
    CASE
        WHEN sic_platelet_component IS NOT NULL AND sic_inr_component IS NOT NULL THEN TRUE
        ELSE FALSE
    END AS complete_for_platelet_inr
FROM components;
"""

TRAJECTORY_SQL_TEMPLATE = """
CREATE UNLOGGED TABLE aris_sic.{table_prefix}_partial_trajectory_features AS
WITH score_wide AS (
    SELECT
        {id_group_cols},
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN sic_coagulation_partial_score END) AS w1_sic_coagulation_partial_score,
        MAX(CASE WHEN window_name = 'w2_24_48h' THEN sic_coagulation_partial_score END) AS w2_sic_coagulation_partial_score,
        MAX(CASE WHEN window_name = 'w3_48_72h' THEN sic_coagulation_partial_score END) AS w3_sic_coagulation_partial_score,
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN coagulation_high_risk_flag::int END)::boolean AS w1_coagulation_high_risk_flag,
        MAX(CASE WHEN window_name = 'w2_24_48h' THEN coagulation_high_risk_flag::int END)::boolean AS w2_coagulation_high_risk_flag,
        MAX(CASE WHEN window_name = 'w3_48_72h' THEN coagulation_high_risk_flag::int END)::boolean AS w3_coagulation_high_risk_flag
    FROM aris_sic.{table_prefix}_partial_sic_scores
    GROUP BY {id_group_cols}
)
SELECT
    w.*,
    f.w1_platelet_min,
    f.w2_platelet_min,
    f.w3_platelet_min,
    f.w1_inr_max,
    f.w2_inr_max,
    f.w3_inr_max,
    f.w3_platelet_min - f.w1_platelet_min AS platelet_min_w3_minus_w1,
    f.w3_inr_max - f.w1_inr_max AS inr_max_w3_minus_w1,
    w.w2_sic_coagulation_partial_score - w.w1_sic_coagulation_partial_score AS partial_sic_w2_minus_w1,
    w.w3_sic_coagulation_partial_score - w.w1_sic_coagulation_partial_score AS partial_sic_w3_minus_w1,
    CASE
        WHEN GREATEST(
            w.w1_sic_coagulation_partial_score,
            w.w2_sic_coagulation_partial_score,
            w.w3_sic_coagulation_partial_score
        ) IS NULL THEN 'no_score_available'
        WHEN LEAST(
            w.w1_sic_coagulation_partial_score,
            w.w2_sic_coagulation_partial_score,
            w.w3_sic_coagulation_partial_score
        ) >= 3 THEN 'persistent_high'
        WHEN w.w1_sic_coagulation_partial_score >= 3
         AND w.w3_sic_coagulation_partial_score < 3 THEN 'resolving'
        WHEN w.w1_sic_coagulation_partial_score IS NOT NULL
         AND w.w3_sic_coagulation_partial_score IS NOT NULL
         AND w.w3_sic_coagulation_partial_score > w.w1_sic_coagulation_partial_score THEN 'worsening'
        WHEN GREATEST(
            w.w1_sic_coagulation_partial_score,
            w.w2_sic_coagulation_partial_score,
            w.w3_sic_coagulation_partial_score
        ) < 3 THEN 'persistently_low'
        ELSE 'mixed_or_incomplete'
    END AS partial_sic_trajectory_group
FROM score_wide w
LEFT JOIN aris_sic.{table_prefix}_lab_window_features_wide f
    ON {wide_join_condition};
"""

EICU_DROP_SQL = """
CREATE SCHEMA IF NOT EXISTS aris_sic;
DROP TABLE IF EXISTS
    aris_sic.eicu_partial_trajectory_features,
    aris_sic.eicu_partial_sic_scores,
    aris_sic.eicu_lab_window_features_wide,
    aris_sic.eicu_lab_window_features_long,
    aris_sic.eicu_core_labs_0_72h,
    aris_sic.eicu_lab_mapping,
    aris_sic.eicu_outcomes,
    aris_sic.eicu_adult_icu_cohort
CASCADE;
"""

EICU_COHORT_SQL = """
CREATE UNLOGGED TABLE aris_sic.eicu_adult_icu_cohort AS
WITH sepsis_dx AS (
    SELECT DISTINCT patientunitstayid, TRUE AS sepsis_text_flag
    FROM eicu.diagnosis
    WHERE LOWER(COALESCE(diagnosisstring, '') || ' ' || COALESCE(icd9code, '')) LIKE '%sepsis%'
       OR LOWER(COALESCE(diagnosisstring, '') || ' ' || COALESCE(icd9code, '')) LIKE '%septic%'
),
base AS (
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.uniquepid,
        p.gender,
        CASE
            WHEN p.age = '> 89' THEN 90
            WHEN p.age ~ '^\\d+$' THEN p.age::int
        END AS age_years,
        p.ethnicity,
        p.hospitalid,
        p.wardid,
        p.apacheadmissiondx,
        CASE WHEN p.admissionheight ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN p.admissionheight::numeric END AS admissionheight,
        CASE WHEN p.admissionweight ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN p.admissionweight::numeric END AS admissionweight,
        p.hospitaladmitoffset,
        p.hospitaldischargeoffset,
        p.hospitaldischargestatus,
        p.unitdischargeoffset,
        p.unitdischargestatus,
        p.unittype,
        p.unitvisitnumber,
        p.unitstaytype
    FROM eicu.patient p
)
SELECT
    b.*,
    CASE
        WHEN COALESCE(s.sepsis_text_flag, FALSE) THEN TRUE
        WHEN LOWER(COALESCE(b.apacheadmissiondx, '')) LIKE '%sepsis%' THEN TRUE
        WHEN LOWER(COALESCE(b.apacheadmissiondx, '')) LIKE '%septic%' THEN TRUE
        ELSE FALSE
    END AS sepsis_text_flag
FROM base b
LEFT JOIN sepsis_dx s ON b.patientunitstayid = s.patientunitstayid
WHERE b.age_years >= 18;
"""

EICU_OUTCOMES_SQL = """
CREATE UNLOGGED TABLE aris_sic.eicu_outcomes AS
SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    uniquepid,
    sepsis_text_flag,
    CASE WHEN unitdischargestatus = 'Expired' THEN TRUE ELSE FALSE END AS unit_mortality,
    CASE WHEN hospitaldischargestatus = 'Expired' THEN TRUE ELSE FALSE END AS hospital_mortality,
    CASE WHEN unitdischargeoffset ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN unitdischargeoffset::numeric / 1440.0 END AS unit_los_days,
    CASE WHEN hospitaldischargeoffset ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN hospitaldischargeoffset::numeric / 1440.0 END AS hospital_los_days
FROM aris_sic.eicu_adult_icu_cohort;
"""

EICU_LAB_MAPPING_SQL = """
CREATE UNLOGGED TABLE aris_sic.eicu_lab_mapping AS
SELECT * FROM (VALUES
    ('platelets x 1000', 'platelet', 'min'),
    ('pt - inr', 'inr', 'max'),
    ('ptt', 'aptt', 'max'),
    ('fibrinogen', 'fibrinogen', 'min'),
    ('total bilirubin', 'bilirubin_total', 'max'),
    ('creatinine', 'creatinine', 'max'),
    ('lactate', 'lactate', 'max')
) AS m(labname_lower, variable, preferred_window_aggregation);
"""

EICU_CORE_LABS_SQL = """
CREATE UNLOGGED TABLE aris_sic.eicu_core_labs_0_72h AS
SELECT
    c.patientunitstayid,
    c.patienthealthsystemstayid,
    c.uniquepid,
    c.sepsis_text_flag,
    m.variable,
    l.labname AS source_label,
    l.labresultoffset::numeric / 60.0 AS hours_from_icu,
    CASE
        WHEN l.labresultoffset::numeric < 1440 THEN 'w1_0_24h'
        WHEN l.labresultoffset::numeric < 2880 THEN 'w2_24_48h'
        ELSE 'w3_48_72h'
    END AS window_name,
    l.labresult::numeric AS value_num,
    l.labmeasurenamesystem,
    l.labmeasurenameinterface
FROM eicu.lab l
INNER JOIN aris_sic.eicu_lab_mapping m ON LOWER(l.labname) = m.labname_lower
INNER JOIN aris_sic.eicu_adult_icu_cohort c ON l.patientunitstayid = c.patientunitstayid
WHERE
    l.labresult ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND l.labresultoffset ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND l.labresultoffset::numeric >= 0
    AND l.labresultoffset::numeric < 4320;
"""

SICDB_DROP_SQL = """
CREATE SCHEMA IF NOT EXISTS aris_sic;
DROP TABLE IF EXISTS
    aris_sic.sicdb_partial_trajectory_features,
    aris_sic.sicdb_partial_sic_scores,
    aris_sic.sicdb_lab_window_features_wide,
    aris_sic.sicdb_lab_window_features_long,
    aris_sic.sicdb_core_labs_0_72h,
    aris_sic.sicdb_lab_mapping,
    aris_sic.sicdb_outcomes,
    aris_sic.sicdb_cases_cohort
CASCADE;
"""

SICDB_COHORT_SQL = """
CREATE UNLOGGED TABLE aris_sic.sicdb_cases_cohort AS
SELECT
    c.caseid,
    c.patientid,
    CASE WHEN c.ageonadmission ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.ageonadmission::numeric END AS age_years,
    c.sex,
    CASE WHEN c.weightonadmission ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.weightonadmission::numeric END AS weightonadmission,
    CASE WHEN c.heightonadmission ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.heightonadmission::numeric END AS heightonadmission,
    c.hospitalunit,
    c.referringunit,
    c.icd10main,
    c.icd10maintext,
    c.diagnosist2,
    c.surgicalsite,
    c.admissionurgency,
    c.admissionformhassepsis,
    CASE WHEN c.admissionformhassepsis = '740' THEN TRUE ELSE FALSE END AS sepsis_admission_form_flag,
    c.surgicaladmissiontype,
    CASE WHEN c.saps3 ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.saps3::numeric END AS saps3,
    CASE WHEN c.timeofstay ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.timeofstay::numeric END AS timeofstay,
    CASE WHEN c.icuoffset ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.icuoffset::numeric END AS icuoffset,
    c.hospitaldischargetype,
    c.dischargestate,
    CASE WHEN c.hospitalstaydays ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.hospitalstaydays::numeric END AS hospitalstaydays,
    CASE WHEN c.offsetofdeath ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.offsetofdeath::numeric END AS offsetofdeath,
    CASE WHEN c.estimatedsurvivalobservationtime ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN c.estimatedsurvivalobservationtime::numeric END AS estimatedsurvivalobservationtime
FROM sicdb.cases c
WHERE c.ageonadmission ~ '^-?[0-9]+(\\.[0-9]+)?$'
  AND c.ageonadmission::numeric >= 18;
"""

SICDB_OUTCOMES_SQL = """
CREATE UNLOGGED TABLE aris_sic.sicdb_outcomes AS
SELECT
    caseid,
    patientid,
    sepsis_admission_form_flag,
    hospitaldischargetype,
    dischargestate,
    hospitalstaydays,
    offsetofdeath,
    estimatedsurvivalobservationtime,
    CASE WHEN offsetofdeath IS NOT NULL THEN TRUE ELSE FALSE END AS observed_mortality,
    CASE WHEN offsetofdeath IS NOT NULL AND offsetofdeath <= 28 * 24 * 60 THEN TRUE ELSE FALSE END AS mortality_28d,
    CASE WHEN offsetofdeath IS NOT NULL AND offsetofdeath <= 90 * 24 * 60 THEN TRUE ELSE FALSE END AS mortality_90d
FROM aris_sic.sicdb_cases_cohort;
"""

SICDB_LAB_MAPPING_SQL = """
CREATE UNLOGGED TABLE aris_sic.sicdb_lab_mapping AS
SELECT * FROM (VALUES
    ('314', 'platelet', 'Thrombocyten (ZL)', 'min'),
    ('315', 'platelet', 'Thrombocyten/Citratblut (ZL)', 'min'),
    ('3128', 'inr', 'PT / INR', 'max'),
    ('237', 'pt_activity', 'PTZ prothrombin activity actual/normal', 'min'),
    ('598', 'pt', 'PT (ZL)', 'max'),
    ('597', 'aptt', 'APTT (ZL)', 'max'),
    ('344', 'fibrinogen', 'Fibrinogen (ZL)', 'min'),
    ('402', 'd_dimer', 'D-Dimer (ZL)', 'max'),
    ('333', 'bilirubin_total', 'Bilirubin total (ZL)', 'max'),
    ('367', 'creatinine', 'Kreatinin (ZL)', 'max'),
    ('368', 'creatinine', 'Kreatinin enzymatisch (ZL)', 'max'),
    ('454', 'lactate', 'Lactat BG (ZL)', 'max'),
    ('465', 'lactate', 'Lactat (ZL)', 'max'),
    ('657', 'lactate', 'Lactat (BGA)', 'max')
) AS m(laboratoryid, variable, source_label, preferred_window_aggregation);
"""

SICDB_CORE_LABS_SQL = """
CREATE UNLOGGED TABLE aris_sic.sicdb_core_labs_0_72h AS
SELECT
    c.caseid,
    c.patientid,
    c.sepsis_admission_form_flag,
    m.variable,
    m.laboratoryid,
    COALESCE(r.referencevalue, m.source_label) AS source_label,
    l."offset"::numeric / 60.0 AS hours_from_icu,
    CASE
        WHEN l."offset"::numeric < 1440 THEN 'w1_0_24h'
        WHEN l."offset"::numeric < 2880 THEN 'w2_24_48h'
        ELSE 'w3_48_72h'
    END AS window_name,
    l.laboratoryvalue::numeric AS value_num,
    r.referenceunit
FROM sicdb.laboratory l
INNER JOIN aris_sic.sicdb_lab_mapping m ON l.laboratoryid = m.laboratoryid
INNER JOIN aris_sic.sicdb_cases_cohort c ON l.caseid = c.caseid
LEFT JOIN sicdb.d_references r ON l.laboratoryid = r.referenceglobalid
WHERE
    l.laboratoryvalue ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND l."offset" ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND l."offset"::numeric >= 0
    AND l."offset"::numeric < 4320;
"""


def process_mimiciv(summary: list[dict[str, str]], build: bool = True) -> None:
    with connect("mimiciv31") as conn:
        setup_connection(conn)
        if build:
            for label, sql in [
                ("mimiciv reset aris_sic tables", MIMIC_DROP_SQL),
                ("mimiciv preflight indexes/statistics", MIMIC_PREFLIGHT_SQL),
                ("mimiciv adult first ICU cohort", MIMIC_COHORT_SQL),
                ("mimiciv outcomes", MIMIC_OUTCOMES_SQL),
                ("mimiciv lab item mapping", MIMIC_LAB_MAPPING_SQL),
                ("mimiciv core labs 0-72h", MIMIC_CORE_LABS_SQL),
                (
                    "mimiciv lab window features long",
                    WINDOW_LONG_SQL_TEMPLATE.format(
                        table_prefix="mimiciv",
                        id_cols="subject_id, hadm_id, stay_id",
                    ),
                ),
                (
                    "mimiciv lab window features wide",
                    WINDOW_WIDE_SQL_TEMPLATE.format(
                        table_prefix="mimiciv",
                        cohort_table="adult_first_icu_cohort",
                        base_id_cols="c.subject_id, c.hadm_id, c.stay_id",
                        join_condition="c.subject_id = f.subject_id AND c.hadm_id = f.hadm_id AND c.stay_id = f.stay_id",
                    ),
                ),
                (
                    "mimiciv partial SIC scores",
                    PARTIAL_SIC_SQL_TEMPLATE.format(
                        table_prefix="mimiciv",
                        id_select="subject_id, hadm_id, stay_id",
                    ),
                ),
                (
                    "mimiciv partial trajectory features",
                    TRAJECTORY_SQL_TEMPLATE.format(
                        table_prefix="mimiciv",
                        id_group_cols="subject_id, hadm_id, stay_id",
                        wide_join_condition="w.subject_id = f.subject_id AND w.hadm_id = f.hadm_id AND w.stay_id = f.stay_id",
                    ),
                ),
            ]:
                execute(conn, label, sql)
            for label, index_sql in [
                ("mimiciv analyze cohort", "ANALYZE aris_sic.mimiciv_adult_first_icu_cohort"),
                ("mimiciv analyze labs", "ANALYZE aris_sic.mimiciv_core_labs_0_72h"),
            ]:
                execute(conn, label, index_sql)
        else:
            print(f"[{datetime.now():%H:%M:%S}] mimiciv build skipped; exporting existing aris_sic tables", flush=True)
        exports = [
            (
                "mimiciv cohort",
                "SELECT * FROM aris_sic.mimiciv_adult_first_icu_cohort ORDER BY subject_id, stay_id",
                OUTPUT_ROOT / "01_cohort" / "mimiciv_adult_first_icu_cohort.csv",
            ),
            (
                "mimiciv sepsis candidate cohort",
                "SELECT * FROM aris_sic.mimiciv_adult_first_icu_cohort WHERE sepsis_icd_flag ORDER BY subject_id, stay_id",
                OUTPUT_ROOT / "01_cohort" / "mimiciv_icd_sepsis_candidate_cohort.csv",
            ),
            (
                "mimiciv outcomes",
                "SELECT * FROM aris_sic.mimiciv_outcomes ORDER BY subject_id, stay_id",
                OUTPUT_ROOT / "05_outcomes" / "mimiciv_outcomes.csv",
            ),
            (
                "mimiciv sepsis candidate outcomes",
                """
                SELECT o.*
                FROM aris_sic.mimiciv_outcomes o
                INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
                    ON o.subject_id = c.subject_id AND o.hadm_id = c.hadm_id AND o.stay_id = c.stay_id
                WHERE c.sepsis_icd_flag
                ORDER BY o.subject_id, o.stay_id
                """,
                OUTPUT_ROOT / "05_outcomes" / "mimiciv_icd_sepsis_candidate_outcomes.csv",
            ),
            (
                "mimiciv lab mapping",
                "SELECT * FROM aris_sic.mimiciv_lab_item_mapping ORDER BY variable, itemid",
                OUTPUT_ROOT / "00_metadata" / "mimiciv_lab_item_mapping.csv",
            ),
            (
                "mimiciv window features long",
                "SELECT * FROM aris_sic.mimiciv_lab_window_features_long ORDER BY subject_id, stay_id, window_name, variable",
                OUTPUT_ROOT / "02_window_features" / "mimiciv_lab_window_features_0_72h_long.csv",
            ),
            (
                "mimiciv sepsis candidate window features long",
                """
                SELECT f.*
                FROM aris_sic.mimiciv_lab_window_features_long f
                INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
                    ON f.subject_id = c.subject_id AND f.hadm_id = c.hadm_id AND f.stay_id = c.stay_id
                WHERE c.sepsis_icd_flag
                ORDER BY f.subject_id, f.stay_id, f.window_name, f.variable
                """,
                OUTPUT_ROOT / "02_window_features" / "mimiciv_icd_sepsis_candidate_lab_window_features_0_72h_long.csv",
            ),
            (
                "mimiciv window features wide",
                "SELECT * FROM aris_sic.mimiciv_lab_window_features_wide ORDER BY subject_id, stay_id",
                OUTPUT_ROOT / "02_window_features" / "mimiciv_lab_window_features_0_72h_wide.csv",
            ),
            (
                "mimiciv sepsis candidate window features wide",
                """
                SELECT f.*
                FROM aris_sic.mimiciv_lab_window_features_wide f
                INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
                    ON f.subject_id = c.subject_id AND f.hadm_id = c.hadm_id AND f.stay_id = c.stay_id
                WHERE c.sepsis_icd_flag
                ORDER BY f.subject_id, f.stay_id
                """,
                OUTPUT_ROOT / "02_window_features" / "mimiciv_icd_sepsis_candidate_lab_window_features_0_72h_wide.csv",
            ),
            (
                "mimiciv partial SIC scores",
                "SELECT * FROM aris_sic.mimiciv_partial_sic_scores ORDER BY subject_id, stay_id, window_name",
                OUTPUT_ROOT / "03_sic_scores" / "mimiciv_partial_sic_scores.csv",
            ),
            (
                "mimiciv sepsis candidate partial SIC scores",
                """
                SELECT s.*
                FROM aris_sic.mimiciv_partial_sic_scores s
                INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
                    ON s.subject_id = c.subject_id AND s.hadm_id = c.hadm_id AND s.stay_id = c.stay_id
                WHERE c.sepsis_icd_flag
                ORDER BY s.subject_id, s.stay_id, s.window_name
                """,
                OUTPUT_ROOT / "03_sic_scores" / "mimiciv_icd_sepsis_candidate_partial_sic_scores.csv",
            ),
            (
                "mimiciv partial trajectory features",
                "SELECT * FROM aris_sic.mimiciv_partial_trajectory_features ORDER BY subject_id, stay_id",
                OUTPUT_ROOT / "04_trajectory_features" / "mimiciv_partial_trajectory_features.csv",
            ),
            (
                "mimiciv sepsis candidate partial trajectory features",
                """
                SELECT t.*
                FROM aris_sic.mimiciv_partial_trajectory_features t
                INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
                    ON t.subject_id = c.subject_id AND t.hadm_id = c.hadm_id AND t.stay_id = c.stay_id
                WHERE c.sepsis_icd_flag
                ORDER BY t.subject_id, t.stay_id
                """,
                OUTPUT_ROOT / "04_trajectory_features" / "mimiciv_icd_sepsis_candidate_partial_trajectory_features.csv",
            ),
        ]
        for label, sql, path in exports:
            copy_query(conn, label, sql, path)
        for table in [
            "mimiciv_adult_first_icu_cohort",
            "mimiciv_outcomes",
            "mimiciv_core_labs_0_72h",
            "mimiciv_lab_window_features_long",
            "mimiciv_lab_window_features_wide",
            "mimiciv_partial_sic_scores",
            "mimiciv_partial_trajectory_features",
        ]:
            summary.append(
                {
                    "database": "mimiciv31",
                    "table": f"aris_sic.{table}",
                    "rows": str(query_scalar(conn, f"SELECT COUNT(*) FROM aris_sic.{table}")),
                }
            )


def process_eicu(summary: list[dict[str, str]]) -> None:
    with connect("eicu20") as conn:
        setup_connection(conn)
        for label, sql in [
            ("eicu reset aris_sic tables", EICU_DROP_SQL),
            ("eicu adult ICU cohort", EICU_COHORT_SQL),
            ("eicu outcomes", EICU_OUTCOMES_SQL),
            ("eicu lab mapping", EICU_LAB_MAPPING_SQL),
            ("eicu core labs 0-72h", EICU_CORE_LABS_SQL),
            (
                "eicu lab window features long",
                WINDOW_LONG_SQL_TEMPLATE.format(
                    table_prefix="eicu",
                    id_cols="patientunitstayid, patienthealthsystemstayid, uniquepid",
                ),
            ),
            (
                "eicu lab window features wide",
                WINDOW_WIDE_SQL_TEMPLATE.format(
                    table_prefix="eicu",
                    cohort_table="adult_icu_cohort",
                    base_id_cols="c.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid",
                    join_condition="c.patientunitstayid = f.patientunitstayid",
                ),
            ),
            (
                "eicu partial SIC scores",
                PARTIAL_SIC_SQL_TEMPLATE.format(
                    table_prefix="eicu",
                    id_select="patientunitstayid, patienthealthsystemstayid, uniquepid",
                ),
            ),
            (
                "eicu partial trajectory features",
                TRAJECTORY_SQL_TEMPLATE.format(
                    table_prefix="eicu",
                    id_group_cols="patientunitstayid, patienthealthsystemstayid, uniquepid",
                    wide_join_condition="w.patientunitstayid = f.patientunitstayid",
                ),
            ),
        ]:
            execute(conn, label, sql)
        exports = [
            (
                "eicu cohort",
                "SELECT * FROM aris_sic.eicu_adult_icu_cohort ORDER BY patientunitstayid",
                OUTPUT_ROOT / "06_external_validation" / "eicu_adult_icu_cohort.csv",
            ),
            (
                "eicu sepsis candidate cohort",
                "SELECT * FROM aris_sic.eicu_adult_icu_cohort WHERE sepsis_text_flag ORDER BY patientunitstayid",
                OUTPUT_ROOT / "06_external_validation" / "eicu_text_sepsis_candidate_cohort.csv",
            ),
            (
                "eicu outcomes",
                "SELECT * FROM aris_sic.eicu_outcomes ORDER BY patientunitstayid",
                OUTPUT_ROOT / "06_external_validation" / "eicu_outcomes.csv",
            ),
            (
                "eicu lab mapping",
                "SELECT * FROM aris_sic.eicu_lab_mapping ORDER BY variable, labname_lower",
                OUTPUT_ROOT / "00_metadata" / "eicu_lab_mapping.csv",
            ),
            (
                "eicu window features long",
                "SELECT * FROM aris_sic.eicu_lab_window_features_long ORDER BY patientunitstayid, window_name, variable",
                OUTPUT_ROOT / "06_external_validation" / "eicu_lab_window_features_0_72h_long.csv",
            ),
            (
                "eicu window features wide",
                "SELECT * FROM aris_sic.eicu_lab_window_features_wide ORDER BY patientunitstayid",
                OUTPUT_ROOT / "06_external_validation" / "eicu_lab_window_features_0_72h_wide.csv",
            ),
            (
                "eicu partial SIC scores",
                "SELECT * FROM aris_sic.eicu_partial_sic_scores ORDER BY patientunitstayid, window_name",
                OUTPUT_ROOT / "06_external_validation" / "eicu_partial_sic_scores.csv",
            ),
            (
                "eicu partial trajectory features",
                "SELECT * FROM aris_sic.eicu_partial_trajectory_features ORDER BY patientunitstayid",
                OUTPUT_ROOT / "06_external_validation" / "eicu_partial_trajectory_features.csv",
            ),
        ]
        for label, sql, path in exports:
            copy_query(conn, label, sql, path)
        for table in [
            "eicu_adult_icu_cohort",
            "eicu_outcomes",
            "eicu_core_labs_0_72h",
            "eicu_lab_window_features_long",
            "eicu_lab_window_features_wide",
            "eicu_partial_sic_scores",
            "eicu_partial_trajectory_features",
        ]:
            summary.append(
                {
                    "database": "eicu20",
                    "table": f"aris_sic.{table}",
                    "rows": str(query_scalar(conn, f"SELECT COUNT(*) FROM aris_sic.{table}")),
                }
            )


def process_sicdb(summary: list[dict[str, str]]) -> None:
    with connect("sicdb108") as conn:
        setup_connection(conn)
        for label, sql in [
            ("sicdb reset aris_sic tables", SICDB_DROP_SQL),
            ("sicdb cases cohort", SICDB_COHORT_SQL),
            ("sicdb outcomes", SICDB_OUTCOMES_SQL),
            ("sicdb lab mapping", SICDB_LAB_MAPPING_SQL),
            ("sicdb core labs 0-72h", SICDB_CORE_LABS_SQL),
            (
                "sicdb lab window features long",
                WINDOW_LONG_SQL_TEMPLATE.format(
                    table_prefix="sicdb",
                    id_cols="caseid, patientid",
                ),
            ),
            (
                "sicdb lab window features wide",
                WINDOW_WIDE_SQL_TEMPLATE.format(
                    table_prefix="sicdb",
                    cohort_table="cases_cohort",
                    base_id_cols="c.caseid, c.patientid",
                    join_condition="c.caseid = f.caseid",
                ),
            ),
            (
                "sicdb partial SIC scores",
                PARTIAL_SIC_SQL_TEMPLATE.format(
                    table_prefix="sicdb",
                    id_select="caseid, patientid",
                ),
            ),
            (
                "sicdb partial trajectory features",
                TRAJECTORY_SQL_TEMPLATE.format(
                    table_prefix="sicdb",
                    id_group_cols="caseid, patientid",
                    wide_join_condition="w.caseid = f.caseid",
                ),
            ),
        ]:
            execute(conn, label, sql)
        exports = [
            (
                "sicdb cohort",
                "SELECT * FROM aris_sic.sicdb_cases_cohort ORDER BY caseid",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_cases_cohort.csv",
            ),
            (
                "sicdb sepsis cohort",
                "SELECT * FROM aris_sic.sicdb_cases_cohort WHERE sepsis_admission_form_flag ORDER BY caseid",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_admission_form_sepsis_cohort.csv",
            ),
            (
                "sicdb outcomes",
                "SELECT * FROM aris_sic.sicdb_outcomes ORDER BY caseid",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_outcomes.csv",
            ),
            (
                "sicdb lab mapping",
                "SELECT * FROM aris_sic.sicdb_lab_mapping ORDER BY variable, laboratoryid",
                OUTPUT_ROOT / "00_metadata" / "sicdb_lab_mapping.csv",
            ),
            (
                "sicdb window features long",
                "SELECT * FROM aris_sic.sicdb_lab_window_features_long ORDER BY caseid, window_name, variable",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_lab_window_features_0_72h_long.csv",
            ),
            (
                "sicdb window features wide",
                "SELECT * FROM aris_sic.sicdb_lab_window_features_wide ORDER BY caseid",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_lab_window_features_0_72h_wide.csv",
            ),
            (
                "sicdb partial SIC scores",
                "SELECT * FROM aris_sic.sicdb_partial_sic_scores ORDER BY caseid, window_name",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_partial_sic_scores.csv",
            ),
            (
                "sicdb partial trajectory features",
                "SELECT * FROM aris_sic.sicdb_partial_trajectory_features ORDER BY caseid",
                OUTPUT_ROOT / "06_external_validation" / "sicdb_partial_trajectory_features.csv",
            ),
        ]
        for label, sql, path in exports:
            copy_query(conn, label, sql, path)
        for table in [
            "sicdb_cases_cohort",
            "sicdb_outcomes",
            "sicdb_core_labs_0_72h",
            "sicdb_lab_window_features_long",
            "sicdb_lab_window_features_wide",
            "sicdb_partial_sic_scores",
            "sicdb_partial_trajectory_features",
        ]:
            summary.append(
                {
                    "database": "sicdb108",
                    "table": f"aris_sic.{table}",
                    "rows": str(query_scalar(conn, f"SELECT COUNT(*) FROM aris_sic.{table}")),
                }
            )


def write_processing_notes(summary: list[dict[str, str]]) -> None:
    notes = OUTPUT_ROOT / "00_metadata" / "processing_notes.md"
    notes.write_text(
        f"""# SIC Trajectory Data Processing Notes

Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}

Source databases:
- MIMIC-IV v3.1: primary processed cohort and 0-72h laboratory trajectory features.
- eICU v2.0: external-validation style cohort, laboratory windows, partial scores.
- SICdb v1.0.8: external-validation style cohort, laboratory windows, partial scores.

Folder layout:
- `01_cohort/`: primary MIMIC-IV cohort tables.
- `02_window_features/`: MIMIC-IV 0-24h, 24-48h, 48-72h laboratory features in long and wide forms.
- `03_sic_scores/`: platelet + INR partial SIC coagulation scores by time window.
- `04_trajectory_features/`: derived trajectory deltas and trajectory groups.
- `05_outcomes/`: MIMIC-IV ICU, hospital, 28-day, and 90-day outcomes.
- `06_external_validation/`: eICU and SICdb cohort/features/outcomes using matched variable definitions where available.
- MIMIC-IV files prefixed with `mimiciv_icd_sepsis_candidate_` are the ICD-code filtered sepsis candidate subset for the SIC trajectory analysis.

Important interpretation notes:
- Raw imported PostgreSQL columns are text; this processing layer casts timestamps and numeric values explicitly.
- The MIMIC-IV primary cohort is adult first ICU stays. `mimiciv_icd_sepsis_candidate_cohort.csv` is ICD-code/text based and should be treated as a candidate sepsis cohort, not a strict Sepsis-3 cohort.
- The files named `partial_sic` contain platelet and INR components plus bilirubin/creatinine SOFA proxy components. They are not a complete official SIC score because complete SOFA requires respiratory, cardiovascular, CNS, and renal urine-output derivations.
- Time windows follow the experiment plan: W1 = 0-24h, W2 = 24-48h, W3 = 48-72h from ICU admission or dataset-specific ICU stay start.
- eICU and SICdb files are placed in `06_external_validation/` because their definitions and coding systems differ from MIMIC-IV and need final harmonization before formal model validation.
""",
        encoding="utf-8",
    )

    summary_path = OUTPUT_ROOT / "00_metadata" / "processing_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["database", "table", "rows"])
        writer.writeheader()
        writer.writerows(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SIC trajectory processed datasets.")
    parser.add_argument(
        "--resume-after-mimiciv-build",
        action="store_true",
        help="Skip rebuilding MIMIC-IV aris_sic tables and continue with CSV export plus eICU/SICdb.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    summary: list[dict[str, str]] = []
    process_mimiciv(summary, build=not args.resume_after_mimiciv_build)
    process_eicu(summary)
    process_sicdb(summary)
    write_processing_notes(summary)
    print(f"[{datetime.now():%H:%M:%S}] done", flush=True)


if __name__ == "__main__":
    main()
