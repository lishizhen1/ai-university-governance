from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import psycopg

# ── Path configuration (self-contained deliverable) ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (MIMIC_DIR, META_DIR, OUTPUT_DIR,
                     PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB_MIMIC)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = OUTPUT_DIR / "data_processed" / "07_full_sofa_sepsis3"
METADATA_ROOT = META_DIR


def connect():
    """Connect to MIMIC-IV PostgreSQL using environment variables."""
    return psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB_MIMIC,
        user=PG_USER,
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


def setup_connection(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        cur.execute("SET idle_in_transaction_session_timeout = 0")
        cur.execute("SET work_mem = '512MB'")
        cur.execute("SET max_parallel_workers_per_gather = 4")


DROP_SQL = """
CREATE SCHEMA IF NOT EXISTS aris_sic;
DROP TABLE IF EXISTS
    aris_sic.mimiciv_sepsis3_sic_trajectory_features,
    aris_sic.mimiciv_sic_scores_full_sofa,
    aris_sic.mimiciv_sepsis3_cohort,
    aris_sic.mimiciv_suspected_infection,
    aris_sic.mimiciv_suspected_infection_pairs,
    aris_sic.mimiciv_culture_events,
    aris_sic.mimiciv_antibiotic_events,
    aris_sic.mimiciv_full_sofa_scores,
    aris_sic.mimiciv_full_sofa_components,
    aris_sic.mimiciv_full_sofa_vaso_windows,
    aris_sic.mimiciv_full_sofa_urine_windows,
    aris_sic.mimiciv_full_sofa_vent_windows,
    aris_sic.mimiciv_full_sofa_gcs_by_time,
    aris_sic.mimiciv_full_sofa_chart_0_72h,
    aris_sic.mimiciv_full_sofa_labs_0_72h
CASCADE;
"""

PREFLIGHT_SQL = """
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_prescriptions_subject_hadm_start
    ON hosp.prescriptions(subject_id, hadm_id, starttime);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_micro_subject_hadm_chart
    ON hosp.microbiologyevents(subject_id, hadm_id, charttime);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_input_stay_time_item
    ON icu.inputevents(stay_id, starttime, endtime, itemid);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_output_stay_chart_item
    ON icu.outputevents(stay_id, charttime, itemid);
CREATE INDEX IF NOT EXISTS aris_idx_mimiciv_procedure_stay_time_item
    ON icu.procedureevents(stay_id, starttime, endtime, itemid);
ANALYZE hosp.prescriptions;
ANALYZE hosp.microbiologyevents;
ANALYZE icu.inputevents;
ANALYZE icu.outputevents;
ANALYZE icu.procedureevents;
"""

LABS_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_labs_0_72h AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    'platelet' AS variable,
    source_label,
    charttime,
    hours_from_icu,
    window_name,
    value_num
FROM aris_sic.mimiciv_core_labs_0_72h
WHERE variable = 'platelet'
UNION ALL
SELECT
    subject_id,
    hadm_id,
    stay_id,
    'bilirubin_total' AS variable,
    source_label,
    charttime,
    hours_from_icu,
    window_name,
    value_num
FROM aris_sic.mimiciv_core_labs_0_72h
WHERE variable = 'bilirubin_total'
UNION ALL
SELECT
    subject_id,
    hadm_id,
    stay_id,
    'creatinine' AS variable,
    source_label,
    charttime,
    hours_from_icu,
    window_name,
    value_num
FROM aris_sic.mimiciv_core_labs_0_72h
WHERE variable = 'creatinine'
UNION ALL
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    'pao2' AS variable,
    d.label AS source_label,
    NULLIF(le.charttime, '')::timestamp AS charttime,
    EXTRACT(EPOCH FROM (NULLIF(le.charttime, '')::timestamp - c.icu_intime)) / 3600.0 AS hours_from_icu,
    CASE
        WHEN EXTRACT(EPOCH FROM (NULLIF(le.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 24 THEN 'w1_0_24h'
        WHEN EXTRACT(EPOCH FROM (NULLIF(le.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 48 THEN 'w2_24_48h'
        ELSE 'w3_48_72h'
    END AS window_name,
    le.valuenum::numeric AS value_num
FROM hosp.labevents le
INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
    ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
INNER JOIN hosp.d_labitems d ON le.itemid = d.itemid
WHERE
    le.itemid = '50821'
    AND le.valuenum ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND le.charttime ~ '^\\d{4}-\\d{2}-\\d{2}'
    AND NULLIF(le.charttime, '')::timestamp >= c.icu_intime
    AND NULLIF(le.charttime, '')::timestamp < c.icu_intime + INTERVAL '72 hours'
    AND le.valuenum::numeric BETWEEN 20 AND 700;
"""

CHART_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_chart_0_72h AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    NULLIF(ce.charttime, '')::timestamp AS charttime,
    EXTRACT(EPOCH FROM (NULLIF(ce.charttime, '')::timestamp - c.icu_intime)) / 3600.0 AS hours_from_icu,
    CASE
        WHEN EXTRACT(EPOCH FROM (NULLIF(ce.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 24 THEN 'w1_0_24h'
        WHEN EXTRACT(EPOCH FROM (NULLIF(ce.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 48 THEN 'w2_24_48h'
        ELSE 'w3_48_72h'
    END AS window_name,
    ce.itemid,
    CASE
        WHEN ce.itemid IN ('220052', '220181') THEN 'map'
        WHEN ce.itemid = '223835' THEN 'fio2'
        WHEN ce.itemid IN ('229841', '229280', '226754', '227010') THEN 'fio2'
        WHEN ce.itemid = '220739' THEN 'gcs_eye'
        WHEN ce.itemid IN ('223900', '228112') THEN 'gcs_verbal'
        WHEN ce.itemid = '223901' THEN 'gcs_motor'
    END AS variable,
    ce.value,
    CASE WHEN ce.valuenum ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN ce.valuenum::numeric END AS value_num
FROM icu.chartevents ce
INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c
    ON ce.stay_id = c.stay_id
WHERE
    ce.itemid IN ('220052', '220181', '223835', '229841', '229280', '226754', '227010',
                  '220739', '223900', '228112', '223901')
    AND ce.charttime ~ '^\\d{4}-\\d{2}-\\d{2}'
    AND NULLIF(ce.charttime, '')::timestamp >= c.icu_intime
    AND NULLIF(ce.charttime, '')::timestamp < c.icu_intime + INTERVAL '72 hours';
"""

GCS_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_gcs_by_time AS
WITH components AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        charttime,
        window_name,
        MAX(CASE WHEN variable = 'gcs_eye' AND value_num BETWEEN 1 AND 4 THEN value_num END) AS gcs_eye,
        MAX(CASE WHEN variable = 'gcs_verbal' AND value_num BETWEEN 1 AND 5 THEN value_num END) AS gcs_verbal,
        MAX(CASE WHEN variable = 'gcs_motor' AND value_num BETWEEN 1 AND 6 THEN value_num END) AS gcs_motor
    FROM aris_sic.mimiciv_full_sofa_chart_0_72h
    WHERE variable LIKE 'gcs_%'
    GROUP BY subject_id, hadm_id, stay_id, charttime, window_name
)
SELECT
    *,
    CASE
        WHEN gcs_eye IS NOT NULL AND gcs_verbal IS NOT NULL AND gcs_motor IS NOT NULL
        THEN gcs_eye + gcs_verbal + gcs_motor
    END AS gcs_total
FROM components;
"""

VENT_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_vent_windows AS
WITH windows AS (
    SELECT stay_id, 'w1_0_24h' AS window_name, icu_intime AS window_start, icu_intime + INTERVAL '24 hours' AS window_end
    FROM aris_sic.mimiciv_adult_first_icu_cohort
    UNION ALL
    SELECT stay_id, 'w2_24_48h', icu_intime + INTERVAL '24 hours', icu_intime + INTERVAL '48 hours'
    FROM aris_sic.mimiciv_adult_first_icu_cohort
    UNION ALL
    SELECT stay_id, 'w3_48_72h', icu_intime + INTERVAL '48 hours', icu_intime + INTERVAL '72 hours'
    FROM aris_sic.mimiciv_adult_first_icu_cohort
),
vent AS (
    SELECT
        stay_id,
        NULLIF(starttime, '')::timestamp AS starttime,
        NULLIF(endtime, '')::timestamp AS endtime,
        itemid
    FROM icu.procedureevents
    WHERE itemid IN ('225792', '225794')
      AND starttime ~ '^\\d{4}-\\d{2}-\\d{2}'
      AND endtime ~ '^\\d{4}-\\d{2}-\\d{2}'
)
SELECT
    w.stay_id,
    w.window_name,
    BOOL_OR(v.itemid = '225792') AS invasive_ventilation,
    BOOL_OR(v.itemid IN ('225792', '225794')) AS any_positive_pressure_ventilation
FROM windows w
LEFT JOIN vent v
    ON w.stay_id = v.stay_id
   AND v.starttime < w.window_end
   AND v.endtime > w.window_start
GROUP BY w.stay_id, w.window_name;
"""

URINE_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_urine_windows AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    CASE
        WHEN EXTRACT(EPOCH FROM (NULLIF(o.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 24 THEN 'w1_0_24h'
        WHEN EXTRACT(EPOCH FROM (NULLIF(o.charttime, '')::timestamp - c.icu_intime)) / 3600.0 < 48 THEN 'w2_24_48h'
        ELSE 'w3_48_72h'
    END AS window_name,
    SUM(o.value::numeric) AS urine_output_ml
FROM icu.outputevents o
INNER JOIN aris_sic.mimiciv_adult_first_icu_cohort c ON o.stay_id = c.stay_id
WHERE
    o.itemid IN ('226559', '226560', '226561', '226563', '226564', '226565', '226566', '226567', '226627', '226631')
    AND o.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
    AND o.charttime ~ '^\\d{4}-\\d{2}-\\d{2}'
    AND NULLIF(o.charttime, '')::timestamp >= c.icu_intime
    AND NULLIF(o.charttime, '')::timestamp < c.icu_intime + INTERVAL '72 hours'
    AND o.value::numeric >= 0
GROUP BY c.subject_id, c.hadm_id, c.stay_id, window_name;
"""

VASO_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_vaso_windows AS
WITH windows AS (
    SELECT subject_id, hadm_id, stay_id, 'w1_0_24h' AS window_name, icu_intime AS window_start, icu_intime + INTERVAL '24 hours' AS window_end
    FROM aris_sic.mimiciv_adult_first_icu_cohort
    UNION ALL
    SELECT subject_id, hadm_id, stay_id, 'w2_24_48h', icu_intime + INTERVAL '24 hours', icu_intime + INTERVAL '48 hours'
    FROM aris_sic.mimiciv_adult_first_icu_cohort
    UNION ALL
    SELECT subject_id, hadm_id, stay_id, 'w3_48_72h', icu_intime + INTERVAL '48 hours', icu_intime + INTERVAL '72 hours'
    FROM aris_sic.mimiciv_adult_first_icu_cohort
),
vaso AS (
    SELECT
        stay_id,
        NULLIF(starttime, '')::timestamp AS starttime,
        NULLIF(endtime, '')::timestamp AS endtime,
        itemid,
        CASE
            WHEN rate ~ '^-?[0-9]+(\\.[0-9]+)?$' AND rateuom = 'mcg/kg/min' THEN rate::numeric
        END AS rate_mcg_kg_min,
        rateuom
    FROM icu.inputevents
    WHERE itemid IN ('221906', '221289', '229617', '221662', '221653', '222315', '221749', '229630', '229631', '229632')
      AND starttime ~ '^\\d{4}-\\d{2}-\\d{2}'
      AND endtime ~ '^\\d{4}-\\d{2}-\\d{2}'
      AND statusdescription != 'Rewritten'
)
SELECT
    w.subject_id,
    w.hadm_id,
    w.stay_id,
    w.window_name,
    MAX(CASE WHEN v.itemid = '221906' THEN v.rate_mcg_kg_min END) AS norepinephrine_max_mcg_kg_min,
    MAX(CASE WHEN v.itemid IN ('221289', '229617') THEN v.rate_mcg_kg_min END) AS epinephrine_max_mcg_kg_min,
    MAX(CASE WHEN v.itemid = '221662' THEN v.rate_mcg_kg_min END) AS dopamine_max_mcg_kg_min,
    BOOL_OR(v.itemid = '221653') AS dobutamine_any,
    BOOL_OR(v.itemid = '222315') AS vasopressin_any,
    MAX(CASE WHEN v.itemid IN ('221749', '229630', '229631', '229632') THEN v.rate_mcg_kg_min END) AS phenylephrine_max_mcg_kg_min
FROM windows w
LEFT JOIN vaso v
    ON w.stay_id = v.stay_id
   AND v.starttime < w.window_end
   AND v.endtime > w.window_start
GROUP BY w.subject_id, w.hadm_id, w.stay_id, w.window_name;
"""

COMPONENTS_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_components AS
WITH windows AS (
    SELECT subject_id, hadm_id, stay_id, 'w1_0_24h' AS window_name FROM aris_sic.mimiciv_adult_first_icu_cohort
    UNION ALL
    SELECT subject_id, hadm_id, stay_id, 'w2_24_48h' FROM aris_sic.mimiciv_adult_first_icu_cohort
    UNION ALL
    SELECT subject_id, hadm_id, stay_id, 'w3_48_72h' FROM aris_sic.mimiciv_adult_first_icu_cohort
),
lab AS (
    SELECT
        subject_id, hadm_id, stay_id, window_name,
        MIN(CASE WHEN variable = 'pao2' THEN value_num END) AS pao2_min,
        MIN(CASE WHEN variable = 'platelet' THEN value_num END) AS platelet_min,
        MAX(CASE WHEN variable = 'bilirubin_total' THEN value_num END) AS bilirubin_total_max,
        MAX(CASE WHEN variable = 'creatinine' THEN value_num END) AS creatinine_max
    FROM aris_sic.mimiciv_full_sofa_labs_0_72h
    GROUP BY subject_id, hadm_id, stay_id, window_name
),
chart AS (
    SELECT
        subject_id, hadm_id, stay_id, window_name,
        MIN(CASE WHEN variable = 'map' AND value_num BETWEEN 20 AND 200 THEN value_num END) AS map_min,
        MAX(CASE
            WHEN variable = 'fio2' AND value_num > 1.5 AND value_num <= 100 THEN value_num / 100.0
            WHEN variable = 'fio2' AND value_num BETWEEN 0.21 AND 1.0 THEN value_num
        END) AS fio2_max
    FROM aris_sic.mimiciv_full_sofa_chart_0_72h
    GROUP BY subject_id, hadm_id, stay_id, window_name
),
gcs AS (
    SELECT
        subject_id, hadm_id, stay_id, window_name,
        MIN(gcs_total) AS gcs_min
    FROM aris_sic.mimiciv_full_sofa_gcs_by_time
    WHERE gcs_total BETWEEN 3 AND 15
    GROUP BY subject_id, hadm_id, stay_id, window_name
)
SELECT
    w.subject_id,
    w.hadm_id,
    w.stay_id,
    w.window_name,
    lab.pao2_min,
    chart.fio2_max,
    CASE
        WHEN lab.pao2_min IS NOT NULL AND chart.fio2_max IS NOT NULL AND chart.fio2_max > 0
        THEN lab.pao2_min / chart.fio2_max
    END AS pao2_fio2_ratio,
    COALESCE(v.invasive_ventilation, FALSE) AS invasive_ventilation,
    lab.platelet_min,
    lab.bilirubin_total_max,
    chart.map_min,
    vaso.norepinephrine_max_mcg_kg_min,
    vaso.epinephrine_max_mcg_kg_min,
    vaso.dopamine_max_mcg_kg_min,
    COALESCE(vaso.dobutamine_any, FALSE) AS dobutamine_any,
    COALESCE(vaso.vasopressin_any, FALSE) AS vasopressin_any,
    vaso.phenylephrine_max_mcg_kg_min,
    gcs.gcs_min,
    lab.creatinine_max,
    urine.urine_output_ml,
    CASE
        WHEN lab.pao2_min IS NULL OR chart.fio2_max IS NULL THEN NULL
        WHEN lab.pao2_min / chart.fio2_max < 100 AND COALESCE(v.invasive_ventilation, FALSE) THEN 4
        WHEN lab.pao2_min / chart.fio2_max < 200 AND COALESCE(v.invasive_ventilation, FALSE) THEN 3
        WHEN lab.pao2_min / chart.fio2_max < 300 THEN 2
        WHEN lab.pao2_min / chart.fio2_max < 400 THEN 1
        ELSE 0
    END AS sofa_respiration,
    CASE
        WHEN lab.platelet_min IS NULL THEN NULL
        WHEN lab.platelet_min < 20 THEN 4
        WHEN lab.platelet_min < 50 THEN 3
        WHEN lab.platelet_min < 100 THEN 2
        WHEN lab.platelet_min < 150 THEN 1
        ELSE 0
    END AS sofa_coagulation,
    CASE
        WHEN lab.bilirubin_total_max IS NULL THEN NULL
        WHEN lab.bilirubin_total_max >= 12.0 THEN 4
        WHEN lab.bilirubin_total_max >= 6.0 THEN 3
        WHEN lab.bilirubin_total_max >= 2.0 THEN 2
        WHEN lab.bilirubin_total_max >= 1.2 THEN 1
        ELSE 0
    END AS sofa_liver,
    CASE
        WHEN COALESCE(vaso.dopamine_max_mcg_kg_min, 0) > 15
          OR COALESCE(vaso.epinephrine_max_mcg_kg_min, 0) > 0.1
          OR COALESCE(vaso.norepinephrine_max_mcg_kg_min, 0) > 0.1 THEN 4
        WHEN COALESCE(vaso.dopamine_max_mcg_kg_min, 0) > 5
          OR COALESCE(vaso.epinephrine_max_mcg_kg_min, 0) > 0
          OR COALESCE(vaso.norepinephrine_max_mcg_kg_min, 0) > 0 THEN 3
        WHEN COALESCE(vaso.dopamine_max_mcg_kg_min, 0) > 0
          OR COALESCE(vaso.dobutamine_any, FALSE) THEN 2
        WHEN chart.map_min IS NOT NULL AND chart.map_min < 70 THEN 1
        WHEN chart.map_min IS NOT NULL
          OR vaso.norepinephrine_max_mcg_kg_min IS NOT NULL
          OR vaso.epinephrine_max_mcg_kg_min IS NOT NULL
          OR vaso.dopamine_max_mcg_kg_min IS NOT NULL
          OR COALESCE(vaso.dobutamine_any, FALSE) THEN 0
        ELSE NULL
    END AS sofa_cardiovascular,
    CASE
        WHEN gcs.gcs_min IS NULL THEN NULL
        WHEN gcs.gcs_min < 6 THEN 4
        WHEN gcs.gcs_min < 10 THEN 3
        WHEN gcs.gcs_min < 13 THEN 2
        WHEN gcs.gcs_min < 15 THEN 1
        ELSE 0
    END AS sofa_cns,
    GREATEST(
        CASE
            WHEN lab.creatinine_max IS NULL THEN NULL
            WHEN lab.creatinine_max >= 5.0 THEN 4
            WHEN lab.creatinine_max >= 3.5 THEN 3
            WHEN lab.creatinine_max >= 2.0 THEN 2
            WHEN lab.creatinine_max >= 1.2 THEN 1
            ELSE 0
        END,
        CASE
            WHEN urine.urine_output_ml IS NULL THEN NULL
            WHEN urine.urine_output_ml < 200 THEN 4
            WHEN urine.urine_output_ml < 500 THEN 3
            ELSE 0
        END
    ) AS sofa_renal
FROM windows w
LEFT JOIN lab
    ON w.subject_id = lab.subject_id AND w.hadm_id = lab.hadm_id AND w.stay_id = lab.stay_id AND w.window_name = lab.window_name
LEFT JOIN chart
    ON w.subject_id = chart.subject_id AND w.hadm_id = chart.hadm_id AND w.stay_id = chart.stay_id AND w.window_name = chart.window_name
LEFT JOIN aris_sic.mimiciv_full_sofa_vent_windows v
    ON w.stay_id = v.stay_id AND w.window_name = v.window_name
LEFT JOIN aris_sic.mimiciv_full_sofa_vaso_windows vaso
    ON w.subject_id = vaso.subject_id AND w.hadm_id = vaso.hadm_id AND w.stay_id = vaso.stay_id AND w.window_name = vaso.window_name
LEFT JOIN gcs
    ON w.subject_id = gcs.subject_id AND w.hadm_id = gcs.hadm_id AND w.stay_id = gcs.stay_id AND w.window_name = gcs.window_name
LEFT JOIN aris_sic.mimiciv_full_sofa_urine_windows urine
    ON w.subject_id = urine.subject_id AND w.hadm_id = urine.hadm_id AND w.stay_id = urine.stay_id AND w.window_name = urine.window_name;
"""

SCORES_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_full_sofa_scores AS
SELECT
    *,
    (CASE WHEN sofa_respiration IS NOT NULL THEN 1 ELSE 0 END
     + CASE WHEN sofa_coagulation IS NOT NULL THEN 1 ELSE 0 END
     + CASE WHEN sofa_liver IS NOT NULL THEN 1 ELSE 0 END
     + CASE WHEN sofa_cardiovascular IS NOT NULL THEN 1 ELSE 0 END
     + CASE WHEN sofa_cns IS NOT NULL THEN 1 ELSE 0 END
     + CASE WHEN sofa_renal IS NOT NULL THEN 1 ELSE 0 END) AS sofa_components_available,
    COALESCE(sofa_respiration, 0)
      + COALESCE(sofa_coagulation, 0)
      + COALESCE(sofa_liver, 0)
      + COALESCE(sofa_cardiovascular, 0)
      + COALESCE(sofa_cns, 0)
      + COALESCE(sofa_renal, 0) AS sofa_total_assuming_missing_normal,
    CASE
        WHEN sofa_respiration IS NOT NULL
         AND sofa_coagulation IS NOT NULL
         AND sofa_liver IS NOT NULL
         AND sofa_cardiovascular IS NOT NULL
         AND sofa_cns IS NOT NULL
         AND sofa_renal IS NOT NULL
        THEN sofa_respiration + sofa_coagulation + sofa_liver + sofa_cardiovascular + sofa_cns + sofa_renal
    END AS sofa_total_complete_only,
    COALESCE(sofa_respiration, 0)
      + COALESCE(sofa_liver, 0)
      + COALESCE(sofa_cardiovascular, 0)
      + COALESCE(sofa_renal, 0) AS sofa_4systems_for_sic
FROM aris_sic.mimiciv_full_sofa_components;
"""

ANTIBIOTIC_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_antibiotic_events AS
SELECT
    subject_id,
    hadm_id,
    NULLIF(starttime, '')::timestamp AS antibiotic_time,
    drug,
    route
FROM hosp.prescriptions
WHERE
    starttime ~ '^\\d{4}-\\d{2}-\\d{2}'
    AND lower(drug) ~ '(vancomycin|cefazolin|cefepime|ceftriaxone|ceftazidime|ceftaroline|cefpodoxime|cefotaxime|cefuroxime|piperacillin|tazobactam|meropenem|imipenem|ertapenem|aztreonam|ciprofloxacin|levofloxacin|moxifloxacin|azithromycin|clarithromycin|doxycycline|minocycline|linezolid|metronidazole|clindamycin|gentamicin|tobramycin|amikacin|ampicillin|amoxicillin|sulbactam|penicillin|oxacillin|nafcillin|trimethoprim|sulfameth|bactrim|daptomycin|tigecycline|colistin|polymyxin|ceftolozane|avibactam)'
    AND lower(drug) !~ '(ophth|otic|cream|ointment|desensitization|antibiotic lock|enema|topical|irrigation)'
    AND lower(COALESCE(route, '')) !~ '(ophth|otic|topical|nebul|inh|lock|enema|irrig)';
"""

CULTURE_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_culture_events AS
SELECT
    subject_id,
    hadm_id,
    COALESCE(
        CASE WHEN charttime ~ '^\\d{4}-\\d{2}-\\d{2}' THEN NULLIF(charttime, '')::timestamp END,
        CASE WHEN chartdate ~ '^\\d{4}-\\d{2}-\\d{2}' THEN NULLIF(chartdate, '')::timestamp END
    ) AS culture_time,
    spec_type_desc,
    test_name,
    org_name
FROM hosp.microbiologyevents
WHERE COALESCE(
        CASE WHEN charttime ~ '^\\d{4}-\\d{2}-\\d{2}' THEN NULLIF(charttime, '')::timestamp END,
        CASE WHEN chartdate ~ '^\\d{4}-\\d{2}-\\d{2}' THEN NULLIF(chartdate, '')::timestamp END
    ) IS NOT NULL;
"""

SUSPECTED_INFECTION_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_suspected_infection_pairs AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    a.antibiotic_time,
    m.culture_time,
    LEAST(a.antibiotic_time, m.culture_time) AS suspected_infection_time,
    a.drug AS antibiotic_drug,
    a.route AS antibiotic_route,
    m.spec_type_desc,
    m.test_name,
    m.org_name,
    EXTRACT(EPOCH FROM (LEAST(a.antibiotic_time, m.culture_time) - c.icu_intime)) / 3600.0 AS suspected_infection_hour_from_icu
FROM aris_sic.mimiciv_adult_first_icu_cohort c
INNER JOIN aris_sic.mimiciv_antibiotic_events a
    ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
INNER JOIN aris_sic.mimiciv_culture_events m
    ON c.subject_id = m.subject_id AND c.hadm_id = m.hadm_id
WHERE
    m.culture_time BETWEEN a.antibiotic_time - INTERVAL '72 hours'
                       AND a.antibiotic_time + INTERVAL '24 hours'
    AND LEAST(a.antibiotic_time, m.culture_time) >= c.hospital_admittime
    AND LEAST(a.antibiotic_time, m.culture_time) <= COALESCE(c.hospital_dischtime, c.icu_outtime)
    AND LEAST(a.antibiotic_time, m.culture_time) >= c.icu_intime - INTERVAL '48 hours'
    AND LEAST(a.antibiotic_time, m.culture_time) < c.icu_intime + INTERVAL '72 hours';

CREATE UNLOGGED TABLE aris_sic.mimiciv_suspected_infection AS
SELECT *
FROM (
    SELECT
        *,
        suspected_infection_time BETWEEN
            (SELECT icu_intime - INTERVAL '48 hours' FROM aris_sic.mimiciv_adult_first_icu_cohort c WHERE c.stay_id = p.stay_id)
            AND
            (SELECT icu_intime + INTERVAL '24 hours' FROM aris_sic.mimiciv_adult_first_icu_cohort c WHERE c.stay_id = p.stay_id)
            AS suspected_infection_near_icu_admit,
        ROW_NUMBER() OVER (
            PARTITION BY stay_id
            ORDER BY
                CASE
                    WHEN suspected_infection_hour_from_icu BETWEEN -48 AND 24 THEN 0
                    ELSE 1
                END,
                suspected_infection_time,
                antibiotic_time,
                culture_time
        ) AS rn
    FROM aris_sic.mimiciv_suspected_infection_pairs p
) x
WHERE rn = 1;
"""

SEPSIS3_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_sepsis3_cohort AS
WITH sofa AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN sofa_total_assuming_missing_normal END) AS sofa_total_w1,
        MAX(sofa_total_assuming_missing_normal) AS sofa_total_max_0_72h,
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN sofa_components_available END) AS sofa_components_available_w1,
        MAX(sofa_components_available) AS sofa_components_available_max_0_72h
    FROM aris_sic.mimiciv_full_sofa_scores
    GROUP BY subject_id, hadm_id, stay_id
)
SELECT
    c.*,
    si.suspected_infection_time,
    si.suspected_infection_hour_from_icu,
    si.suspected_infection_near_icu_admit,
    si.antibiotic_time,
    si.culture_time,
    si.antibiotic_drug,
    si.antibiotic_route,
    si.spec_type_desc,
    si.test_name,
    si.org_name,
    sofa.sofa_total_w1,
    sofa.sofa_total_max_0_72h,
    sofa.sofa_components_available_w1,
    sofa.sofa_components_available_max_0_72h,
    COALESCE(si.suspected_infection_near_icu_admit, FALSE)
      AND COALESCE(sofa.sofa_total_w1, 0) >= 2 AS sepsis3_icu_admit_flag,
    si.suspected_infection_time IS NOT NULL
      AND COALESCE(sofa.sofa_total_max_0_72h, 0) >= 2 AS sepsis3_0_72h_flag
FROM aris_sic.mimiciv_adult_first_icu_cohort c
LEFT JOIN aris_sic.mimiciv_suspected_infection si
    ON c.subject_id = si.subject_id AND c.hadm_id = si.hadm_id AND c.stay_id = si.stay_id
LEFT JOIN sofa
    ON c.subject_id = sofa.subject_id AND c.hadm_id = sofa.hadm_id AND c.stay_id = sofa.stay_id;
"""

SIC_FULL_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_sic_scores_full_sofa AS
SELECT
    s.subject_id,
    s.hadm_id,
    s.stay_id,
    s.window_name,
    p.platelet_min,
    p.inr_max,
    s.sofa_respiration,
    s.sofa_liver,
    s.sofa_cardiovascular,
    s.sofa_renal,
    s.sofa_4systems_for_sic,
    CASE
        WHEN p.platelet_min IS NULL THEN NULL
        WHEN p.platelet_min < 100 THEN 2
        WHEN p.platelet_min < 150 THEN 1
        ELSE 0
    END AS sic_platelet_component,
    CASE
        WHEN p.inr_max IS NULL THEN NULL
        WHEN p.inr_max > 1.4 THEN 2
        WHEN p.inr_max >= 1.2 THEN 1
        ELSE 0
    END AS sic_inr_component,
    CASE
        WHEN s.sofa_4systems_for_sic IS NULL THEN NULL
        WHEN s.sofa_4systems_for_sic >= 2 THEN 2
        WHEN s.sofa_4systems_for_sic = 1 THEN 1
        ELSE 0
    END AS sic_organ_sofa_component,
    CASE
        WHEN p.platelet_min IS NULL OR p.inr_max IS NULL THEN NULL
        ELSE
            CASE WHEN p.platelet_min < 100 THEN 2 WHEN p.platelet_min < 150 THEN 1 ELSE 0 END
            + CASE WHEN p.inr_max > 1.4 THEN 2 WHEN p.inr_max >= 1.2 THEN 1 ELSE 0 END
            + CASE WHEN s.sofa_4systems_for_sic >= 2 THEN 2 WHEN s.sofa_4systems_for_sic = 1 THEN 1 ELSE 0 END
    END AS sic_total_score,
    CASE
        WHEN p.platelet_min IS NULL OR p.inr_max IS NULL THEN NULL
        ELSE (
            CASE WHEN p.platelet_min < 100 THEN 2 WHEN p.platelet_min < 150 THEN 1 ELSE 0 END
            + CASE WHEN p.inr_max > 1.4 THEN 2 WHEN p.inr_max >= 1.2 THEN 1 ELSE 0 END
            + CASE WHEN s.sofa_4systems_for_sic >= 2 THEN 2 WHEN s.sofa_4systems_for_sic = 1 THEN 1 ELSE 0 END
        ) >= 4
    END AS sic_flag
FROM aris_sic.mimiciv_full_sofa_scores s
LEFT JOIN aris_sic.mimiciv_partial_sic_scores p
    ON s.subject_id = p.subject_id AND s.hadm_id = p.hadm_id AND s.stay_id = p.stay_id AND s.window_name = p.window_name;
"""

TRAJECTORY_SQL = """
CREATE UNLOGGED TABLE aris_sic.mimiciv_sepsis3_sic_trajectory_features AS
WITH sic_wide AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN sic_total_score END) AS w1_sic_total_score,
        MAX(CASE WHEN window_name = 'w2_24_48h' THEN sic_total_score END) AS w2_sic_total_score,
        MAX(CASE WHEN window_name = 'w3_48_72h' THEN sic_total_score END) AS w3_sic_total_score,
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN sic_flag::int END)::boolean AS w1_sic_flag,
        MAX(CASE WHEN window_name = 'w2_24_48h' THEN sic_flag::int END)::boolean AS w2_sic_flag,
        MAX(CASE WHEN window_name = 'w3_48_72h' THEN sic_flag::int END)::boolean AS w3_sic_flag
    FROM aris_sic.mimiciv_sic_scores_full_sofa
    GROUP BY subject_id, hadm_id, stay_id
),
sofa_wide AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        MAX(CASE WHEN window_name = 'w1_0_24h' THEN sofa_total_assuming_missing_normal END) AS w1_sofa_total,
        MAX(CASE WHEN window_name = 'w2_24_48h' THEN sofa_total_assuming_missing_normal END) AS w2_sofa_total,
        MAX(CASE WHEN window_name = 'w3_48_72h' THEN sofa_total_assuming_missing_normal END) AS w3_sofa_total
    FROM aris_sic.mimiciv_full_sofa_scores
    GROUP BY subject_id, hadm_id, stay_id
)
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.sepsis3_icu_admit_flag,
    c.sepsis3_0_72h_flag,
    c.suspected_infection_time,
    c.suspected_infection_hour_from_icu,
    s.w1_sofa_total,
    s.w2_sofa_total,
    s.w3_sofa_total,
    sw.w1_sic_total_score,
    sw.w2_sic_total_score,
    sw.w3_sic_total_score,
    sw.w1_sic_flag,
    sw.w2_sic_flag,
    sw.w3_sic_flag,
    s.w3_sofa_total - s.w1_sofa_total AS sofa_w3_minus_w1,
    sw.w3_sic_total_score - sw.w1_sic_total_score AS sic_w3_minus_w1,
    CASE
        WHEN GREATEST(sw.w1_sic_total_score, sw.w2_sic_total_score, sw.w3_sic_total_score) IS NULL THEN 'no_sic_score_available'
        WHEN COALESCE(sw.w1_sic_flag, FALSE) AND COALESCE(sw.w2_sic_flag, FALSE) AND COALESCE(sw.w3_sic_flag, FALSE) THEN 'persistent_sic'
        WHEN COALESCE(sw.w1_sic_flag, FALSE) AND NOT COALESCE(sw.w3_sic_flag, FALSE) THEN 'resolving_sic'
        WHEN NOT COALESCE(sw.w1_sic_flag, FALSE) AND COALESCE(sw.w3_sic_flag, FALSE) THEN 'new_or_worsening_sic'
        WHEN NOT COALESCE(sw.w1_sic_flag, FALSE) AND NOT COALESCE(sw.w2_sic_flag, FALSE) AND NOT COALESCE(sw.w3_sic_flag, FALSE) THEN 'no_sic'
        ELSE 'mixed_or_incomplete'
    END AS sic_trajectory_group
FROM aris_sic.mimiciv_sepsis3_cohort c
LEFT JOIN sofa_wide s
    ON c.subject_id = s.subject_id AND c.hadm_id = s.hadm_id AND c.stay_id = s.stay_id
LEFT JOIN sic_wide sw
    ON c.subject_id = sw.subject_id AND c.hadm_id = sw.hadm_id AND c.stay_id = sw.stay_id;
"""


def build_tables(conn, skip_heavy: bool = False) -> None:
    execute(conn, "reset full SOFA/Sepsis-3 tables", DROP_SQL)
    execute(conn, "preflight indexes/statistics", PREFLIGHT_SQL)
    if not skip_heavy:
        execute(conn, "full SOFA labs 0-72h including PaO2", LABS_SQL)
        execute(conn, "full SOFA chart subset 0-72h", CHART_SQL)
    else:
        print(f"[{datetime.now():%H:%M:%S}] heavy lab/chart extraction skipped", flush=True)
    for label, sql in [
        ("GCS by chart time", GCS_SQL),
        ("ventilation windows", VENT_SQL),
        ("urine-output windows", URINE_SQL),
        ("vasopressor windows", VASO_SQL),
        ("SOFA components", COMPONENTS_SQL),
        ("SOFA scores", SCORES_SQL),
        ("antibiotic events", ANTIBIOTIC_SQL),
        ("culture events", CULTURE_SQL),
        ("suspected infection", SUSPECTED_INFECTION_SQL),
        ("Sepsis-3 cohort", SEPSIS3_SQL),
        ("SIC scores with full SOFA", SIC_FULL_SQL),
        ("Sepsis-3/SIC trajectory features", TRAJECTORY_SQL),
    ]:
        execute(conn, label, sql)
    for table in [
        "mimiciv_full_sofa_labs_0_72h",
        "mimiciv_full_sofa_chart_0_72h",
        "mimiciv_full_sofa_scores",
        "mimiciv_suspected_infection",
        "mimiciv_sepsis3_cohort",
        "mimiciv_sic_scores_full_sofa",
    ]:
        execute(conn, f"analyze {table}", f"ANALYZE aris_sic.{table}")


def export_outputs(conn) -> list[dict[str, str]]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    exports = [
        (
            "MIMIC-IV full SOFA scores",
            "SELECT * FROM aris_sic.mimiciv_full_sofa_scores ORDER BY subject_id, stay_id, window_name",
            OUTPUT_ROOT / "mimiciv_full_sofa_scores_0_72h.csv",
        ),
        (
            "MIMIC-IV suspected infection",
            "SELECT * FROM aris_sic.mimiciv_suspected_infection ORDER BY subject_id, stay_id",
            OUTPUT_ROOT / "mimiciv_suspected_infection.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 cohort all stays",
            "SELECT * FROM aris_sic.mimiciv_sepsis3_cohort ORDER BY subject_id, stay_id",
            OUTPUT_ROOT / "mimiciv_sepsis3_cohort_all_adult_first_icu.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 ICU-admit cohort",
            "SELECT * FROM aris_sic.mimiciv_sepsis3_cohort WHERE sepsis3_icu_admit_flag ORDER BY subject_id, stay_id",
            OUTPUT_ROOT / "mimiciv_sepsis3_icu_admit_cohort.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 0-72h cohort",
            "SELECT * FROM aris_sic.mimiciv_sepsis3_cohort WHERE sepsis3_0_72h_flag ORDER BY subject_id, stay_id",
            OUTPUT_ROOT / "mimiciv_sepsis3_0_72h_cohort.csv",
        ),
        (
            "MIMIC-IV SIC scores with full SOFA",
            "SELECT * FROM aris_sic.mimiciv_sic_scores_full_sofa ORDER BY subject_id, stay_id, window_name",
            OUTPUT_ROOT / "mimiciv_sic_scores_full_sofa.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 SIC trajectory features",
            "SELECT * FROM aris_sic.mimiciv_sepsis3_sic_trajectory_features ORDER BY subject_id, stay_id",
            OUTPUT_ROOT / "mimiciv_sepsis3_sic_trajectory_features.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 ICU-admit SOFA scores",
            """
            SELECT s.*
            FROM aris_sic.mimiciv_full_sofa_scores s
            INNER JOIN aris_sic.mimiciv_sepsis3_cohort c
                ON s.subject_id = c.subject_id AND s.hadm_id = c.hadm_id AND s.stay_id = c.stay_id
            WHERE c.sepsis3_icu_admit_flag
            ORDER BY s.subject_id, s.stay_id, s.window_name
            """,
            OUTPUT_ROOT / "mimiciv_sepsis3_icu_admit_full_sofa_scores_0_72h.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 ICU-admit SIC scores",
            """
            SELECT s.*
            FROM aris_sic.mimiciv_sic_scores_full_sofa s
            INNER JOIN aris_sic.mimiciv_sepsis3_cohort c
                ON s.subject_id = c.subject_id AND s.hadm_id = c.hadm_id AND s.stay_id = c.stay_id
            WHERE c.sepsis3_icu_admit_flag
            ORDER BY s.subject_id, s.stay_id, s.window_name
            """,
            OUTPUT_ROOT / "mimiciv_sepsis3_icu_admit_sic_scores_full_sofa.csv",
        ),
        (
            "MIMIC-IV Sepsis-3 ICU-admit trajectory features",
            """
            SELECT *
            FROM aris_sic.mimiciv_sepsis3_sic_trajectory_features
            WHERE sepsis3_icu_admit_flag
            ORDER BY subject_id, stay_id
            """,
            OUTPUT_ROOT / "mimiciv_sepsis3_icu_admit_sic_trajectory_features.csv",
        ),
    ]
    for label, sql, path in exports:
        copy_query(conn, label, sql, path)

    mapping_sql = """
    SELECT * FROM (VALUES
        ('respiration', 'pao2', 'hosp.labevents itemid 50821 pO2'),
        ('respiration', 'fio2', 'icu.chartevents FiO2 itemids 223835, 229841, 229280, 226754, 227010'),
        ('respiration', 'invasive_ventilation', 'icu.procedureevents itemid 225792'),
        ('coagulation', 'platelet', 'hosp.labevents platelet count via aris_sic.mimiciv_core_labs_0_72h'),
        ('liver', 'bilirubin_total', 'hosp.labevents bilirubin total via aris_sic.mimiciv_core_labs_0_72h'),
        ('cardiovascular', 'map', 'icu.chartevents itemids 220052, 220181'),
        ('cardiovascular', 'vasopressors', 'icu.inputevents norepinephrine/epinephrine/dopamine/dobutamine itemids'),
        ('cns', 'gcs', 'icu.chartevents GCS eye/verbal/motor itemids 220739, 223900/228112, 223901'),
        ('renal', 'creatinine', 'hosp.labevents creatinine via aris_sic.mimiciv_core_labs_0_72h'),
        ('renal', 'urine_output', 'icu.outputevents urine output itemids')
    ) AS m(sofa_component, variable, source_definition)
    ORDER BY sofa_component, variable
    """
    copy_query(
        conn,
        "MIMIC-IV full SOFA variable mapping",
        mapping_sql,
        METADATA_ROOT / "mimiciv_full_sofa_variable_mapping.csv",
    )

    summary_tables = [
        "mimiciv_full_sofa_labs_0_72h",
        "mimiciv_full_sofa_chart_0_72h",
        "mimiciv_full_sofa_scores",
        "mimiciv_antibiotic_events",
        "mimiciv_culture_events",
        "mimiciv_suspected_infection",
        "mimiciv_sepsis3_cohort",
        "mimiciv_sic_scores_full_sofa",
        "mimiciv_sepsis3_sic_trajectory_features",
    ]
    summary: list[dict[str, str]] = []
    for table in summary_tables:
        summary.append(
            {
                "database": "mimiciv31",
                "table": f"aris_sic.{table}",
                "rows": str(query_scalar(conn, f"SELECT COUNT(*) FROM aris_sic.{table}")),
            }
        )
    summary.extend(
        [
            {
                "database": "mimiciv31",
                "table": "sepsis3_icu_admit_flag",
                "rows": str(query_scalar(conn, "SELECT COUNT(*) FROM aris_sic.mimiciv_sepsis3_cohort WHERE sepsis3_icu_admit_flag")),
            },
            {
                "database": "mimiciv31",
                "table": "sepsis3_0_72h_flag",
                "rows": str(query_scalar(conn, "SELECT COUNT(*) FROM aris_sic.mimiciv_sepsis3_cohort WHERE sepsis3_0_72h_flag")),
            },
        ]
    )
    summary_path = METADATA_ROOT / "full_sofa_sepsis3_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["database", "table", "rows"])
        writer.writeheader()
        writer.writerows(summary)
    return summary


def write_notes() -> None:
    notes = METADATA_ROOT / "full_sofa_sepsis3_processing_notes.md"
    notes.write_text(
        f"""# Full SOFA and Sepsis-3 Processing Notes

Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}

Scope:
- Primary derivation: MIMIC-IV v3.1 adult first ICU stays.
- Windows: W1 = 0-24h, W2 = 24-48h, W3 = 48-72h from ICU admission.

Files:
- `07_full_sofa_sepsis3/mimiciv_full_sofa_scores_0_72h.csv`: six SOFA components and total score by time window.
- `07_full_sofa_sepsis3/mimiciv_suspected_infection.csv`: antibiotic+culture suspected infection episodes.
- `07_full_sofa_sepsis3/mimiciv_sepsis3_icu_admit_cohort.csv`: suspected infection near ICU admission plus SOFA >= 2 in W1.
- `07_full_sofa_sepsis3/mimiciv_sepsis3_0_72h_cohort.csv`: suspected infection and SOFA >= 2 using 0-72h maximum.
- `07_full_sofa_sepsis3/mimiciv_sic_scores_full_sofa.csv`: SIC score using platelet, INR, and four-system SOFA.
- `07_full_sofa_sepsis3/mimiciv_sepsis3_icu_admit_sic_trajectory_features.csv`: final Sepsis-3 cohort trajectory table.

Definitions:
- SOFA total is exported as `sofa_total_assuming_missing_normal` and `sofa_total_complete_only`.
- Sepsis-3 flags use `sofa_total_assuming_missing_normal >= 2`, consistent with common retrospective ICU extraction practice when pre-ICU baseline SOFA is unavailable.
- SIC organ component uses respiratory, hepatic, cardiovascular, and renal SOFA only; CNS and coagulation SOFA are still retained in the full SOFA table.
- SIC score is platelet component + INR component + four-system SOFA component. `sic_flag` is total SIC score >= 4.

Limitations:
- This is a reproducible EHR-derived research phenotype, not bedside diagnosis.
- Respiratory SOFA uses window-level minimum PaO2 and maximum FiO2 rather than nearest-time PaO2/FiO2 pairing.
- Vasopressor scoring uses rates recorded as `mcg/kg/min`; phenylephrine and vasopressin are retained as features but not used in the original SOFA cardiovascular score.
- Sepsis-3 suspected infection is based on systemic antibiotic orders paired with microbiology cultures, with culture time between antibiotic time minus 72h and plus 24h.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive full MIMIC-IV SOFA, Sepsis-3, and SIC scores.")
    parser.add_argument(
        "--skip-heavy-extract",
        action="store_true",
        help="Reuse existing heavy lab/chart SOFA subset tables and rebuild downstream tables.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing aris_sic full SOFA/Sepsis-3 tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        setup_connection(conn)
        if not args.export_only:
            build_tables(conn, skip_heavy=args.skip_heavy_extract)
        export_outputs(conn)
    write_notes()
    print(f"[{datetime.now():%H:%M:%S}] done", flush=True)


if __name__ == "__main__":
    main()
