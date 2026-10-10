"""
Centralized path configuration for SIC Trajectory Study reproduction.

All paths are relative to this file's location (02_可复现代码/).
The three deliverable folders (01_论文文档, 02_可复现代码, 03_数据源)
must be placed in the same parent directory.

Usage (at the top of each script):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import DATA_DIR, MIMIC_DIR, EICU_DIR, SICDB_DIR, OUTPUT_DIR, FIGURE_DIR
"""
import os
from pathlib import Path

# ── Core directories ──────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent          # 02_可复现代码/
DELIVERABLE_ROOT = CODE_DIR.parent                   # Parent of 01_, 02_, 03_

# ── Data source directories ───────────────────────────────────────────
DATA_ROOT = DELIVERABLE_ROOT / "03_数据源"

MIMIC_DIR   = DATA_ROOT / "MIMIC-IV"    # MIMIC-IV processed data
EICU_DIR    = DATA_ROOT / "eICU"        # eICU processed data
SICDB_DIR   = DATA_ROOT / "SICdb"       # SICdb processed data
META_DIR    = DATA_ROOT / "metadata"    # Lab mappings, processing notes

# ── Analysis output ───────────────────────────────────────────────────
OUTPUT_DIR  = CODE_DIR / "output"                     # All generated results
FIGURE_DIR  = OUTPUT_DIR / "figures"                  # Generated figures
TABLE_DIR   = OUTPUT_DIR / "tables"                   # Generated tables
METRICS_DIR = OUTPUT_DIR / "metrics"                  # JSON metrics

# ── Legacy compatibility aliases ──────────────────────────────────────
# Scripts originally used a flat data_processed/ structure.
# These aliases map old subdirectory names to actual locations.

# MIMIC-IV subdirectories (01_cohort through 07_full_sofa_sepsis3)
MIMIC_COHORT     = MIMIC_DIR / "01_cohort"
MIMIC_WINDOWS    = MIMIC_DIR / "02_window_features"
MIMIC_SIC        = MIMIC_DIR / "03_sic_scores"
MIMIC_TRAJ       = MIMIC_DIR / "04_trajectory_features"
MIMIC_OUTCOMES   = MIMIC_DIR / "05_outcomes"
MIMIC_FULL_SOFA  = MIMIC_DIR / "07_full_sofa_sepsis3"

# eICU subdirectories
EICU_FULL_SOFA   = EICU_DIR / "08_full_sofa_sepsis3_eicu"

# SICdb subdirectories
SICDB_FULL_SOFA  = SICDB_DIR / "09_full_sofa_sicdb"

# ── Raw database paths (for optional re-derivation from source) ───────
# Only needed if running scripts 01-04 to re-derive from raw databases.
# Set via environment variable RAW_DB_DIR or edit below.
RAW_DB_DIR = Path(os.environ.get(
    "RAW_DB_DIR",
    str(DELIVERABLE_ROOT / "raw_databases")
))

MIMIC_RAW  = RAW_DB_DIR / "mimic-iv-3.1"
EICU_RAW   = RAW_DB_DIR / "eicu-collaborative-research-database-2.0"
SICDB_RAW  = RAW_DB_DIR / "sicDB-1.0.8"

# PostgreSQL connection (for MIMIC-IV raw derivation via psycopg)
PG_HOST     = os.environ.get("PG_HOST", "localhost")
PG_PORT     = os.environ.get("PG_PORT", "5432")
PG_USER     = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_DB_MIMIC = os.environ.get("PG_DB_MIMIC", "mimiciv31")

# ── Helper ────────────────────────────────────────────────────────────

def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def verify_data_dirs():
    """Check that required data directories exist. Returns list of missing."""
    missing = []
    for name, path in [
        ("MIMIC-IV", MIMIC_DIR),
        ("eICU", EICU_DIR),
        ("SICdb", SICDB_DIR),
    ]:
        if not path.exists():
            missing.append(f"{name}: {path}")
    return missing
