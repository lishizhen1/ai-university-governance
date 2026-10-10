> Historical source definitions. This README documents the original pipeline. Its old K4 modelling workflow does not reproduce the revised manuscript. Use ../../REPRODUCTION_README.md and the revised analysis scripts for final results. No processed or raw database files are distributed here.

# SIC 动态轨迹论文 — 可复现代码

## 前置条件

- **Python 3.11+**（推荐 3.11 或 3.12）
- 三个交付文件夹放在同一父目录下：
  ```
  交付根目录/
  ├── 01_论文文档/
  ├── 02_可复现代码/     ← 你在这里
  └── 03_数据源/
  ```

## 一键环境配置

### Windows
```cmd
cd 02_可复现代码
setup_env.bat
```

### Linux / macOS
```bash
cd 02_可复现代码
chmod +x setup_env.sh
./setup_env.sh
```

脚本会自动：创建 `.venv` → 安装依赖 → 验证 → 检查数据目录。

## 目录结构

```
02_可复现代码/
├── config.py                          # 路径配置（核心，所有脚本依赖此文件）
├── requirements.txt                   # Python 依赖清单
├── setup_env.bat                      # Windows 一键配置
├── setup_env.sh                       # Linux/macOS 一键配置
├── 01_数据预处理/                      # 需要原始 MIMIC-IV PostgreSQL
│   └── process_sic_trajectory_data.py
├── 02_SOFA_SIC派生/                    # 需要原始数据库文件
│   ├── derive_full_sofa_sepsis3_mimiciv.py   # 需 PostgreSQL
│   ├── derive_full_sofa_sepsis3_eicu.py      # 需原始 eICU .csv.gz
│   └── derive_full_sofa_sicdb.py             # 需原始 SICdb CSV
├── 03_轨迹聚类与预测模型/              # 只需 03_数据源
│   └── run_sic_trajectory_analysis.py
├── 04_外部验证/                        # 只需 03_数据源 + 步骤5输出
│   └── run_eicu_full_external_validation.py
└── 05_补充分析/                        # 只需 03_数据源 + 步骤5输出
    ├── run_supplementary_analysis.py
    └── run_sensitivity_subgroup_analysis.py
```

## 运行顺序

### 快速复现（推荐，从已处理数据开始）

激活环境后，按顺序运行：

```bash
# 激活虚拟环境
# Windows: .venv\Scripts\activate.bat
# Linux:   source .venv/bin/activate

# 步骤 5：轨迹聚类 + 预测模型（生成 output/full_k4/）
python 03_轨迹聚类与预测模型/run_sic_trajectory_analysis.py --full --force-k 4

# 步骤 6：eICU 外部验证
python 04_外部验证/run_eicu_full_external_validation.py

# 步骤 7：补充分析（DeLong, SHAP, KM, DCA, 校准曲线）
python 05_补充分析/run_supplementary_analysis.py

# 步骤 8：敏感性分析 + 亚组森林图
python 05_补充分析/run_sensitivity_subgroup_analysis.py
```

### 完整复现（从原始数据库重新派生，可选）

步骤 1-4 需要原始数据库文件，且需要额外配置：

```bash
# 设置原始数据库路径（环境变量或修改 config.py）
export RAW_DB_DIR="/path/to/raw_databases"
# 数据库内需包含:
#   mimic-iv-3.1/
#   eicu-collaborative-research-database-2.0/
#   sicDB-1.0.8/

# MIMIC-IV 需 PostgreSQL（步骤 1, 2）
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=postgres
export PG_PASSWORD=your_password
export PG_DB_MIMIC=mimiciv31

# 步骤 1：数据预处理
python 01_数据预处理/process_sic_trajectory_data.py

# 步骤 2-4：SOFA/SIC 派生
python 02_SOFA_SIC派生/derive_full_sofa_sepsis3_mimiciv.py
python 02_SOFA_SIC派生/derive_full_sofa_sepsis3_eicu.py
python 02_SOFA_SIC派生/derive_full_sofa_sicdb.py
```

## 输出位置

所有分析结果输出到 `output/` 目录：

```
output/
├── full_k4/                          # 主分析结果
│   ├── tables/                       # cluster_assignments.csv, 表格
│   ├── metrics/                      # analysis_metrics.json
│   ├── figures/                      # 生成的图表
│   └── external_validation_eicu_full_sofa/
├── full_k4/supplementary/            # 补充分析结果
└── data_processed/                   # (仅从原始数据重跑时生成)
```

## 配置说明

所有路径在 `config.py` 中集中配置。如需修改数据位置或输出位置，编辑此文件即可。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DATA_ROOT` | `../03_数据源` | 处理后数据根目录 |
| `MIMIC_DIR` | `../03_数据源/MIMIC-IV` | MIMIC-IV 数据 |
| `EICU_DIR` | `../03_数据源/eICU` | eICU 数据 |
| `SICDB_DIR` | `../03_数据源/SICdb` | SICdb 数据 |
| `OUTPUT_DIR` | `./output` | 分析输出 |
| `RAW_DB_DIR` | `../raw_databases` 或环境变量 `RAW_DB_DIR` | 原始数据库（仅步骤 1-4） |
| `PG_*` | 环境变量或默认值 | PostgreSQL 连接（仅 MIMIC-IV） |

## 注意事项

- **SICdb 缺失 GCS/CNS 数据**：公开 SICdb 表未提供 Glasgow Coma Scale，完整六组分 SOFA 的 CNS 组分不可估计。脚本中 `sofa_total_assuming_missing_normal` 将缺失 CNS 按 0 分处理。
- **eICU 有完整 GCS/CNS**：`nurseCharting.csv.gz` 和 `apacheApsVar.csv.gz` 提供 GCS 数据，full SOFA 派生已包含 CNS 组分。
- **eICU microLab 稀疏**：严格 Sepsis-3（抗生素-培养配对）在 eICU 中样本量不足，外部验证使用 text-sepsis 队列。
