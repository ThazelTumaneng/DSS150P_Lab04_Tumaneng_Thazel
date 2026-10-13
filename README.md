# Production End-to-End Stroke Risk Prediction Pipeline

A production-grade, reproducible data engineering and predictive machine learning pipeline implementing a Medallion Architecture (Source $\rightarrow$ Staging $\rightarrow$ Curated) to evaluate 10-year stroke risks under clinical cardiovascular guidelines.

---

## 1. Project Overview & Clinical Problem Formulation

Cerebrovascular accidents (strokes) are among the leading global causes of mortality and long-term disability. In standard health screening populations, stroke occurrences exhibit severe class imbalance (~20:1 healthy-to-stroke ratio). Standard off-the-shelf machine learning classifiers optimized for accuracy collapse under this distribution by applying an arbitrary decision boundary ($\tau = 0.50$), failing to detect more than half of imminent stroke cases.

This repository implements an end-to-end engineered workflow that:
1. Enforces data lineage with cryptographic SHA-256 source hashing and Pandera schema validation gates.
2. Ingests and transforms clinical data through an immutable Medallion Architecture.
3. Implements domain-specific cardiovascular feature engineering (Framingham Vascular Age Acceleration, Triglyceride-Glucose Index, Asian BMI stratification).
4. Enforces data hygiene by purging ingestion audit metadata to prevent model feature leakage.
5. Trains an imbalanced, cost-sensitive XGBoost classifier using dynamic positive class weighting (`scale_pos_weight = 12.28`).
6. Evaluates model predictions against American Heart Association / American College of Cardiology (AHA/ACC) ASCVD clinical risk tiers, prioritizing **clinical recall/sensitivity** over raw accuracy.
7. Containerizes execution via Docker and orchestrates tasks using Apache Airflow.

---

## 2. Medallion Pipeline Architecture

The pipeline processes data through sequential, decoupled stages to ensure auditability, idempotency, and isolation:

![Data Pipeline Process](proofs%20and%20images/data%20pipeline%20process.png)

```text
[Raw Source CSV] (5,110 rows)
       │
       ▼ (SHA-256 Ingestion Gate & Row Hashing)
[Source Layer: stroke_source.parquet] (5,110 rows, immutable audit columns)
       │
       ▼ (Target Cohort 35-90 Filter & Median BMI Imputation)
[Staging Layer: stroke_staging.parquet] (3,265 rows, 154 BMIs imputed)
       │
       ▼ (Clinical Feature Engineering & Audit Column Purging)
[Curated Layer: stroke_curated.parquet] (3,265 rows, 21 clinical feature cols)
       │
       ├── Multi-format: stroke_curated.jsonl
       ├── Partitioned Store: stroke_curated_partitioned_age/
       │
       ▼ (Stratified 70/30 Split, random_state=42)
[Partitioned Partitions: train.parquet (2,285) & test.parquet (980)]
```

## 3. Engineering & Transformation Specifications
Ingestion & Data Integrity
1. Data Source: Raw clinical data originates from the Stroke Prediction Dataset on Hugging Face.
2. Cryptographic Checksum: The raw dataset (data/raw/Stroke_Prediction_Dataset.csv) is verified with SHA-256 (144ea5366832bb5432645c0e49fbb951aadf4ee1e75e73d2e15d3dc0841525bd) before any operation begins. Ingestion runs are logged in metadata/source_manifest.yml.
3. Data Contracts: Schema validation is executed via pandera. Out-of-bounds metrics (e.g., negative ages, impossible glucose values) or unexpected categories are flagged immediately and isolated to data/quarantine/.

Staging Layer Cleaning Rules
1. Cohort Age Scoping: Restricted to adults aged $35 \le \text{age} \le 90$. Pruned $1,845$ pediatric/young-adult rows where historical stroke incidence is $<0.1\%$, preventing gradient dilution.
2. Gender Imputation: Replaced unassigned Other records with the modal category (Female).
3. Pediatric Correction: Corrected records where work_type == 'children' to never smoked.
4. Cohort Median BMI Imputation: Within the target adult cohort, exactly 154 missing BMI values (out of 201 across the raw population) were imputed using the target cohort median ($29.80\text{ kg/m}^2$).

Curated Layer Clinical Feature Engineering
1. Asian BMI Classification: Flags elevated cardiovascular and metabolic risks starting at lower body mass index boundaries ($\text{BMI} \ge 23.0\text{ kg/m}^2$).
2. Glycemic Risk Tiers: Stratifies avg_glucose_level into Normal ($<100\text{ mg/dL}$), Pre-diabetic ($100\text{--}125.99\text{ mg/dL}$), and Diabetic ($\ge 126\text{ mg/dL}$).
3. Framingham Vascular Age Acceleration Factor (VAAF):$$\text{VAAF} = \frac{\text{Vascular Age}}{\text{Chronological Age}}$$Derives biological vascular wear relative to chronological age using blood pressure, glycemic index, and BMI indicators.
4. Triglyceride-Glucose Index (TGI):$$\text{TGI} = \ln\left(\frac{\text{avg\_glucose\_level} \times \text{bmi}}{2}\right)$$Serves as a surrogate index for insulin resistance and metabolic dysfunction.
5. Data Leakage Prevention: Ingestion metadata columns (row_hash, pipeline_run_id, ingested_at) are explicitly purged before model training.
6. Multi-Format Persistence: Automatically exports Curated data to both Parquet and JSONL, alongside age-stratified partitioned directories.

## 4. Modeling & Clinical Evaluation Benchmark

Class Imbalance Adjustment
Across the 2,285 training samples, 2,113 belong to non-stroke cases and 172 to stroke cases. The XGBoost objective function uses positive instance weighting (scale_pos_weight = 12.28):$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{2113}{172} \approx 12.28$$

## 5. Repository Structure
```bash
├── config/
│   └── settings.yml                     # Pipeline runtime configurations & hyperparams
├── data/
│   ├── raw/                             # Raw source CSV dataset
│   ├── source/                          # SHA-256 validated Parquet table
│   ├── staging/                         # Cleaned cohort dataset (ages 35-90)
│   ├── curated/                         # Curated Parquet & JSONL feature tables
│   │   ├── stroke_curated.parquet
│   │   ├── stroke_curated.jsonl
│   │   └── stroke_curated_partitioned_age/
│   └── partitioned/                     # Stratified train.parquet and test.parquet
├── dags/
│   └── predictive_repro_pipeline.py     # Apache Airflow orchestration DAG
├── docs/
│   ├── benchmark_report.md              # Clinical evaluation & ASCVD simulation report
│   ├── reproducibility_report.md        # Cryptographic lineage & state transition audit
│   └── runbook.md                       # Complete operational runbook
├── metadata/
│   ├── expected_metrics.json            # Model acceptance benchmark contract
│   ├── source_manifest.yml              # Cryptographic lineage hash and source profile
│   └── split_manifest.json              # Train/test partition indices & seed audit
├── outputs/
│   ├── metrics.json                     # Generated threshold evaluation results
│   ├── predictions.csv                  # Test partition inference probabilities
│   └── model/
│       └── xgb_model.joblib             # Serialized trained XGBoost binary
├── sql/
│   ├── schema.sql                       # Schema definitions for pipeline layers
│   └── transform.sql                    # SQL transformation specifications
├── src/
│   ├── __init__.py
│   ├── cli.py                           # Unified pipeline CLI controller
│   ├── config.py                        # Path resolution and YAML loader
│   ├── evaluate.py                      # Threshold simulation & metrics generator
│   ├── ingest.py                        # Checksum verification & Parquet ingestion
│   ├── storage.py                       # Atomic I/O read/write helpers
│   ├── train.py                         # Stratified splitting & weighted XGBoost training
│   ├── transform.py                     # Medallion cleaning & feature engineering
│   └── validate.py                      # Pandera schema data validation contracts
├── tests/
│   └── test_pipeline.py                 # Automated pytest test suite (5 passed)
├── .env.example                         # Environment configuration template
├── .gitignore                           # Git exclusion rules
├── Dockerfile                           # Container definition (with libgomp1 OpenMP)
├── docker-compose.yml                   # Service composition (Airflow mapped to port 8081)
├── pytest.ini                           # Test configuration & pythonpath settings
├── requirements.txt                     # Pinned project Python dependencies
└── README.md
```

## 6. Quickstart Guide
- Prerequisites
- Python 3.11 or 3.12
- macOS users: OpenMP runtime installed via Homebrew (brew install libomp)
- Docker Desktop (for containerized execution)

## 7. Operational Documents
For operational deployments and failure recovery runbooks, refer to `docs/runbook.md`.

For detailed sensitivity simulations and clinical ASCVD comparisons, refer to `docs/benchmark_report.md`.

For cryptographic data provenance and cross-environment determinism audits, refer to `docs/reproducibility_report.md`.