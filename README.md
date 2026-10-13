# Production End-to-End Stroke Risk Prediction Pipeline

A production-grade, reproducible data engineering and predictive machine learning pipeline implementing a Medallion Architecture (Source $\rightarrow$ Staging $\rightarrow$ Curated) to evaluate 10-year stroke risks under clinical cardiovascular guidelines.

---

## 1. Project Overview & Clinical Problem Formulation

Cerebrovascular accidents (strokes) are among the leading global causes of mortality and long-term disability. In standard health screening populations, stroke occurrences exhibit severe class imbalance (~20:1 healthy-to-stroke ratio). Standard off-the-shelf machine learning classifiers optimized for accuracy collapse under this distribution by applying an arbitrary decision boundary ($\tau = 0.50$), failing to detect more than half of imminent stroke cases.

This repository implements an end-to-end engineered workflow that:
1. Enforces data lineage with cryptographic SHA-256 source hashing and Pandera schema validation gates.
2. Ingests and transforms clinical data through an immutable Medallion Architecture.
3. Implements domain-specific cardiovascular feature engineering (Framingham Vascular Age Acceleration, Triglyceride-Glucose Index, Asian BMI stratification).
4. Trains an imbalanced, cost-sensitive XGBoost classifier using dynamic positive class weighting (`scale_pos_weight = 12.28`).
5. Evaluates model predictions against American Heart Association / American College of Cardiology (AHA/ACC) ASCVD clinical risk tiers, prioritizing **clinical recall/sensitivity** over raw accuracy.
6. Containerizes execution via Docker and orchestrates tasks using Apache Airflow 3.

---

## 2. Medallion Pipeline Architecture

The pipeline processes data through sequential, decoupled stages to ensure auditability, idempotency, and isolation:

![Data Pipeline Process](proofs%20and%20images/data%20pipeline%20process.png)

## 3. Engineering & Transformation Specifications

### Ingestion & Data Integrity
* **Data Source:** Raw clinical data originates from the [Stroke Prediction Dataset on Hugging Face](https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset).
* **Cryptographic Checksum:** The raw dataset (`data/raw/Stroke_Prediction_Dataset.csv`) is verified with SHA-256 (`144ea5366832bb5432645c0e49fbb951aadf4ee1e75e73d2e15d3dc0841525bd`) before any operation begins. Ingestion logs are tracked in `metadata/source_manifest.yml`.
* **Data Contracts:** Schema validation is executed via `pandera.pandas`. Out-of-bounds metrics (e.g., negative ages, impossible glucose values) or unexpected categories are flagged immediately and isolated to `data/quarantine/`.

### Staging Layer Cleaning Rules
* **Cohort Age Scoping:** Restricted to adults aged $35 \le \text{age} \le 90$. Pruned $1,845$ pediatric/young-adult rows where historical stroke incidence is $<0.1\%$, preventing gradient dilution.
* **Gender Imputation:** Replaced unassigned `Other` records with the modal category (`Female`).
* **Pediatric Correction:** Corrected records where `work_type == 'children'` to `never smoked`.
* **Cohort Median BMI Imputation:** Missing BMI values ($N = 201$) were imputed using the target cohort median ($29.80\text{ kg/m}^2$).

### Curated Layer Clinical Feature Engineering
* **Asian BMI Classification:** Flags elevated cardiovascular and metabolic risks starting at lower body mass index boundaries ($\text{BMI} \ge 23.0\text{ kg/m}^2$).
* **Glycemic Risk Tiers:** Stratifies `avg_glucose_level` into Normal ($<100\text{ mg/dL}$), Pre-diabetic ($100\text{--}125\text{ mg/dL}$), and Diabetic ($\ge 126\text{ mg/dL}$).
* **Framingham Vascular Age Acceleration Factor (`VAAF`):**
  $$\text{VAAF} = \frac{\text{Vascular Age}}{\text{Chronological Age}}$$
  Derives biological vascular wear relative to chronological age using blood pressure, glycemic index, and BMI indicators.
* **Triglyceride-Glucose Index (`TGI`):**
  $$\text{TGI} = \ln\left(\frac{\text{avg glucose level} \times \text{bmi}}{2}\right)$$
  Serves as a surrogate index for insulin resistance and metabolic dysfunction.


## 4. Modeling & Clinical Evaluation Benchmark

### Class Imbalance Adjustment
Across the 2,285 training samples, 2,113 belong to non-stroke cases and 172 to stroke cases. The XGBoost objective function uses positive instance weighting (`scale_pos_weight = 12.28`):
$$\text{scale\_weight} = \frac{2113}{172} \approx 12.28$$

### Test Set Threshold Simulation Results ($N = 980$, $74$ Stroke Cases)

| Decision Threshold ($\tau$) | Clinical Stratum / Category | Accuracy | Sensitivity (Recall) | Precision | Missed Strokes (FN) | False Alarms (FP) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0.05** | ASCVD Borderline Cutoff | 16.63% | **94.59%** | 8.44% | **4** | 813 |
| **0.07** | ASCVD Intermediate Boundary | 20.82% | 93.24% | 8.81% | 5 | 771 |
| **0.20** | **ASCVD High-Risk Cutoff (Selected)** | **42.96%** | **83.78%** | **10.65%** | **12** | **547** |
| **0.35** | Operational Screening Midpoint | 64.08% | 64.86% | 12.87% | 26 | 326 |
| **0.40** | Transition Threshold | 68.98% | 58.11% | 13.27% | 31 | 273 |
| **0.45** | Transition Threshold | 72.14% | 50.00% | 12.76% | 37 | 236 |
| **0.50** | Default ML Standard Cutoff | 74.59% | **44.59%** | 12.50% | **41** | 208 |

### Decision Analysis
* **Why the 0.50 Default Threshold Fails:** At $\tau = 0.50$, the model produces an apparent $74.59\%$ accuracy, but misses **41 out of 74 stroke patients (55.4% false-negative rate)**. In clinical screening, missing over half of high-risk patients defeats the purpose of preventive intervention.
* **Selected Cutoff ($\tau = 0.20$):** Aligned with the ACC/AHA 10-year high-risk threshold, this gate captures **83.78%** of true stroke cases (limiting false negatives to 12) while filtering out hundreds of healthy patients from unnecessary diagnostic procedures.


## 5. Repository Structure

```text
├── config/
│   └── settings.yml                     # Pipeline runtime configurations & hyperparams
├── data/
│   ├── raw/                             # Raw source CSV dataset
│   ├── source/                          # SHA-256 validated Parquet table
│   ├── staging/                         # Cleaned cohort dataset (ages 35-90)
│   ├── curated/                         # Feature engineered Parquet table
│   └── partitioned/                     # Stratified train.parquet and test.parquet
├── dags/
│   └── predictive_repro_pipeline.py     # Airflow 3 orchestration DAG
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
│   └── test_pipeline.py                 # Automated pytest test suite
├── .env.example                         # Environment configuration template
├── .gitignore                           # Git exclusion rules
├── Dockerfile                           # Container definition (with libgomp1 OpenMP)
├── docker-compose.yml                   # Service composition (Airflow mapped to port 8081)
├── pytest.ini                           # Test configuration & pythonpath settings
├── requirements.txt                     # Pinned project Python dependencies
└── README.md
```

## 6. Quickstart Guide

Prerequisites
- Python 3.11 or 3.12
- macOS users: OpenMP runtime installed via Homebrew (`brew install libomp`)
- Docker Desktop (for containerized execution)

Local Environment Setup and Data Acquisition
```
# Clone the repository
git clone <REPO_URL>

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Dataset Acquisition
Download `Stroke_Prediction_Dataset.csv` directly from hugging face:
    `https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset?fbclid=IwY2xjawU6sbFleHRuA2FlbQIxMQBwZG9mA3NydGMGYXBwX2lkATAAAR593nd7zWNJODMdRBzaUn2nk1Azu8vldi3-Gi2dtDKHDY8p5I8cUyAnFap0ow_aem_ZIlA9wRJEuwlSzGZXT_ZoQ`

# Ensure the downloaded csv file is placed in the raw data directory
mkdir -p data/raw
# Place the file at: data/raw/Stroke_Prediction_Dataset.csv

# Run the complete pipeline (Ingest-Validate-Transform-Train-Evaluate)
python -m src.cli --all
or run individual stages
python -m src.cli --step ingest
python -m src.cli --step validate
python -m src.cli --step transform
python -m src.cli --step train
python -m src.cli --step evaluate

#Run automated tests
pytest -v

#Run via docker compose
docker compose build
docker compose up

#Run with Apache Airflow Web UI
docker compose up -d
docker exec -it dss150p_stroke_airflow cat /app/airflow/simple_auth_manager_passwords.json.generated
Open 'http://localhost:8081' with your browser
Login with your credetials and click the trigger DAG

```

## 7. Operational Documents
- For deployment commands and configuration, refer to `docs/runbook.md`
- For detailed sensitivity analysis and ASCVD clinical comparison, see `docs/benchmark_report.md`
- For data provenance and cross-environment determinism audits, see `docs/reproducibility_report.md`