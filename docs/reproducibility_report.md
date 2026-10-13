# Reproducibility & Data Lineage Report

**Project:** Stroke Risk Predictive Pipeline (DSS150P)  
**Execution Environments Verified:** macOS (Darwin ARM64 / Apple Silicon) & Docker Linux Container (`python:3.11-slim`)  
**Determinism Engine:** Fixed Pseudo-Random Seeds (`random_state = 42`)  
**Data Contract Framework:** Pandera Schema Enforcement  
**Orchestration Engine:** Apache Airflow (`stroke_predictive_repro_pipeline`)  

---

## 1. Cryptographic Data Lineage & Integrity Checks

To guarantee complete auditability and prevent silent data corruption, input data is treated as immutable with cryptographic checksums computed across each storage boundary.

### Raw Source Ingestion Gate
- **Raw File Path:** `data/raw/Stroke_Prediction_Dataset.csv`
- **Algorithm:** Cryptographic Hash Algorithm SHA-256
- **Verified SHA-256 Digest:**
  ```text
  144ea5366832bb5432645c0e49fbb951aadf4ee1e75e73d2e15d3dc0841525bd

## 2. Medallion Layer Lineage & Transformation Audit
```bash
[Raw Source CSV] (5,110 rows)
       │
       ▼ (SHA-256 & Schema Validation)
[Source Layer: stroke_source.parquet] (5,110 rows, audit columns)
       │
       ▼ (Cohort Filter 35-90, Median BMI Imputation)
[Staging Layer: stroke_staging.parquet] (3,265 rows, 154 BMIs imputed)
       │
       ▼ (Clinical Feature Engineering, Audit Column Stripping)
[Curated Layer: stroke_curated.parquet] (3,265 rows, 21 feature cols)
       │
       ├── Multi-format: stroke_curated.jsonl
       ├── Partitioned Store: stroke_curated_partitioned_age/
       │
       ▼ (Stratified 70/30 Split, seed=42)
[Partitioned Partitions: train.parquet (2,285) & test.parquet (980)]
```

Data Cleaning & Cohort Isolation (Staging)
1. Target Adult Cohort ($35 \le \text{age} \le 90$): Isolates the clinically actionable population where stroke prevalence is non-negligible ($3,265$ records retained from $5,110$ raw rows).
2. Missing Value Imputation: In the target cohort, exactly 154 BMI records were missing. Missing values were imputed using the cohort median of $29.80\text{ kg/m}^2$ without out-of-fold target leakage.
3. Domain Logic Sanity: Records with work type children were mapped to never smoked, and Other gender values were resolved to the cohort mode.

Domain-Specific Feature Store (Curated)
1. ADA/WHO Glycemic Tiers (glucose_score_risk): Normal ($<100\text{ mg/dL}$), Pre-diabetic ($100\text{--}125.99\text{ mg/dL}$), Diabetic ($\ge 126\text{ mg/dL}$).
2. Asian BMI Risk Tiers (`bmi_score_risk`): Thresholded at $\ge 23.0\text{ kg/m}^2$ to capture elevated metabolic risk in Asian populations.
3. Vascular Age Acceleration Factor (VAAF): Calculated as the true ratio of Framingham Vascular Age to Chronological Age:$$\text{VAAF} = \frac{\text{Vascular Age}}{\text{Chronological Age}}$$
4. Triglyceride-Glucose Index (TGI): Calculated with natural logarithmic scaling:$$\text{TGI} = \ln\left(\frac{\text{avg\_glucose\_level} \times \text{bmi}}{2}\right)$$
5. Leakage Prevention: Audit metadata columns (`row_hash`, `pipeline_run_id`, `ingested_at`) are explicitly purged from the Curated dataset prior to model consumption.

## 3. Deterministic Training & Artifact Serialization
1. Split Allocation: 70% Train ($N=2,285$, 172 stroke events) and 30% Test ($N=980$, 74 stroke events) using stratified sampling with `random_state = 42`.
2. Manifest: Verified and preserved in `metadata/split_manifest.json`.
3. Class Reweighting: scale_pos_weight = 12.28 calculated dynamically as:$$\frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{2113}{172} \approx 12.28$$
4. Model Serialization: Trained estimator serialized via joblib to `outputs/model/xgb_model.joblib`.

## 4. Verification Suite & Acceptance Contract Gate
Automated test execution via `pytest` validates structural integrity across five test boundaries:
```
tests/test_pipeline.py::test_configuration_keys PASSED           [ 20%]
tests/test_pipeline.py::test_source_layer_artifact PASSED        [ 40%]
tests/test_pipeline.py::test_staging_layer_cleaning PASSED       [ 60%]
tests/test_pipeline.py::test_curated_layer_features PASSED       [ 80%]
tests/test_pipeline.py::test_model_and_evaluation_artifacts PASSED [100%]

============================== 5 passed in 0.63s ==============================
```
| Evaluation Contract Metric | Tolerance Threshold | Realized Pipeline Run | Contract Verdict |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | $\ge 0.6500$ | 0.6940 | PASSED |
| **Sensitivity at $\tau = 0.20$** | $\ge 80.00\%$ | 87.84% | PASSED |
| **Maximum False Negatives at $\tau = 0.20$** | $\le 15$ cases | 9 cases | PASSED |


## 5. Architectural Gap Analysis
| Architectural Component | Prior Baseline Implementation | Our Production Implementation | Status |
| :--- | :--- | :--- | :---: |
| **Ingestion** | Full unvalidated CSV load without lineage | Cryptographic SHA-256 + 3-Batch Incremental Watermarking | Upgraded |
| **Data Contracts** | Implicit Pandas type casts | Pandera Schema Enforcement + Quarantine Isolation | Upgraded |
| **Feature Store** | Standard uncalibrated features | Domain-specific ASCVD tiers, VAAF ratio, $\ln$-scaled TGI | Upgraded |
| **Model Thresholding** | Unweighted default $\tau=0.50$ (55.4% missed strokes) | Scaled XGBoost (`scale_pos_weight=12.28`) with ASCVD $\tau=0.20$ | Upgraded |
| **Data Storage** | Monolithic local CSV overwrite | Medallion Architecture (Source $\rightarrow$ Staging $\rightarrow$ Curated) + Parquet/JSONL | Upgraded |
| **Orchestration** | Ad-hoc linear script execution | Apache Airflow containerized DAG (`stroke_predictive_repro_pipeline`) + CLI | Upgraded |
| **Quality Assurance** | Manual visual inspection | Automated 5-point Pytest suite + Contract Gating Enforcement | Upgraded |

## 6. Execution Modalities
1. Local CLI Execution
```
python -m src.cli --all
```
2. Containerized Orchestration Execution
```
docker compose up --build -d
docker compose exec airflow airflow dags trigger stroke_predictive_repro_pipeline
```
3. Airflow UI accessible at http://localhost:8081 (DAG status: active and operational).