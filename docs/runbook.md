# Operations Runbook: End-to-End Stroke Risk Pipeline

**System:** Medallion Data Engineering & Predictive ML Pipeline (DSS150P)  
**Execution Modalities:** Native Python CLI, Pytest Test Harness, Docker Compose, Apache Airflow  
**Orchestration DAG:** `stroke_predictive_repro_pipeline`  
**Containerized Endpoint:** `http://localhost:8081`  

---

## 1. System Requirements & Environment Setup

### Native Host Requirements (macOS / Darwin ARM64)
- **Python:** Version 3.11 or 3.12
- **Package Manager:** Homebrew
- **C++ OpenMP Runtime:** Required for XGBoost multi-threaded gradient boosting:
  ```bash
  brew install libomp

  # 1. Create and activate isolated Python environment
  python3 -m venv venv
  source venv/bin/activate

  # 2. Upgrade pip and install pinned project dependencies
  pip install --upgrade pip
  pip install -r requirements.txt

  PIPELINE_ENV=development
  LOG_LEVEL=INFO
  CONFIG_PATH=config/settings.yml
  RANDOM_STATE=42
  DISABLE_PANDERA_IMPORT_WARNING=True
  AIRFLOW_HOME=/app/airflow

  # 3. End-to-End Pipeline Execution
  python -m src.cli --all

  # 4. Step-by-Step Execution
  # Step 1: Ingestion (Source Layer - SHA-256 lineage, row hashing, deduplication)
  python -m src.cli --step ingest

  # Step 2: Data Quality & Contract Gate (Pandera schema & boundary validation)
  python -m src.cli --step validate

  # Step 3: Medallion Transformation (Staging cohort filtering & Curated clinical features)
  python -m src.cli --step transform

  # Step 4: Model Training (Stratified 70/30 split & scale_pos_weight=12.28 XGBoost)
  python -m src.cli --step train

  # Step 5: Model Evaluation & Contract Gate (ASCVD multi-threshold simulation & acceptance check)
  python -m src.cli --step evaluate

  # 5. Automated Verification Suite
  pytest -v

  # 6. Build the container image and launch services in detached mode
  docker compose up --build -d

  # 7. Verify container health and port bindings
  docker compose ps

  # 8. Inspect container startup logs
  docker compose logs airflow | grep -i "password"

  # 9. Verify the DAG is parsed and unpaused
  docker compose exec airflow airflow dags list

  # 10. Trigger an execution run
  docker compose exec airflow airflow dags trigger stroke_predictive_repro_pipeline

  # 11. Or run an integrated end-to-end DAG test run
  docker compose exec airflow airflow dags test stroke_predictive_repro_pipeline 2026-10-13