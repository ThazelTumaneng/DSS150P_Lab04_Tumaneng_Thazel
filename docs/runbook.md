# Operations Runbook: End-to-End Stroke Risk Pipeline

**System:** Medallion Data Engineering & Predictive ML Pipeline  
**Orchestration Support:** Python CLI, Pytest, Docker Compose, Apache Airflow 3  

---

## 1. System Requirements & Prerequisites

### Native Host Requirements (macOS)
- **Python:** Version 3.11 or 3.12
- **Homebrew Package Manager**
- **C++ OpenMP Multithreading Runtime:** XGBoost requires OpenMP:
  ```bash
  brew install libomp
  ```

### 2. Local Environment Setup

##### Clone Repository 
```
git clone <REPOSITORY_URL>
```
##### Configure Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
###### Environment Variable Configuration
```
PIPELINE_ENV=development
LOG_LEVEL=INFO
CONFIG_PATH=config/settings.yml
RANDOM_STATE=42
DISABLE_PANDERA_IMPORT_WARNING=True
AIRFLOW_HOME=/app/airflow
```
##### Pipeline Execution Modes
Execute all 5 stages:
```
python -m src.cli --all
```
Step-by-step execution:
```
python -m src.cli --step ingest      # Raw CSV -> Parquet with SHA-256 lineage
python -m src.cli --step validate    # Pandera data contract validation
python -m src.cli --step transform   # Medallion Staging and Curated feature store
python -m src.cli --step train       # Stratified split & weighted XGBoost training
python -m src.cli --step evaluate    # Multi-threshold ASCVD simulation & metrics
```

##### Automated Testing Suite 
Verify pipeline integrity, data contracts, and output artifacts:
```
pytest -v
```

##### Containerized Pipeline Execution
```
# 1. Build the container image
docker compose build

# 2. Execute the containerized pipeline
docker compose up

# 3. Stop and clean up containers
docker compose down
```

##### Apache Airflow Orchestration
Start Airflow Standalone
```
docker compose up -d
```

##### Retrieve Generated Admin Credentials
Airflow 3 generates user credential sinside the container's auth manager file:
```
docker exec -it dss150p_stroke_airflow cat /app/airflow/simple_auth_manager_passwords.json.generated
```
Access Airflow Web Interface
 ```
  # Open browser and navigate to:
  http://localhost:8081

  # Log in using username and the password retrieved from the previous step

  # Locate the DAG and toggle the DAG switch to make it active

  # Click Trigger DAG on the right side

  # Monitor the task execution if it was a success
```

### 3. Disaster Recovery & Rollback Procedures

1. **Source Corrupted / Checksum Mismatch**:
   - Inspect `data/quarantine/` logs.
   - Re-fetch source from Hugging Face:
     `curl -L -o data/raw/Stroke_Prediction_Dataset.csv "https://huggingface.co/datasets/Nnaodeh/Stroke_Prediction_Dataset/resolve/main/Stroke_Prediction_Dataset.csv"`
   - Run `python -m src.cli --step ingest --force`.

2. **Model Regression / Failed Contract Gate**:
   - If `src.evaluate` fails the tolerance gate, inspect `metadata/expected_metrics.json`.
   - Revert `outputs/model/xgb_model.joblib` to the last known checkpoint in git.