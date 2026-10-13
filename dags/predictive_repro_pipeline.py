from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append("/app")

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.ingest import ingest_data
from src.validate import validate_data
from src.transform import run_transformations
from src.train import train_stroke_model
from src.evaluate import evaluate_model


def task_ingest():
    ingest_data()


def task_validate():
    passed = validate_data()
    if not passed:
        raise ValueError("Data contract validation failed!")


def task_transform():
    run_transformations()


def task_train():
    train_stroke_model()


def task_evaluate():
    evaluate_model()


default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="stroke_predictive_repro_pipeline",
    default_args=default_args,
    description="End-to-end reproducible stroke prediction pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["stroke_prediction", "dss150p"],
) as dag:

    t1 = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=task_ingest,
    )

    t2 = PythonOperator(
        task_id="validate_data_contract",
        python_callable=task_validate,
    )

    t3 = PythonOperator(
        task_id="transform_staging_and_curated",
        python_callable=task_transform,
    )

    t4 = PythonOperator(
        task_id="train_xgboost_model",
        python_callable=task_train,
    )

    t5 = PythonOperator(
        task_id="evaluate_model_performance",
        python_callable=task_evaluate,
    )

    t1 >> t2 >> t3 >> t4 >> t5
