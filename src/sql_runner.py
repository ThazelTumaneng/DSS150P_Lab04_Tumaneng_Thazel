# src/sql_runner.py
import sqlite3
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent

def run_sql_transformations():
    db_path = ROOT_DIR / "data/stroke_database.db"
    sql_path = ROOT_DIR / "sql/transformations.sql"
    staging_path = ROOT_DIR / "data/staging/stroke_staging.parquet"

    df_staging = pd.read_parquet(staging_path)
    conn = sqlite3.connect(db_path)
    df_staging.to_sql("stroke_staging", conn, if_exists="replace", index=False)

    if sql_path.exists():
        with open(sql_path, "r") as f:
            sql_script = f.read()
        conn.executescript(sql_script)
        logger.info("Successfully executed SQL transformations against database.")
    conn.close()

if __name__ == "__main__":
    run_sql_transformations()