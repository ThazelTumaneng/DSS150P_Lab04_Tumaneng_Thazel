"""
src/ingest.py
Handles cryptographic lineage tracking, SHA-256 validation, and
both full and 3-batch incremental ingestion for the stroke dataset.
"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
import click
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data/raw/Stroke_Prediction_Dataset.csv"
SOURCE_PARQUET = ROOT_DIR / "data/source/stroke_source.parquet"
CHECKPOINT_PATH = ROOT_DIR / "metadata/ingestion_checkpoint.json"


def calculate_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of input raw file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_row_hashes(df: pd.DataFrame) -> pd.Series:
    """Computes deterministic SHA-256 hash per row for deduplication."""
    return df.astype(str).sum(axis=1).apply(lambda row: hashlib.sha256(row.encode("utf-8")).hexdigest())


def run_ingestion(mode: str = "full", batch_num: int = 1, force: bool = False):
    """Executes full or 3-batch incremental ingestion."""
    pipeline_run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    logger.info(f"Initiating ingestion | Mode: {mode} | Run ID: {pipeline_run_id}")
    
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw source file not found at {RAW_PATH}")
        
    source_hash = calculate_file_sha256(RAW_PATH)
    logger.info(f"Verified Source SHA-256: {source_hash}")

    df_raw = pd.read_csv(RAW_PATH)
    df_raw["row_hash"] = compute_row_hashes(df_raw)
    df_raw["pipeline_run_id"] = pipeline_run_id
    df_raw["ingested_at"] = datetime.utcnow().isoformat()

    if mode == "full" or force:
        logger.info("Executing FULL ingestion. Overwriting source store...")
        SOURCE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(SOURCE_PARQUET, index=False)
        
        checkpoint = {
            "last_mode": "full",
            "last_run_id": pipeline_run_id,
            "total_rows": len(df_raw),
            "watermark_id": int(df_raw["id"].max()),
            "last_batch": 3,
            "timestamp": datetime.utcnow().isoformat()
        }
    elif mode == "incremental":
        logger.info(f"Executing INCREMENTAL batch {batch_num} of 3...")
        chunks = [df_raw.iloc[i::3].copy() for i in range(3)]
        target_chunk = chunks[batch_num - 1]
        
        if SOURCE_PARQUET.exists() and not force:
            existing_df = pd.read_parquet(SOURCE_PARQUET)
            new_rows = target_chunk[~target_chunk["row_hash"].isin(existing_df["row_hash"])]
            combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
        else:
            combined_df = target_chunk

        combined_df.to_parquet(SOURCE_PARQUET, index=False)
        checkpoint = {
            "last_mode": "incremental",
            "last_batch": batch_num,
            "last_run_id": pipeline_run_id,
            "total_rows": len(combined_df),
            "watermark_id": int(combined_df["id"].max()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Ingestion complete. Current Source Rows: {checkpoint['total_rows']}")


def ingest_data(mode: str = "full", batch_num: int = 1, force: bool = False):
    """Callable entry point required by src/cli.py and Airflow DAG."""
    return run_ingestion(mode=mode, batch_num=batch_num, force=force)


@click.command()
@click.option("--mode", type=click.Choice(["full", "incremental"]), default="full")
@click.option("--batch", type=int, default=1)
@click.option("--force", is_flag=True, default=False)
def cli_ingest(mode, batch, force):
    run_ingestion(mode=mode, batch_num=batch, force=force)


if __name__ == "__main__":
    cli_ingest()