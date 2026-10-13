# src/ingest.py
from datetime import datetime, timezone
from pathlib import Path
import yaml
import pandas as pd
from src.config import ROOT_DIR, load_config
from src.storage import (
    compute_file_hash,
    ensure_dir,
    load_dataframe,
    save_dataframe,
)


def update_source_manifest(
    manifest_path: Path,
    raw_file: Path,
    source_file: Path,
    raw_hash: str,
    df: pd.DataFrame,
) -> None:
    """Writes or updates metadata/source_manifest.yml for end-to-end reproducibility."""
    manifest_data = {
        "dataset_name": "Stroke_Prediction_Dataset",
        "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_source_path": str(raw_file.relative_to(ROOT_DIR)),
        "raw_source_sha256": raw_hash,
        "persisted_source_path": str(source_file.relative_to(ROOT_DIR)),
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

    print(f"📋 Lineage manifest recorded: {manifest_path}")


def ingest_data() -> Path:
    """Ingests raw data into the source directory with verifiable integrity and lineage."""
    cfg = load_config()

    raw_path = ROOT_DIR / cfg["paths"]["raw_dir"] / cfg["dataset"]["raw_filename"]
    source_dir = ROOT_DIR / cfg["paths"]["source_dir"]
    manifest_file = ROOT_DIR / "metadata" / "source_manifest.yml"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {raw_path}\n"
            f"Please verify '{cfg['dataset']['raw_filename']}' exists in '{cfg['paths']['raw_dir']}/'."
        )

    print(f"📥 Loading raw data from: {raw_path}")
    raw_hash = compute_file_hash(raw_path)
    print(f"🔒 Raw SHA-256 Checksum: {raw_hash}")

    df = load_dataframe(raw_path, file_format="csv")

    # Attach immutable audit metadata
    ingestion_ts = datetime.now(timezone.utc).isoformat()
    df["_ingested_at"] = ingestion_ts
    df["_source_file"] = raw_path.name
    df["_source_sha256"] = raw_hash

    # Save canonical dataset in data/source/
    canonical_filename = "stroke_source.parquet"
    target_filepath = source_dir / canonical_filename
    save_dataframe(df, target_filepath, file_format="parquet")

    # Update metadata/source_manifest.yml
    update_source_manifest(manifest_file, raw_path, target_filepath, raw_hash, df)

    print(f"✅ Ingestion complete: {len(df)} rows saved to {target_filepath}")
    return target_filepath


if __name__ == "__main__":
    ingest_data()