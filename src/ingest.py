from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.config import ROOT_DIR, load_config


def ingest_data() -> Path:
    cfg = load_config()

    # Input: CSV located in data/raw/
    raw_path = (
        ROOT_DIR / cfg["paths"]["raw_dir"] / cfg["dataset"]["raw_filename"]
    )

    # Output: Ingested file written to data/source/
    source_dir = ROOT_DIR / cfg["paths"]["source_dir"]
    source_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected raw dataset at: {raw_path}\n"
            f"Please verify that {cfg['dataset']['raw_filename']} is inside {cfg['paths']['raw_dir']}/"
        )

    print(f"- Ingesting from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Attach lineage metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    df["_ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["_source_file"] = raw_path.name
    df["_batch_id"] = timestamp

    # Save to data/source/ as Parquet
    ingested_filepath = source_dir / f"stroke_source_{timestamp}.parquet"
    df.to_parquet(ingested_filepath, index=False)

    print(
        f"- Ingested {len(df)} records from {raw_path.name} -> {ingested_filepath}"
    )
    return ingested_filepath


if __name__ == "__main__":
    ingest_data()