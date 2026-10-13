from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from src.config import ROOT_DIR, load_config
from src.storage import load_dataframe, save_dataframe, ensure_dir

# Schema definition based on metadata/data_contract.yml
raw_stroke_schema = DataFrameSchema(
    columns={
        "id": Column(int, unique=True, nullable=False),
        "gender": Column(
            str, 
            Check.isin(["Male", "Female", "Other"]), 
            nullable=False
        ),
        "age": Column(float, Check.in_range(0.0, 120.0), nullable=False),
        "hypertension": Column(int, Check.isin([0, 1]), nullable=False),
        "heart_disease": Column(int, Check.isin([0, 1]), nullable=False),
        "ever_married": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "work_type": Column(str, nullable=False),
        "Residence_type": Column(
            str, 
            Check.isin(["Urban", "Rural"]), 
            nullable=False
        ),
        "avg_glucose_level": Column(
            float, 
            Check.greater_than(0.0), 
            nullable=False
        ),
        "bmi": Column(float, Check.greater_than(5.0), nullable=True),
        "smoking_status": Column(
            str,
            Check.isin(["formerly smoked", "never smoked", "smokes", "Unknown"]),
            nullable=False
        ),
        "stroke": Column(int, Check.isin([0, 1]), nullable=False),
    },
    coerce=True,
    strict=False,  # Allows internal audit columns like _ingested_at
)


def validate_data() -> bool:
    """Validates the ingested source parquet file against schema constraints."""
    cfg = load_config()
    source_dir = ROOT_DIR / cfg["paths"]["source_dir"]
    target_source = source_dir / "stroke_source.parquet"
    quarantine_dir = ROOT_DIR / cfg["paths"]["quarantine_dir"]

    if not target_source.exists():
        raise FileNotFoundError(
            f"Source file not found at: {target_source}. Run 'python -m src.ingest' first."
        )

    print(f"🔍 Validating data: {target_source}")
    df = load_dataframe(target_source)

    try:
        raw_stroke_schema.validate(df, lazy=True)
        print(f"✅ Data contract validation passed ({len(df)} records verified).")
        return True

    except pa.errors.SchemaErrors as err:
        print(f"❌ Validation failed with {len(err.failure_cases)} error(s).")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        quarantine_file = quarantine_dir / f"quarantine_failures_{timestamp}.csv"
        
        save_dataframe(err.failure_cases, quarantine_file, file_format="csv")
        print(f"⚠️ Quarantined error report saved to: {quarantine_file}")
        return False


if __name__ == "__main__":
    success = validate_data()
    if not success:
        exit(1)