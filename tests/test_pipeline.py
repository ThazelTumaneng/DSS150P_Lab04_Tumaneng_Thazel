# tests/test_pipeline.py
import pytest
from pathlib import Path
from src.config import ROOT_DIR, load_config
from src.storage import load_dataframe


def test_configuration_keys():
    """Verify config/settings.yml contains all required sections."""
    cfg = load_config()
    assert "dataset" in cfg, "Missing 'dataset' section in config"
    assert "paths" in cfg, "Missing 'paths' section in config"
    assert "model" in cfg, "Missing 'model' section in config"


def test_source_layer_artifact():
    """Verify source layer parquet exists with raw row count."""
    cfg = load_config()
    source_file = ROOT_DIR / cfg["paths"]["source_dir"] / "stroke_source.parquet"
    assert source_file.exists(), f"Source file missing: {source_file}"

    df = load_dataframe(source_file)
    assert len(df) == 5110, f"Expected 5,110 source records, found {len(df)}"
    assert "stroke" in df.columns, "Target column 'stroke' missing from source"


def test_staging_layer_cleaning():
    """Verify staging layer applies age scoping and BMI imputation."""
    cfg = load_config()
    staging_file = ROOT_DIR / cfg["paths"]["staging_dir"] / "stroke_staging.parquet"
    assert staging_file.exists(), f"Staging file missing: {staging_file}"

    df = load_dataframe(staging_file)
    # Target cohort age scope: 35-90
    assert df["age"].min() >= 35.0, "Found age < 35 in staging"
    assert df["age"].max() <= 90.0, "Found age > 90 in staging"
    assert len(df) == 3265, f"Expected 3,265 scoped records, found {len(df)}"

    # Check zero nulls in BMI
    assert df["bmi"].isna().sum() == 0, "Null BMI values present in staging"

    # Gender 'Other' imputed
    assert "Other" not in df["gender"].unique(), "'Other' gender was not imputed"


def test_curated_layer_features():
    """Verify curated layer generates clinical risk scores, VAAF, and TGI."""
    cfg = load_config()
    curated_file = ROOT_DIR / cfg["paths"]["curated_dir"] / "stroke_curated.parquet"
    assert curated_file.exists(), f"Curated file missing: {curated_file}"

    df = load_dataframe(curated_file)
    required_features = [
        "age_score_risk",
        "glucose_score_risk",
        "bmi_score_risk",
        "gender_score_risk",
        "smoking_score_risk",
        "residence_score_risk",
        "VAAF",
        "TGI",
    ]
    for col in required_features:
        assert col in df.columns, f"Engineered column '{col}' missing from curated layer"


def test_model_and_evaluation_artifacts():
    """Verify trained model and evaluation outputs exist."""
    cfg = load_config()
    model_file = ROOT_DIR / cfg["paths"]["model_dir"] / "xgb_model.joblib"
    metrics_file = ROOT_DIR / cfg["paths"]["outputs_dir"] / "metrics.json"
    preds_file = ROOT_DIR / cfg["paths"]["outputs_dir"] / "predictions.csv"

    assert model_file.exists(), "Model binary 'xgb_model.joblib' missing"
    assert metrics_file.exists(), "Evaluation file 'metrics.json' missing"
    assert preds_file.exists(), "Evaluation file 'predictions.csv' missing"