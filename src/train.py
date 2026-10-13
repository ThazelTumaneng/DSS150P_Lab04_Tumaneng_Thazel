# src/train.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.config import ROOT_DIR, load_config
from src.storage import ensure_dir, load_dataframe, save_dataframe

HYBRID_FEATURES = [
    "age",
    "age_score_risk",
    "avg_glucose_level",
    "glucose_score_risk",
    "bmi",
    "bmi_score_risk",
    "hypertension",
    "heart_disease",
    "gender_score_risk",
    "smoking_score_risk",
    "residence_score_risk",
    "VAAF",
    "TGI",
]


def train_stroke_model():
    cfg = load_config()
    curated_dir = ROOT_DIR / cfg["paths"]["curated_dir"]
    partitioned_dir = ROOT_DIR / "data/partitioned"
    model_dir = ROOT_DIR / cfg["paths"]["model_dir"]
    split_manifest_file = ROOT_DIR / "metadata/split_manifest.json"

    # 1. Load curated dataset
    curated_file = curated_dir / "stroke_curated.parquet"
    print(f"📥 Loading curated data from: {curated_file}")
    df = load_dataframe(curated_file)

    X = df[HYBRID_FEATURES]
    y = df[cfg["dataset"]["target_column"]]

    # 2. Stratified train/test split
    test_size = cfg["model"].get("test_size", 0.30)
    random_state = cfg["model"].get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Compute class imbalance ratio (negative / positive)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_weight = float(neg_count / pos_count)

    print(f"📊 Training distribution: {neg_count} Non-Stroke vs {pos_count} Stroke")
    print(f"⚖️ Calculated scale_pos_weight: {scale_weight:.2f}")

    # 4. Save partitioned datasets to data/partitioned/
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    save_dataframe(train_df, partitioned_dir / "train.parquet")
    save_dataframe(test_df, partitioned_dir / "test.parquet")
    print(f"📦 Partitions saved to: {partitioned_dir}")

    # 5. Save metadata/split_manifest.json
    manifest_data = {
        "random_state": random_state,
        "test_size": test_size,
        "stratify": True,
        "features": HYBRID_FEATURES,
        "target": cfg["dataset"]["target_column"],
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_positives": pos_count,
        "train_negatives": neg_count,
        "test_positives": int((y_test == 1).sum()),
        "test_negatives": int((y_test == 0).sum()),
        "scale_pos_weight": round(scale_weight, 4),
    }

    ensure_dir(split_manifest_file.parent)
    with open(split_manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"📋 Split manifest written to: {split_manifest_file}")

    # 6. Initialize and fit XGBoost model
    print("🚀 Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=cfg["model"].get("n_estimators", 200),
        learning_rate=cfg["model"].get("learning_rate", 0.03),
        max_depth=cfg["model"].get("max_depth", 3),
        scale_pos_weight=scale_weight,
        subsample=cfg["model"].get("subsample", 0.7),
        colsample_bytree=cfg["model"].get("colsample_bytree", 0.7),
        eval_metric="logloss",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # 7. Save trained model artifact to outputs/model/
    ensure_dir(model_dir)
    model_path = model_dir / "xgb_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model serialized and saved to: {model_path}")

    return model_path


if __name__ == "__main__":
    train_stroke_model()