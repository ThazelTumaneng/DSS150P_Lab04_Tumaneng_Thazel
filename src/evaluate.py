"""
src/evaluate.py
Evaluates trained XGBoost stroke model on the out-of-sample test partition.
Simulates ASCVD clinical risk tiers (tau in [0.05, 0.50]), records metrics,
and tests performance against acceptance contracts in metadata/expected_metrics.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.storage import ensure_dir, load_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = ROOT_DIR / "data/partitioned/test.parquet"
MODEL_PATH = ROOT_DIR / "outputs/model/xgb_model.joblib"
METRICS_PATH = ROOT_DIR / "outputs/metrics.json"
PREDICTIONS_PATH = ROOT_DIR / "outputs/predictions.csv"
CONTRACT_PATH = ROOT_DIR / "metadata/expected_metrics.json"


def verify_acceptance_contract(realized_auc: float, realized_sens: float, realized_fn: int):
    """Verifies realized metrics against baseline tolerance contracts if present."""
    if not CONTRACT_PATH.exists():
        logger.warning(f"No contract file found at {CONTRACT_PATH}. Skipping gate.")
        return

    with open(CONTRACT_PATH, "r") as f:
        contract = json.load(f)

    tolerances = contract.get("acceptance_tolerances", {})
    min_auc = tolerances.get("min_roc_auc", 0.6500)
    min_sens = tolerances.get("min_sensitivity_at_0_20", 0.8000)
    max_fn = tolerances.get("max_false_negatives_at_0_20", 15)

    logger.info(f"Checking acceptance contract (AUC: {realized_auc:.4f} >= {min_auc}, Sens: {realized_sens:.4f} >= {min_sens})")

    if realized_auc < min_auc:
        raise AssertionError(f"Model failed ROC-AUC contract! Realized {realized_auc:.4f} < Required {min_auc:.4f}")

    if realized_sens < min_sens:
        raise AssertionError(f"Model failed Sensitivity contract! Realized {realized_sens:.4f} < Required {min_sens:.4f}")

    if realized_fn > max_fn:
        raise AssertionError(f"Model failed False Negatives contract! Realized {realized_fn} > Allowed {max_fn}")

    logger.info("All model benchmark acceptance contracts PASSED.")


def evaluate_model() -> Dict[str, Any]:
    """Evaluates the test partition across clinical ASCVD risk thresholds."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test partition missing at {TEST_DATA_PATH}. Run training first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model missing at {MODEL_PATH}. Run training first.")

    logger.info(f"Loading test dataset from: {TEST_DATA_PATH}")
    test_df = load_dataframe(TEST_DATA_PATH)

    logger.info(f"Loading trained model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    drop_cols = ["stroke", "id", "row_hash", "pipeline_run_id", "ingested_at"]
    feature_cols = [c for c in test_df.columns if c not in drop_cols]

    X_test = pd.get_dummies(test_df[feature_cols], drop_first=True)
    y_test = test_df["stroke"].astype(int).values

    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[model.feature_names_in_]

    y_prob = model.predict_proba(X_test)[:, 1]

    ensure_dir(PREDICTIONS_PATH.parent)
    predictions_df = test_df.copy()
    predictions_df["predicted_stroke_prob"] = y_prob
    predictions_df["predicted_high_risk_flag_0_20"] = (y_prob >= 0.20).astype(int)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info(f"Inference probabilities saved to: {PREDICTIONS_PATH}")

    roc_auc = float(roc_auc_score(y_test, y_prob))

    simulated_thresholds = [0.05, 0.07, 0.20, 0.35, 0.40, 0.45, 0.50]
    simulations = []
    selected_metrics = {}

    for thresh in simulated_thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        acc = float(accuracy_score(y_test, y_pred))
        sens = float(recall_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        sim_entry = {
            "threshold": thresh,
            "accuracy": round(acc, 4),
            "sensitivity_recall": round(sens, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
        simulations.append(sim_entry)

        if thresh == 0.20:
            selected_metrics = sim_entry

    metrics_payload = {
        "test_sample_size": len(y_test),
        "test_stroke_cases": int(np.sum(y_test)),
        "roc_auc": round(roc_auc, 4),
        "selected_operational_threshold": 0.20,
        "selected_threshold_metrics": selected_metrics,
        "all_threshold_simulations": simulations,
    }

    ensure_dir(METRICS_PATH.parent)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Evaluation metrics written to: {METRICS_PATH}")
    logger.info(f"ROC-AUC: {roc_auc:.4f} | Operational (tau=0.20) Recall: {selected_metrics['sensitivity_recall']*100:.2f}% | FN: {selected_metrics['false_negatives']}")

    verify_acceptance_contract(
        realized_auc=roc_auc,
        realized_sens=selected_metrics["sensitivity_recall"],
        realized_fn=selected_metrics["false_negatives"],
    )

    return metrics_payload


run_evaluation = evaluate_model

if __name__ == "__main__":
    evaluate_model()