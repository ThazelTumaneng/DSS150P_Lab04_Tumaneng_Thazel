# src/evaluate.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.config import ROOT_DIR, load_config
from src.storage import load_dataframe, save_dataframe, ensure_dir
from src.train import HYBRID_FEATURES


def evaluate_model():
    cfg = load_config()
    test_path = ROOT_DIR / "data/partitioned/test.parquet"
    model_path = ROOT_DIR / cfg["paths"]["model_dir"] / "xgb_model.joblib"
    outputs_dir = ROOT_DIR / cfg["paths"]["outputs_dir"]
    ensure_dir(outputs_dir)

    print(f"📥 Loading test partition from: {test_path}")
    test_df = load_dataframe(test_path)
    X_test = test_df[HYBRID_FEATURES]
    y_test = test_df[cfg["dataset"]["target_column"]]

    print(f"📥 Loading trained model from: {model_path}")
    model = joblib.load(model_path)

    # 1. Compute predicted probabilities and ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = float(roc_auc_score(y_test, y_prob))
    print(f"⭐ Structural Model ROC-AUC Score: {auc_score:.4f}")

    # 2. ASCVD-Aligned Clinical Threshold Simulation
    simulated_thresholds = [0.05, 0.07, 0.20, 0.35, 0.40, 0.45, 0.50]
    threshold_results = []

    print("\n" + "=" * 95)
    print("           CLINICAL THRESHOLD SIMULATION (ASCVD ALIGNED RISK TIERS)")
    print("=" * 95)

    for thresh in simulated_thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        acc = float(accuracy_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        tier_name = (
            "ASCVD Borderline" if thresh == 0.05
            else "ASCVD High-Risk Cutoff" if thresh == 0.20
            else "Balanced Screening Point" if thresh == 0.35
            else "Default ML Standard" if thresh == 0.50
            else "Intermediate Tier"
        )

        res = {
            "threshold": thresh,
            "clinical_tier": tier_name,
            "accuracy": round(acc, 4),
            "recall_sensitivity": round(rec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
        threshold_results.append(res)
        print(f"Threshold: {thresh:.2f} [{tier_name:<26}] | Acc: {acc*100:6.2f}% | Sensitivity: {rec*100:6.2f}% | Missed (FN): {fn:2d}")

    # 3. Save outputs/metrics.json based on chosen clinical cutoff (0.20)
    selected_threshold = cfg["model"].get("clinical_threshold", 0.20)
    y_pred_selected = (y_prob >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_selected).ravel()

    metrics_payload = {
        "model_name": "Hybrid_XGBoost_Classifier",
        "roc_auc_score": round(auc_score, 4),
        "selected_operational_threshold": selected_threshold,
        "selected_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_selected)), 4),
            "recall_sensitivity": round(float(recall_score(y_test, y_pred_selected)), 4),
            "precision": round(float(precision_score(y_test, y_pred_selected, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_selected, zero_division=0)), 4),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        },
        "threshold_simulations": threshold_results,
    }

    metrics_file = outputs_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n📊 Evaluation metrics saved to: {metrics_file}")

    # 4. Save outputs/predictions.csv
    pred_df = test_df.copy()
    pred_df["predicted_stroke_probability"] = np.round(y_prob, 4)
    pred_df["predicted_stroke_flag"] = y_pred_selected
    predictions_file = outputs_dir / "predictions.csv"
    save_dataframe(pred_df, predictions_file, file_format="csv")
    print(f"💾 Predictions saved to: {predictions_file}")

    return metrics_file


if __name__ == "__main__":
    evaluate_model()