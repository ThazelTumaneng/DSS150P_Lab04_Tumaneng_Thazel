"""
src/transform.py
Transforms ingested source data into clean Staging and Curated Medallion tiers.
Applies domain-specific clinical feature engineering:
- Asian BMI risk tiers (cutoff >= 23.0 kg/m^2)
- ADA/WHO Fasting Glycemic tiers (<100 Normal, 100-125 Pre-diabetic, >=126 Diabetic)
- Triglyceride-Glucose Index (TGI) with natural log scaling
- Framingham Vascular Age Acceleration Factor (VAAF) as a true ratio
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.storage import load_dataframe, save_dataframe, export_multiformat

# Configure module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard directory resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT_DIR / "data/source/stroke_source.parquet"
STAGING_PATH = ROOT_DIR / "data/staging/stroke_staging.parquet"
CURATED_PATH = ROOT_DIR / "data/curated/stroke_curated.parquet"


def clean_staging(df: pd.DataFrame) -> pd.DataFrame:
    """Filters target adult cohort (35-90) and performs cleaning without feature leakage."""
    logger.info("Starting staging data cleaning...")
    
    # 1. Target Clinical Cohort Filter (35 <= age <= 90)
    # Prunes pediatric & young adult records where stroke prevalence is <0.1%
    df_clean = df[(df["age"] >= 35.0) & (df["age"] <= 90.0)].copy()
    logger.info(f"Filtered cohort size: {len(df_clean)} rows (pruned from {len(df)})")
    
    # 2. Audit Missing BMI Count in Target Cohort
    # In cohort 35-90, exactly 154 BMI records are missing (201 was across all 5,110 raw rows)
    missing_bmi_count = df_clean["bmi"].isna().sum()
    logger.info(f"Missing BMI values in target adult cohort: {missing_bmi_count} (Expected: 154)")
    
    # 3. Categorical Imputation: Gender
    mode_gender = df_clean["gender"].mode()[0]
    df_clean["gender"] = df_clean["gender"].replace("Other", mode_gender)
    
    # 4. Domain Logic Correction: Children work type cannot have smoking history
    mask_children = df_clean["work_type"] == "children"
    df_clean.loc[mask_children, "smoking_status"] = "never smoked"
    
    # 5. Impute Missing BMI with Cohort Median (29.80 kg/m^2)
    cohort_median_bmi = round(float(df_clean["bmi"].median()), 2)
    df_clean["bmi"] = df_clean["bmi"].fillna(cohort_median_bmi)
    logger.info(f"Imputed missing BMI values using cohort median: {cohort_median_bmi}")
    
    return df_clean


def calculate_vascular_age(df: pd.DataFrame) -> pd.Series:
    """
    Computes Vascular Age based on Framingham Risk Score components:
    Age, sex, hypertension, heart disease, smoking, diabetes, and BMI.
    """
    is_male = df["gender_score_risk"] == 1
    smokes = df["smoking_score_risk"] == 3
    has_hyper = df["hypertension"] == 1
    has_diabetes = df["avg_glucose_level"] >= 126.0
    elevated_bmi = df["bmi"] >= 25.0

    vasc_age = df["age"].copy()
    vasc_age += np.where(has_hyper, np.where(is_male, 3.0, 4.0), np.where(is_male, 0.0, -1.0))
    vasc_age += np.where(smokes, np.where(is_male, 4.0, 3.0), 0.0)
    vasc_age += np.where(has_diabetes, np.where(is_male, 2.0, 3.0), 0.0)
    vasc_age += np.where(elevated_bmi, 1.5, 0.0)
    vasc_age += df["heart_disease"] * 5.0
    return vasc_age


def engineer_curated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives clinical hybrid features matching src/train.py expectations."""
    logger.info("Computing domain-specific cardiovascular features...")
    df_curated = df.copy()

    # 1. Demographic & Clinical Categorical Risk Scores
    df_curated["gender_score_risk"] = df_curated["gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)
    df_curated["age_score_risk"] = pd.cut(df_curated["age"], bins=[34, 54, 90], labels=[0, 1]).astype(int)
    
    # 2. ADA/WHO Fasting Glycemic Tiers (<100 Normal: 0, 100-125.99 Pre-diabetic: 1, >=126 Diabetic: 2)
    df_curated["glucose_score_risk"] = pd.cut(
        df_curated["avg_glucose_level"],
        bins=[-np.inf, 99.99, 125.99, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # 3. Asian BMI Threshold (<23.0 Normal: 0, >=23.0 Elevated Risk: 1)
    df_curated["bmi_score_risk"] = pd.cut(
        df_curated["bmi"],
        bins=[-np.inf, 22.99, np.inf],
        labels=[0, 1]
    ).astype(int)

    # 4. Lifestyle & Residence Risk Scores
    df_curated["smoking_score_risk"] = df_curated["smoking_status"].map({
        "never smoked": 0,
        "Unknown": 1,
        "formerly smoked": 2,
        "smokes": 3
    }).fillna(0).astype(int)
    df_curated["residence_score_risk"] = df_curated["Residence_type"].map({"Rural": 1, "Urban": 2}).fillna(1).astype(int)

    # 5. Framingham Vascular Age & VAAF Ratio (Vascular Age / Chronological Age)
    df_curated["vascular_age"] = calculate_vascular_age(df_curated)
    df_curated["VAAF"] = df_curated["vascular_age"] / df_curated["age"]

    # 6. Triglyceride-Glucose Index (TGI): ln((glucose * bmi) / 2)
    df_curated["TGI"] = np.log((df_curated["avg_glucose_level"] * df_curated["bmi"]) / 2.0)

    # 7. Strip Metadata Audit Columns to prevent data leakage in training
    audit_columns = ["row_hash", "pipeline_run_id", "ingested_at"]
    df_curated = df_curated.drop(columns=[col for col in audit_columns if col in df_curated.columns], errors="ignore")

    return df_curated


def run_transformations() -> pd.DataFrame:
    """Executes full transformation workflow: Source -> Staging -> Curated -> Multi-format Storage."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source Parquet not found at {SOURCE_PATH}. Run ingestion first.")

    logger.info(f"Loading source dataset from: {SOURCE_PATH}")
    df_source = load_dataframe(SOURCE_PATH)

    # Staging Layer Execution
    df_staging = clean_staging(df_source)
    save_dataframe(df_staging, STAGING_PATH)
    logger.info(f"Staging dataset saved to: {STAGING_PATH} ({len(df_staging)} rows)")

    # Curated Layer Execution
    df_curated = engineer_curated_features(df_staging)
    save_dataframe(df_curated, CURATED_PATH)
    logger.info(f"Curated dataset saved to: {CURATED_PATH} ({len(df_curated)} rows, {df_curated.shape[1]} cols)")

    # Multi-format Storage Exports (JSONL + Age-Partitioned Parquet)
    logger.info("Exporting multi-format representations for downstream analytics...")
    export_multiformat(df_curated, CURATED_PATH)
    logger.info("Multi-format export complete.")

    return df_curated


if __name__ == "__main__":
    run_transformations()