# src/transform.py
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT_DIR, load_config
from src.storage import load_dataframe, save_dataframe


def clean_to_staging(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data into the Staging layer:
      - Deduplicates records by primary key ('id')
      - Imputes 'Other' gender using the statistical mode ('Female')
      - Fixes pediatric smoking status ('Unknown' -> 'never smoked' for children)
      - Filters the target cohort age (35 to 90 years old)
      - Imputes missing BMI values using the cohort median (29.8)
    """
    print("🧹 Cleaning data for Staging layer...")
    df = df.copy()

    # 1. Deduplicate by ID
    df = df.drop_duplicates(subset=["id"])

    # 2. Impute 'Other' gender using statistical mode
    gender_mode = df[df["gender"] != "Other"]["gender"].mode()[0]
    df["gender"] = df["gender"].replace({"Other": gender_mode})

    # 3. Correct pediatric smoking status
    df.loc[
        (df["work_type"] == "children") & (df["smoking_status"] == "Unknown"),
        "smoking_status",
    ] = "never smoked"

    # 4. Target population cohort filter (Ages 35 - 90)
    df = df[(df["age"] >= 35.0) & (df["age"] <= 90.0)].copy()

    # 5. Cohort median imputation for missing BMI
    cohort_bmi_median = df["bmi"].median()  # Expected: 29.8
    df["bmi"] = df["bmi"].fillna(cohort_bmi_median)

    # Assertions to ensure data quality
    assert df["bmi"].isna().sum() == 0, "Missing BMI values found after imputation"
    assert df["age"].between(35, 90).all(), "Ages outside 35-90 detected in staging"
    assert set(df["gender"].unique()).issubset({"Male", "Female"}), "Unresolved gender values"

    return df


def calculate_frs_vascular_age(row: pd.Series) -> int:
    """
    Computes Vascular Age based on the adapted Framingham Risk Score 
    (FRS-BMI-HnT) point system.
    """
    age = row["age"]
    is_male = row["gender_score_risk"] == 1
    smokes = row["smoking_score_risk"] == 3
    has_hyper = row["hypertension"] == 1
    has_diabetes = row["avg_glucose_level"] >= 126.0

    points = 0

    # Age points
    if age < 35: points += 0
    elif age < 40: points += 2
    elif age < 45: points += 5 if is_male else 4
    elif age < 50: points += 6 if is_male else 5
    elif age < 55: points += 8 if is_male else 7
    elif age < 60: points += 10 if is_male else 8
    elif age < 65: points += 11 if is_male else 9
    elif age < 70: points += 12 if is_male else 10
    elif age < 75: points += 14 if is_male else 11
    else: points += 15 if is_male else 12

    # Smoking points
    if smokes:
        points += 4 if is_male else 3

    # Hypertension points
    if has_hyper:
        points += 3 if is_male else 4
    else:
        points += 0 if is_male else -1

    # Diabetes indicator
    if has_diabetes:
        points += 3 if is_male else 4

    # BMI points (Asian obesity threshold indicator)
    if row["bmi"] >= 25.0:
        points += 2

    # Map points to Vascular Age
    if is_male:
        if points <= -1: v_age = 28
        elif points == 0: v_age = 30
        elif points <= 2: v_age = 33
        elif points <= 4: v_age = 37
        elif points <= 6: v_age = 41
        elif points <= 8: v_age = 46
        elif points <= 10: v_age = 52
        elif points <= 12: v_age = 58
        elif points <= 14: v_age = 66
        elif points == 15: v_age = 72
        elif points == 16: v_age = 76
        else: v_age = 82
    else:
        if points <= 1: v_age = 28
        elif points <= 3: v_age = 32
        elif points <= 5: v_age = 38
        elif points <= 7: v_age = 43
        elif points <= 9: v_age = 49
        elif points <= 11: v_age = 57
        elif points <= 13: v_age = 66
        elif points == 14: v_age = 76
        else: v_age = 82

    return v_age


def engineer_curated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain risk anchors and composite indices for the Curated layer:
      - Gender, age, glucose, BMI, smoking, and residence risk scores
      - FRS-BMI-HnT vascular age
      - VAAF (Vascular Age - Chronological Age)
      - TGI (BMI * Glucose Level)
    """
    print("🧬 Engineering clinical features for Curated layer...")
    df = df.copy()

    # 1. Clinical Risk Scores
    df["gender_score_risk"] = df["gender"].map({"Female": 0, "Male": 1}).astype(int)
    df["age_score_risk"] = pd.cut(df["age"], bins=[34, 54, 90], labels=[0, 1]).astype(int)
    df["glucose_score_risk"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 140, 200, np.inf],
        labels=[0, 1, 2],
        right=False
    ).astype(int)
    df["bmi_score_risk"] = pd.cut(
        df["bmi"],
        bins=[0, 23, np.inf],
        labels=[0, 1],
        right=False
    ).astype(int)
    df["smoking_score_risk"] = df["smoking_status"].map({
        "never smoked": 0,
        "Unknown": 1,
        "formerly smoked": 2,
        "smokes": 3
    }).astype(int)
    df["residence_score_risk"] = df["Residence_type"].map({"Rural": 1, "Urban": 2}).astype(int)

    # 2. Drop unneeded categorical columns
    df = df.drop(columns=["ever_married", "work_type"], errors="ignore")

    # 3. Cardiovascular Risk Indices
    vascular_age = df.apply(calculate_frs_vascular_age, axis=1)
    df["VAAF"] = vascular_age - df["age"]
    df["TGI"] = df["bmi"] * df["avg_glucose_level"]

    return df


def run_transformations() -> Path:
    cfg = load_config()
    source_dir = ROOT_DIR / cfg["paths"]["source_dir"]
    staging_dir = ROOT_DIR / cfg["paths"]["staging_dir"]
    curated_dir = ROOT_DIR / cfg["paths"]["curated_dir"]

    # Load validated source file
    source_file = source_dir / "stroke_source.parquet"
    df_source = load_dataframe(source_file)

    # 1. Staging Step
    df_staging = clean_to_staging(df_source)
    staging_file = staging_dir / "stroke_staging.parquet"
    save_dataframe(df_staging, staging_file)
    print(f"✅ Staging complete: {len(df_staging)} rows saved to {staging_file}")

    # 2. Curated Step
    df_curated = engineer_curated_features(df_staging)
    curated_file = curated_dir / "stroke_curated.parquet"
    save_dataframe(df_curated, curated_file)
    print(f"✅ Curated complete: {len(df_curated)} rows, {df_curated.shape[1]} columns saved to {curated_file}")

    return curated_file


if __name__ == "__main__":
    run_transformations()