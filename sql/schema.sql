-- sql/schema.sql
-- Table DDL definitions for the Stroke Prediction Pipeline

-- 1. Raw Layer (Ingested Snapshot)
CREATE TABLE IF NOT EXISTS raw_stroke (
    id INTEGER PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    age DOUBLE PRECISION NOT NULL,
    hypertension INTEGER NOT NULL,
    heart_disease INTEGER NOT NULL,
    ever_married VARCHAR(5) NOT NULL,
    work_type VARCHAR(50) NOT NULL,
    Residence_type VARCHAR(20) NOT NULL,
    avg_glucose_level DOUBLE PRECISION NOT NULL,
    bmi DOUBLE PRECISION,
    smoking_status VARCHAR(50) NOT NULL,
    stroke INTEGER NOT NULL,
    _ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    _source_file VARCHAR(255)
);

-- 2. Staging Layer (Cleaned, Cohort-Filtered, Imputed)
CREATE TABLE IF NOT EXISTS staging_stroke (
    id INTEGER PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    age DOUBLE PRECISION NOT NULL,
    hypertension INTEGER NOT NULL,
    heart_disease INTEGER NOT NULL,
    Residence_type VARCHAR(20) NOT NULL,
    avg_glucose_level DOUBLE PRECISION NOT NULL,
    bmi DOUBLE PRECISION NOT NULL,
    smoking_status VARCHAR(50) NOT NULL,
    stroke INTEGER NOT NULL
);

-- 3. Curated Layer (Model-Ready Feature Store)
CREATE TABLE IF NOT EXISTS curated_stroke (
    id INTEGER PRIMARY KEY,
    age DOUBLE PRECISION NOT NULL,
    age_score_risk INTEGER NOT NULL,
    avg_glucose_level DOUBLE PRECISION NOT NULL,
    glucose_score_risk INTEGER NOT NULL,
    bmi DOUBLE PRECISION NOT NULL,
    bmi_score_risk INTEGER NOT NULL,
    hypertension INTEGER NOT NULL,
    heart_disease INTEGER NOT NULL,
    gender_score_risk INTEGER NOT NULL,
    smoking_score_risk INTEGER NOT NULL,
    residence_score_risk INTEGER NOT NULL,
    VAAF DOUBLE PRECISION NOT NULL,
    TGI DOUBLE PRECISION NOT NULL,
    stroke INTEGER NOT NULL
);