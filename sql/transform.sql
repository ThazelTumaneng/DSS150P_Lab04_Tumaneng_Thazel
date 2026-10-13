-- sql/transformations.sql
-- Relational transformations replicating the Staging and Curated data tiers

-- =========================================================================
-- STEP 1: RAW -> STAGING (Cleaning, Cohort Scoping, Imputation)
-- =========================================================================
INSERT INTO staging_stroke (
    id, gender, age, hypertension, heart_disease,
    Residence_type, avg_glucose_level, bmi, smoking_status, stroke
)
SELECT 
    id,
    -- Rule 1: Impute 'Other' gender using cohort mode ('Female')
    CASE 
        WHEN gender = 'Other' THEN 'Female' 
        ELSE gender 
    END AS gender,
    
    age,
    hypertension,
    heart_disease,
    Residence_type,
    avg_glucose_level,
    
    -- Rule 2: Cohort median imputation for missing BMI (29.8)
    COALESCE(bmi, 29.8) AS bmi,
    
    -- Rule 3: Reclassify pediatric 'Unknown' smoking status to 'never smoked'
    CASE 
        WHEN work_type = 'children' AND smoking_status = 'Unknown' THEN 'never smoked'
        ELSE smoking_status 
    END AS smoking_status,
    
    stroke
FROM raw_stroke
-- Rule 4: Filter target clinical study cohort (ages 35 to 90)
WHERE age >= 35.0 AND age <= 90.0;


-- =========================================================================
-- STEP 2: STAGING -> CURATED (Clinical Risk Scoring & Feature Engineering)
-- =========================================================================
WITH risk_base AS (
    SELECT 
        id,
        age,
        -- Age risk score (35-54: 0, 55-90: 1)
        CASE WHEN age <= 54 THEN 0 ELSE 1 END AS age_score_risk,
        
        avg_glucose_level,
        -- Glycemic tiers: <140 normal (0), 140-199 prediabetic (1), >=200 diabetic (2)
        CASE 
            WHEN avg_glucose_level < 140.0 THEN 0
            WHEN avg_glucose_level < 200.0 THEN 1
            ELSE 2 
        END AS glucose_score_risk,
        
        bmi,
        -- Asian obesity threshold (BMI >= 23.0)
        CASE WHEN bmi < 23.0 THEN 0 ELSE 1 END AS bmi_score_risk,
        
        hypertension,
        heart_disease,
        
        -- Gender score (Female: 0, Male: 1)
        CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS gender_score_risk,
        
        -- Smoking risk tier
        CASE 
            WHEN smoking_status = 'never smoked' THEN 0
            WHEN smoking_status = 'Unknown' THEN 1
            WHEN smoking_status = 'formerly smoked' THEN 2
            WHEN smoking_status = 'smokes' THEN 3
            ELSE 0 
        END AS smoking_score_risk,
        
        -- Residence risk score (Rural: 1, Urban: 2)
        CASE WHEN Residence_type = 'Urban' THEN 2 ELSE 1 END AS residence_score_risk,
        
        -- TGI Index: BMI * Average Glucose Level
        (bmi * avg_glucose_level) AS TGI,
        
        -- Diabetic indicator for Framingham scoring
        CASE WHEN avg_glucose_level >= 126.0 THEN 1 ELSE 0 END AS has_diabetes,
        
        stroke
    FROM staging_stroke
),
framingham_points AS (
    SELECT 
        *,
        (
            -- Age points
            CASE 
                WHEN age < 35 THEN 0
                WHEN age < 40 THEN 2
                WHEN age < 45 THEN (CASE WHEN gender_score_risk = 1 THEN 5 ELSE 4 END)
                WHEN age < 50 THEN (CASE WHEN gender_score_risk = 1 THEN 6 ELSE 5 END)
                WHEN age < 55 THEN (CASE WHEN gender_score_risk = 1 THEN 8 ELSE 7 END)
                WHEN age < 60 THEN (CASE WHEN gender_score_risk = 1 THEN 10 ELSE 8 END)
                WHEN age < 65 THEN (CASE WHEN gender_score_risk = 1 THEN 11 ELSE 9 END)
                WHEN age < 70 THEN (CASE WHEN gender_score_risk = 1 THEN 12 ELSE 10 END)
                WHEN age < 75 THEN (CASE WHEN gender_score_risk = 1 THEN 14 ELSE 11 END)
                ELSE (CASE WHEN gender_score_risk = 1 THEN 15 ELSE 12 END)
            END
            -- Smoking points
            + CASE WHEN smoking_score_risk = 3 THEN (CASE WHEN gender_score_risk = 1 THEN 4 ELSE 3 END) ELSE 0 END
            -- Hypertension points
            + CASE WHEN hypertension = 1 THEN (CASE WHEN gender_score_risk = 1 THEN 3 ELSE 4 END) ELSE (CASE WHEN gender_score_risk = 1 THEN 0 ELSE -1 END) END
            -- Diabetes points
            + CASE WHEN has_diabetes = 1 THEN (CASE WHEN gender_score_risk = 1 THEN 3 ELSE 4 END) ELSE 0 END
            -- BMI points
            + CASE WHEN bmi >= 25.0 THEN 2 ELSE 0 END
        ) AS total_points
    FROM risk_base
),
vascular_age_calculation AS (
    SELECT 
        *,
        CASE 
            WHEN gender_score_risk = 1 THEN
                CASE 
                    WHEN total_points <= -1 THEN 28
                    WHEN total_points = 0 THEN 30
                    WHEN total_points <= 2 THEN 33
                    WHEN total_points <= 4 THEN 37
                    WHEN total_points <= 6 THEN 41
                    WHEN total_points <= 8 THEN 46
                    WHEN total_points <= 10 THEN 52
                    WHEN total_points <= 12 THEN 58
                    WHEN total_points <= 14 THEN 66
                    WHEN total_points = 15 THEN 72
                    WHEN total_points = 16 THEN 76
                    ELSE 82
                END
            ELSE
                CASE 
                    WHEN total_points <= 1 THEN 28
                    WHEN total_points <= 3 THEN 32
                    WHEN total_points <= 5 THEN 38
                    WHEN total_points <= 7 THEN 43
                    WHEN total_points <= 9 THEN 49
                    WHEN total_points <= 11 THEN 57
                    WHEN total_points <= 13 THEN 66
                    WHEN total_points = 14 THEN 76
                    ELSE 82
                END
        END AS vascular_age
    FROM framingham_points
)
INSERT INTO curated_stroke (
    id, age, age_score_risk, avg_glucose_level, glucose_score_risk,
    bmi, bmi_score_risk, hypertension, heart_disease,
    gender_score_risk, smoking_score_risk, residence_score_risk,
    VAAF, TGI, stroke
)
SELECT 
    id,
    age,
    age_score_risk,
    avg_glucose_level,
    glucose_score_risk,
    bmi,
    bmi_score_risk,
    hypertension,
    heart_disease,
    gender_score_risk,
    smoking_score_risk,
    residence_score_risk,
    (vascular_age - age) AS VAAF,
    TGI,
    stroke
FROM vascular_age_calculation;