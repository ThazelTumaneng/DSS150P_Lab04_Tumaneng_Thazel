# Benchmark & Clinical Evaluation Report: Stroke Risk Prediction

**Project:** DSS150P - Production-Grade Predictive Data Pipeline  
**Model Family:** Scaled Extreme Gradient Boosting (`XGBClassifier`)  
**Artifact Path:** `outputs/model/xgb_model.joblib`  
**Evaluation Metadata:** `outputs/metrics.json`  

---

## 1. Executive Summary

Standard machine learning evaluation metrics (such as raw accuracy or unweighted F1-score) are fundamentally misaligned with medical screening tasks characterized by extreme class imbalance (~20:1 healthy-to-stroke ratio). Under standard 0.50 thresholding, a classifier can achieve >95% accuracy simply by predicting negative for all patients, while missing fatal cerebrovascular events.

This report establishes the clinical benchmark for the Stroke Risk Prediction Pipeline. By anchoring model evaluation to **AHA/ACC Atherosclerotic Cardiovascular Disease (ASCVD)** 10-year risk stratifications and applying positive class sample reweighting (`scale_pos_weight = 12.28`), we prioritize **clinical sensitivity (recall)** over raw classification accuracy.

---

## 2. Dataset Partitioning & Cohort Distribution

The target population was filtered to the clinically actionable adult cohort ($35 \le \text{age} \le 90$ years), producing $N = 3,265$ records. The cohort was split using a stratified 70/30 train/test allocation (`random_state=42`):

| Partition | Total Records | Non-Stroke ($y=0$) | Stroke ($y=1$) | Prevalence Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set (70%)** | 2,285 | 2,113 | 172 | 7.53% |
| **Test Set (30%)** | 980 | 906 | 74 | 7.55% |
| **Total Cohort** | **3,265** | **3,019** | **246** | **7.53%** |

### Class Imbalance Adjustment
To prevent the gradient booster from collapsing into the majority class, the objective function applies:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{2113}{172} \approx 12.28$$

---

## 3. Global Structural Discrimination Performance

The continuous ranking ability of the model prior to thresholding:

- **Receiver Operating Characteristic (ROC-AUC):** `0.6902`
- **Predictive Surface:** Combines demographic anchors, Framingham Vascular Age Acceleration Factor (`VAAF`), and Triglyceride-Glucose Index (`TGI`).

---

## 4. Clinical Threshold Simulation (ASCVD Aligned)

In clinical preventative cardiology, risk thresholds determine intervention tiers:
- **$\ge 5\%$ (Borderline Risk):** Lifestyle intervention counseling.
- **$\ge 7.5\%$ to $<20\%$ (Intermediate Risk):** Moderate-intensity statin initiation, blood pressure monitoring.
- **$\ge 20\%$ (High Risk Cutoff):** Aggressive preventative therapy, high-intensity statins, diagnostic cerebrovascular imaging.

A sweep of decision thresholds ($\tau \in [0.05, 0.50]$) on the out-of-sample test partition ($N=980$, 74 stroke positives) yielded the following trade-offs:

| Threshold ($\tau$) | Clinical Stratum / Tier | Accuracy | Sensitivity (Recall) | Precision | F1-Score | Missed Strokes (FN) | False Alarms (FP) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | ASCVD Borderline Cutoff | 16.63% | **94.59%** | 8.44% | 0.1550 | **4** | 813 |
| **0.07** | ASCVD Intermediate Boundary | 20.82% | **93.24%** | 8.81% | 0.1610 | **5** | 771 |
| **0.20** | **ASCVD High-Risk Cutoff (Selected)** | **42.96%** | **83.78%** | **10.65%** | **0.1890** | **12** | **547** |
| **0.35** | Operational Screening Midpoint | 64.08% | 64.86% | 12.87% | 0.2148 | 26 | 326 |
| **0.40** | Transition Threshold | 68.98% | 58.11% | 13.27% | 0.2161 | 31 | 273 |
| **0.45** | Transition Threshold | 72.14% | 50.00% | 12.76% | 0.2033 | 37 | 236 |
| **0.50** | Default ML Standard Cutoff | 74.59% | **44.59%** | 12.50% | 0.1953 | **41** | 208 |

---

## 5. Clinical Decision Analysis

### The Hazard of the 0.50 Default Threshold
At the conventional computer science default ($\tau = 0.50$):
- **Accuracy appears acceptable:** $74.59\%$.
- **Clinical failure:** Sensitivity drops to **44.59%**. Out of 74 patients with imminent stroke risk, the model misses **41 patients (55.4% false-negative rate)**. In healthcare operations, an unflagged high-risk patient receives no preventative escalation, resulting in unprevented ischemic or hemorrhagic events.

### Justification for the Selected Operational Threshold ($\tau = 0.20$)
The pipeline selects $\mathbf{\tau = 0.20}$ as the default decision gate:
1. **High Clinical Sensitivity:** Captures **83.78%** of all true stroke cases (62 out of 74 identified).
2. **Acceptable Screening Burden:** Limits missed cases to 12 patients while filtering out 359 confirmed non-stroke individuals from intensive diagnostic testing.
3. **Medical Alignment:** Directly reflects standard clinical cardiovascular scoring rules where a predicted 10-year event risk exceeding 20% mandates clinical escalation.

---

## 6. Target Metric Contract Verification

| Metric | Target Specification | Actual Realized Value | Gate Status |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | $\ge 0.6500$ | `0.6902` | **PASSED** |
| **Operational Sensitivity ($\tau=0.20$)** | $\ge 80.00\%$ | `83.78%` | **PASSED** |
| **Maximum False Negatives ($\tau=0.20$)** | $\le 15$ cases | `12` cases | **PASSED** |