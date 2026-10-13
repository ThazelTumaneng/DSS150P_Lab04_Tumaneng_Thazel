# Reproducibility & Data Lineage Report

**Project:** Stroke Risk Predictive Pipeline  
**Execution Environments Verified:** macOS (Darwin ARM64 / Apple Silicon) & Docker Linux Container (`python:3.11-slim`)  
**Determinism Engine:** Fixed Pseudo-Random Seeds (`seed = 42`)  
**Data Contract Framework:** Pandera Schema Enforcement  

---

## 1. Cryptographic Data Lineage

To guarantee complete auditability, the pipeline treats input data as immutable and computes cryptographic checksums at each layer boundary.

### Raw Source Ingestion Gate
- **Raw File Path:** `data/raw/Stroke_Prediction_Dataset.csv`
- **Algorithm:** Cryptographic Hash Algorithm SHA-256
- **Verified SHA-256 Digest:**
  ```text
  144ea5366832bb5432645c0e49fbb951aadf4ee1e75e73d2e15d3dc0841525bd

### Baseline Gap Analysis

| Architectural Component | Prior Group Implementation | Our Production Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion** | Full unvalidated CSV load | Cryptographic SHA-256 + 3-Batch Incremental with Watermark | Upgraded |
| **Data Contracts** | Implicit Pandas type casts | Pandera Schema Enforcement + Quarantine Isolation | Upgraded |
| **Feature Store** | Standard linear features | Domain-specific ASCVD tiers, VAAF ratio, TGI index | Upgraded |
| **Model Thresholding** | Unweighted default $\tau=0.50$ (55% missed strokes) | Scaled XGBoost (`scale_pos_weight=12.28`) with ASCVD $\tau=0.20$ | Upgraded |
| **Downstream Consumer** | None (ad-hoc notebook) | REST API Inference Endpoint & Partitioned Parquet Store | Upgraded |