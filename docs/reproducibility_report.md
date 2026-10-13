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