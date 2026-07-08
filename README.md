# FairLens — Fairness-Aware Mortgage Loan Approval Modeling (HMDA 2024)

An algorithmic-bias audit of ~800k U.S. mortgage applications: predict loan approval, quantify demographic disparity across racial groups, then apply fairness-constrained optimization and measure the accuracy–fairness tradeoff.

[![Live demo](https://img.shields.io/badge/Live_demo-fairlensweb.vercel.app-000000?style=flat-square&logo=vercel&logoColor=white)](https://fairlensweb.vercel.app)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square&logo=xgboost&logoColor=white)
![Fairlearn](https://img.shields.io/badge/Fairlearn-8A2BE2?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)

> **Interactive demo:** [fairlensweb.vercel.app](https://fairlensweb.vercel.app) — adjust an applicant and watch the approval decision shift by race, with live disparate-impact charts and a Fairlearn debiasing walkthrough.

## Overview

This project investigates algorithmic fairness in mortgage loan approval prediction using the 2024 Home Mortgage Disclosure Act (HMDA) dataset. 

We build and evaluate multiple machine learning models to predict loan approval outcomes and quantify demographic disparities across racial groups. We further implement a fairness-constrained optimization method to reduce bias while analyzing the resulting accuracy–fairness tradeoff.

This project sits at the intersection of:
- Applied Machine Learning
- Responsible AI
- Financial Risk Modeling
- Regulatory-Aligned AI Systems

---

## Problem Statement

Automated credit decision systems can unintentionally reproduce or amplify historical lending disparities. 

This project answers three key questions:

1. Can we accurately predict mortgage approval outcomes using structured financial and demographic data?
2. Do high-performing ML models exhibit measurable demographic disparities?
3. Can fairness-constrained optimization reduce bias while preserving predictive performance?

---

## Dataset

- Source: 2024 HMDA Public LAR dataset (FFIEC)
- Records used: ~800,000 loan applications
- States analyzed: New Jersey (NJ) and Georgia (GA)
- Features: 36 domain-selected underwriting attributes
- Final feature space after encoding: 114 features

Target Variable:
- `approved` (binary loan approval outcome derived from action_taken codes)

Sensitive Attributes Evaluated:
- Race
- Sex
- Ethnicity

---

## Modeling Pipeline

1. Data Cleaning & Filtering (NJ and GA subset)
2. Domain-driven feature selection
3. Missing value imputation (median/mode)
4. One-hot encoding for categorical variables
5. Standard scaling for numeric features
6. Stratified train-test split (70/30)
7. Model training
8. Fairness evaluation
9. Fairness-constrained optimization

---

## Models Implemented

### Baseline Models
- Logistic Regression
- XGBoost (hist-based gradient boosting)
- Feed-forward Neural Network (Keras, GPU-enabled)

### Fairness-Constrained Model
- Fair-XGB (Exponentiated Gradient Reduction with Demographic Parity constraint using Fairlearn)

---

## Fairness Metrics

We evaluate group fairness using:

### Demographic Parity Difference (DPD)
Difference in positive prediction rates between privileged and unprivileged groups.

### Equalized Odds Difference (EOD)
Difference in true positive and false positive rates across demographic groups.

---

## Results

### Race-Based Fairness Comparison

| Model     | Accuracy | DPD (Race) | EOD (Race) |
|-----------|----------|------------|------------|
| LR        | 0.983    | 0.272      | 0.015      |
| XGBoost   | 0.995    | 0.270      | 0.012      |
| Neural Net| 0.989    | 0.266      | 0.011      |
| Fair-XGB  | 0.939    | 0.128      | 0.226      |

### Key Observations

- High predictive accuracy does not imply fairness.
- Baseline models exhibited significant demographic disparities.
- Fair-XGB reduced Demographic Parity Difference by ~53%.
- Fairness improvements resulted in measurable accuracy tradeoff.
- Bias levels were comparable across NJ and GA, suggesting systemic patterns rather than region-specific effects.

---

## Technical Highlights

- GPU-accelerated training (NVIDIA A6000)
- Custom fairness reporting functions
- State-level subgroup analysis
- Hyperparameterized feature subset experimentation
- Integration of Fairlearn’s Exponentiated Gradient reduction

---

## Fairness–Accuracy Tradeoff

The fairness-constrained model significantly reduces demographic parity disparity but increases Equalized Odds difference and decreases overall accuracy. 

This demonstrates the real-world tension between predictive performance and regulatory-aligned fairness constraints in financial systems.

---

## Repository Structure

- `Final_fixed.ipynb` — full analysis: cleaning, feature selection, modeling, fairness evaluation, and constrained optimization
- `Final Report.pdf` — written report
- `README.md`
- `state_mortgage_data.csv` — not included (size constraint); download the HMDA data as described below

---

## Dataset

Due to size constraints, the full HMDA dataset is not included in this repository.

The 2024 HMDA Public LAR dataset can be downloaded from:
https://ffiec.cfpb.gov/data-publication/

After downloading, filter to NJ and GA as shown in the notebook.

---

## Business & Regulatory Relevance

This work is directly relevant to:

- FinTech risk modeling
- Responsible AI initiatives
- Fair lending compliance (ECOA, CFPB alignment)
- Model risk management frameworks

The project demonstrates how fairness-aware optimization can be integrated into production ML pipelines for high-stakes decision systems.

---

## Future Improvements

- Intersectional fairness analysis (e.g., race × gender)
- Equalized Odds constrained optimization
- Causal fairness modeling
- Feature-level bias attribution via SHAP
- Real-time fairness monitoring dashboards

---

## Technologies Used

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Fairlearn
- Matplotlib / Seaborn

---

## Author

Shauryaditya Singh  
Master’s in Applied Artificial Intelligence (Data Engineering)  
Stevens Institute of Technology  

Focus Areas: Applied ML • Responsible AI • FinTech Analytics • Optimization Systems

