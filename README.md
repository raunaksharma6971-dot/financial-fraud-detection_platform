# Financial Fraud Detection_Platform

---

## Live Demo
https://financial-fraud-detection-dashboard.streamlit.app/

---

## Project Overview
This project implements an end-to-end financial fraud detection platform designed to identify high-risk credit card transactions using advanced predictive analytics. The solution combines a machine learning model optimized for highly imbalanced data with an interactive web dashboard to support operational decision-making and fraud review workflows.

The project demonstrates practical skills in financial risk modeling, model evaluation under class imbalance, threshold optimization, data engineering, and cloud deployment.

---

## Business Problem
Financial institutions face significant challenges in detecting fraudulent transactions due to extreme class imbalance, where fraudulent events represent a very small fraction of total activity. Traditional accuracy-based models fail to capture this risk effectively.

This project focuses on:
- Maximizing fraud detection while controlling false positives
- Supporting operational decision-making through adjustable risk thresholds
- Presenting results in a clear, interpretable, and interactive format

---

## Data
- Publicly available, anonymized credit card transaction dataset
- Over 280,000 transactions with severe class imbalance
- Sensitive raw data is excluded from the repository for security and compliance

A curated, scored sample dataset is generated for dashboard visualization and deployment.

---

## Modeling Approach
- Algorithm: XGBoost (gradient boosted decision trees)
- Objective: Binary classification (fraud vs non-fraud)
- Imbalance handling:
  - Class weighting using scale_pos_weight
  - Evaluation using Precision-Recall AUC rather than accuracy
- Threshold optimization:
  - Precision-constrained threshold selection
  - Explicit trade-off analysis between fraud capture and review volume

---

## Model Performance
- Precision-Recall AUC: ~0.88
- ROC AUC: ~0.98
- Precision at selected threshold: ~0.91
- Recall at selected threshold: ~0.81

Metrics are stored as versioned artifacts and used directly in the dashboard.

---

## Application Features
The deployed dashboard enables users to:
- Adjust fraud probability thresholds dynamically
- View precision and recall trade-offs in real time
- Inspect confusion matrices at different thresholds
- Analyze predicted fraud probability distributions
- Review high-risk transactions by score and amount
- Understand operational impact through review volume analysis

---

## Project Structure
financial-fraud-detection_platform/
├── app/
│ └── streamlit_app.py
├── src/
│ ├── train_xgboost.py
│ └── prepare_app_data.py
├── models/
│ ├── xgb_model.joblib
│ ├── metrics.json
│ └── app_sample.parquet
├── data/
│ └── (ignored: raw dataset not committed)
├── requirements.txt
├── .gitignore
└── README.md


---

## To Run Locally
1. Clone the repository
2. Install dependencies:
pip install -r requirements.txt
3. Run the Streamlit application:
streamlit run app/streamlit_app.py


---

## Deployment
- Hosted on Streamlit Community Cloud
- Continuous deployment from GitHub main branch
- Model artifacts and app-ready datasets are versioned and included
- Raw training data excluded to ensure reproducibility and compliance

---

## Key Skills Demonstrated
- Financial risk analytics
- Machine learning with imbalanced datasets
- XGBoost model development and evaluation
- Threshold optimization and operational trade-offs
- Python data engineering
- Streamlit dashboard development
- Cloud deployment and GitHub-based workflows

---

## Author
Raunak Sharma


