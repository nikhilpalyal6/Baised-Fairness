# Classification of Bias in Automated Hiring Systems using Counterfactual Fairness

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🚀 Live Interactive Demo:** [https://fairrecruit-ai.streamlit.app](https://fairrecruit-ai.streamlit.app) *(Replace with your deployed Streamlit link)*

---

## 📌 Overview

This project presents a fairness-aware machine learning framework for detecting and auditing bias in automated hiring systems. The proposed approach combines traditional fairness metrics with counterfactual analysis techniques (**Predictive Counterfactual Fairness - PreCoF**) to identify bias-sensitive decision regions while maintaining high predictive performance.

The framework evaluates multiple machine learning models on both real-world and controlled datasets to analyze the trade-off between prediction accuracy and algorithmic fairness.

---

## 🎯 Objectives

- Detect hidden bias in automated hiring decisions.
- Compare the fairness of multiple machine learning models (Logistic Regression, Random Forest, XGBoost).
- Introduce a counterfactual fairness metric (PreCoF) for decision-level bias analysis:
  $$\mathcal{P}(\hat{Y}_{A \leftarrow a} = 1 \mid X = x) = \mathcal{P}(\hat{Y}_{A \leftarrow a'} = 1 \mid X = x)$$
- Develop an auditor model capable of classifying decision regions as **Fair** or **Bias-Prone**.
- Improve transparency and interpretability in AI-based recruitment systems.

---

## 🚀 Interactive Web Application (`app.py`)

An interactive Streamlit web dashboard has been built for live auditing and candidate testing:

1. **Candidate Profile Simulator:** Test applicant qualifications (scores, experience, role) in real time.
2. **Counterfactual Twin Generator:** 1-click test that flips protected attributes (Gender, Age) while keeping qualifications identical to evaluate algorithmic invariance.
3. **Legal Compliance Sandbox:** Live tracking under the **US EEOC 80% (Four-Fifths) Disparate Impact Rule**.
4. **Model Debiasing Toggle:** Instant comparison between standard unmitigated ML vs. debiased fairness-calibrated models.

### Run Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Datasets

### Adult Income Dataset
- **Source:** UCI Machine Learning Repository
- **Instances:** 48,842 | **Features:** 14
- **Target:** Income >50K vs ≤50K
- **Purpose:** Real-world benchmark fairness evaluation

### AI Recruitment Dataset
- **Instances:** 121,200 | **Features:** 15
- **Target:** Selected vs Rejected
- **Purpose:** Controlled algorithmic fairness analysis

---

## ⚙️ Data Preprocessing Pipeline

- Missing value handling via median and modal imputation
- Identifier attribute stripping (`Candidate_ID`, etc.)
- One-Hot Encoding for categorical features
- Z-score Standard Scaling for numerical features
- Proxy feature removal using correlation thresholds
- Stratified Train-Test Splitting (80:20) across 5 random seeds (42, 52, 62, 72, 82)

---

## 🤖 Machine Learning Models & Results

| Dataset | Model | Accuracy | ROC-AUC | Disparate Impact Ratio | EEOC 80% Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adult Income** | Logistic Regression | 83% | 0.902 | 0.323 | ❌ Adverse Impact |
| | Random Forest | 84% | 0.915 | 0.256 | ❌ Adverse Impact |
| | XGBoost | **86%** | **0.928** | 0.332 | ❌ Adverse Impact |
| **AI Recruitment** | Baseline LR | **99%** | **0.990** | 0.380 (Borderline) | ❌ Disparate Impact |
| | **Fair Model (Mitigated)** | **92%** | **0.942** | **0.985** | ✅ **Compliant** |

---

## 📁 Repository Structure

```
├── app.py                     # Streamlit Interactive Demo Web Application
├── models/                    # Serialized lightweight production models & metadata
├── data/                      # Benchmark datasets (Adult & AI Recruitment)
├── results/                   # Experimental results, metric CSVs & publication figures
├── src/                       # Preprocessing and utility modules
├── project_complete.py        # Complete training and evaluation pipeline
├── requirements.txt           # Production dependencies for cloud deployment
└── README.md                  # Comprehensive project documentation
```

---

## 🛠️ Technologies Used

- **Python, Streamlit, Plotly**
- **Scikit-learn, XGBoost, Pandas, NumPy**
- **Matplotlib, Seaborn**
