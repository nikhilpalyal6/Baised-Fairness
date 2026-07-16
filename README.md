# Classification of Bias in Automated Hiring Systems using Counterfactual Fairness

## 📌 Overview

This project presents a fairness-aware machine learning framework for detecting and auditing bias in automated hiring systems. The proposed approach combines traditional fairness metrics with a novel counterfactual analysis technique called **Predictive Counterfactual Fairness (PreCoF)** to identify bias-sensitive decision regions while maintaining high predictive performance.

The framework evaluates multiple machine learning models on both real-world and controlled datasets to analyze the trade-off between prediction accuracy and algorithmic fairness.

---

## 🎯 Objectives

- Detect hidden bias in automated hiring decisions.
- Compare the fairness of multiple machine learning models.
- Introduce a counterfactual fairness metric (PreCoF) for decision-level bias analysis.
- Develop an auditor model capable of classifying decision regions as **Fair** or **Bias-Prone**.
- Improve transparency and interpretability in AI-based recruitment systems.

---

## 📂 Datasets

### Adult Income Dataset
- Source: UCI Machine Learning Repository
- Instances: 48,842
- Features: 14
- Target:
  - Income >50K
  - Income ≤50K
- Purpose:
  - Real-world fairness evaluation

### AI Recruitment Dataset
- Synthetic hiring dataset
- Instances: 121,200
- Features: 15
- Target:
  - Selected
  - Rejected
- Purpose:
  - Controlled fairness analysis

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Removal of identifier attributes
- Feature selection
- One-Hot Encoding
- Z-score Normalization
- Proxy feature removal using Pearson Correlation (>0.8)
- Controlled Gaussian noise injection (±5%)
- Stratified Train-Test Split (80:20)

---

## 🤖 Machine Learning Models

The following benchmark models were implemented:

- Logistic Regression
- Random Forest
- XGBoost

Each model was trained using:

- RandomizedSearchCV
- 5-Fold Cross Validation
- Threshold Optimization
- Probability Calibration

---

## 🧠 Proposed Fairness Auditing Framework

The proposed framework consists of the following stages:

```
Datasets
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
PreCoF Counterfactual Analysis
      ↓
Bias Auditor Model
      ↓
Fairness Metrics Evaluation
      ↓
Performance Comparison
```

---

## 📊 Fairness Metrics

The framework evaluates fairness using:

- Demographic Parity Difference
- Equal Opportunity Difference
- Equalized Odds Difference
- Disparate Impact Ratio
- Predictive Counterfactual Fairness (PreCoF)

---

## 📈 Results

### Adult Income Dataset

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 83% |
| Random Forest | 84% |
| XGBoost | **86%** |

### AI Recruitment Dataset

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **99%** |
| Random Forest | 96% |
| XGBoost | 98% |

Key observations:

- XGBoost achieved the highest accuracy on the Adult Income dataset.
- Logistic Regression demonstrated the best fairness-performance balance.
- PreCoF successfully identified bias-sensitive decision regions.
- Removing proxy variables significantly reduced fairness disparity.

---

## 📁 Repository Structure

```
├── datasets/
├── notebooks/
├── models/
├── results/
├── figures/
├── src/
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📚 Research Contribution

This work proposes a classification-oriented fairness auditing framework that combines counterfactual reasoning with conventional fairness metrics. The introduced **Predictive Counterfactual Fairness (PreCoF)** metric enables systematic identification of bias-sensitive decision regions, improving the interpretability and transparency of automated hiring systems.

---

## 📖 Research Paper

**Title:**
> Classification of Bias in Automated Hiring Systems using Counterfactual Fairness

Conference:
> 5th International Conference on Technological Advancements in Computational Sciences (ICTACS 2026)

---

## 👨‍💻 Authors

**Nikhil Palyal**  
Chitkara University Institute of Engineering and Technology  
Punjab, India

**Taifa Ayoub Mir**  
Chitkara University Institute of Engineering and Technology  
Punjab, India

---

## 📄 License

This project is intended for academic and research purposes.

---

## ⭐ If you find this project useful, please consider giving it a star.
