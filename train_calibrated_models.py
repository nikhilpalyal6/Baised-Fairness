import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

os.makedirs('models', exist_ok=True)

# Generate a clean, realistic calibrated dataset based on the real dataset distribution
np.random.seed(42)
n_samples = 30000

roles = ["Software Engineer", "ML Engineer", "Data Analyst", "HR Executive", "Manager"]
role_weights = {"ML Engineer": -0.5, "Manager": -0.4, "Software Engineer": -0.2, "Data Analyst": 0.2, "HR Executive": 0.4}

genders = ["Female", "Male"]
educations = ["Bachelors", "Masters", "PhD", "Diploma", "High School"]
locations = ["Remote", "Onsite", "Hybrid"]

sample_roles = np.random.choice(roles, size=n_samples, p=[0.25, 0.20, 0.25, 0.15, 0.15])
sample_genders = np.random.choice(genders, size=n_samples, p=[0.45, 0.55])
sample_edus = np.random.choice(educations, size=n_samples, p=[0.40, 0.25, 0.05, 0.15, 0.15])
sample_locs = np.random.choice(locations, size=n_samples, p=[0.40, 0.40, 0.20])
sample_age = np.random.randint(21, 60, size=n_samples)
sample_exp = np.clip((sample_age - 21) // 2 + np.random.randint(-2, 3, size=n_samples), 0, 25)

# Competency scores
sample_tech = np.random.normal(55, 16, size=n_samples).clip(10, 100)
sample_skill = np.random.normal(55, 16, size=n_samples).clip(10, 100)
sample_interview = np.random.normal(55, 16, size=n_samples).clip(10, 100)
sample_apt = np.random.normal(55, 16, size=n_samples).clip(10, 100)
sample_comm = np.random.normal(55, 16, size=n_samples).clip(10, 100)

sample_certs = np.random.poisson(2, size=n_samples).clip(0, 10)
sample_prev_comp = np.clip(sample_exp // 3 + np.random.poisson(1, size=n_samples), 0, 8)
sample_salary = (40000 + sample_exp * 3500 + sample_tech * 400).clip(20000, 180000)

# Realistic latent hiring decision score:
avg_score = (sample_tech + sample_skill + sample_interview + sample_apt) / 4.0
z = -1.2 + ((avg_score - 55.0) / 10.0) * 1.8 + (sample_exp / 5.0) * 0.8
for i in range(n_samples):
    z[i] += role_weights.get(sample_roles[i], 0.0)

# Baseline labels (with historical gender bias: Male +0.8, Female -0.8)
z_baseline = z.copy()
z_baseline[sample_genders == "Male"] += 0.8
z_baseline[sample_genders == "Female"] -= 0.8
prob_baseline = 1.0 / (1.0 + np.exp(-z_baseline))
y_baseline = (np.random.random(n_samples) < prob_baseline).astype(int)

# Fair labels (gender-neutral)
prob_fair = 1.0 / (1.0 + np.exp(-z))
y_fair = (np.random.random(n_samples) < prob_fair).astype(int)

df_all = pd.DataFrame({
    "Gender": sample_genders,
    "Age": sample_age,
    "Education_Level": sample_edus,
    "Experience_Years": sample_exp,
    "Skill_Score": sample_skill,
    "Aptitude_Test_Score": sample_apt,
    "Technical_Test_Score": sample_tech,
    "Communication_Score": sample_comm,
    "Certifications_Count": sample_certs,
    "Previous_Companies": sample_prev_comp,
    "Interview_Score": sample_interview,
    "Location": sample_locs,
    "Job_Role_Applied": sample_roles,
    "Expected_Salary": sample_salary
})

num_cols = ["Age", "Experience_Years", "Skill_Score", "Aptitude_Test_Score", "Technical_Test_Score", "Communication_Score", "Certifications_Count", "Previous_Companies", "Interview_Score", "Expected_Salary"]
cat_cols = ["Gender", "Education_Level", "Location", "Job_Role_Applied"]

prep_baseline = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ]
)

cat_cols_fair = ["Education_Level", "Location", "Job_Role_Applied"]
prep_fair = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_fair)
    ]
)

pipe_baseline = Pipeline([("prep", prep_baseline), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
pipe_baseline.fit(df_all, y_baseline)

pipe_fair = Pipeline([("prep", prep_fair), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
pipe_fair.fit(df_all.drop(columns=["Gender"]), y_fair)

joblib.dump(pipe_baseline, "models/ai_recruitment_lr.pkl")
joblib.dump(pipe_fair, "models/ai_recruitment_fair_lr.pkl")
print("Models calibrated and saved successfully!")
