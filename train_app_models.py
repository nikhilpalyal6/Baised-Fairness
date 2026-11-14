import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

os.makedirs('models', exist_ok=True)

print("1. Processing AI Fair Recruitment Dataset...")
df_ai = pd.read_csv('data/fair_recrutment_dataset_final.csv')
drop_cols = [c for c in df_ai.columns if 'id' in c.lower()]
df_ai = df_ai.drop(columns=drop_cols)
y_ai = df_ai['Hiring_Decision'].astype(int)
X_ai = df_ai.drop(columns=['Hiring_Decision'])

# Preprocessing for baseline AI model
num_cols_ai = X_ai.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_ai = X_ai.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_ai = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_ai),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols_ai)
    ]
)

# Preprocessing for Blinded / Fairness-Mitigated Model (Exclude Gender)
X_ai_fair = X_ai.drop(columns=['Gender'])
num_cols_ai_fair = X_ai_fair.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_ai_fair = X_ai_fair.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor_ai_fair = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_ai_fair),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols_ai_fair)
    ]
)

# Train on a stratified subset of 35k for rapid inference and lean deployment size
np.random.seed(42)
sample_idx = np.random.choice(len(df_ai), size=min(35000, len(df_ai)), replace=False)
X_train_ai = X_ai.iloc[sample_idx]
y_train_ai = y_ai.iloc[sample_idx]

# Model 1: Baseline AI Logistic Regression
pipe_ai_lr = Pipeline([('prep', preprocessor_ai), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
pipe_ai_lr.fit(X_train_ai, y_train_ai)
joblib.dump(pipe_ai_lr, 'models/ai_recruitment_lr.pkl')

# Model 2: Fairness-Mitigated AI Logistic Regression (Blinded to sensitive attributes)
pipe_ai_fair = Pipeline([('prep', preprocessor_ai_fair), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
pipe_ai_fair.fit(X_train_ai.drop(columns=['Gender']), y_train_ai)
joblib.dump(pipe_ai_fair, 'models/ai_recruitment_fair_lr.pkl')

# Model 3: Baseline Random Forest
pipe_ai_rf = Pipeline([('prep', preprocessor_ai), ('clf', RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1))])
pipe_ai_rf.fit(X_train_ai, y_train_ai)
joblib.dump(pipe_ai_rf, 'models/ai_recruitment_rf.pkl')

# Save metadata for UI sliders/dropdowns
metadata = {
    'ai_recruitment': {
        'numerical_features': {
            col: {
                'min': float(df_ai[col].min()),
                'max': float(df_ai[col].max()),
                'mean': float(df_ai[col].mean()),
                'median': float(df_ai[col].median()),
                'default': float(df_ai[col].median())
            } for col in num_cols_ai
        },
        'categorical_features': {
            col: sorted([str(x) for x in df_ai[col].dropna().unique().tolist()])
            for col in cat_cols_ai
        },
        'features_order': X_ai.columns.tolist()
    }
}

# 2. Adult Income Dataset Model
print("2. Processing Adult Income Dataset...")
if os.path.exists('data/adult.csv'):
    df_adult = pd.read_csv('data/adult.csv')
    df_adult.columns = [c.replace('.', '_').replace('-', '_') for c in df_adult.columns]
    y_adult = (df_adult['income'].astype(str).str.contains('>50K')).astype(int)
    X_adult = df_adult.drop(columns=['income'])
    if 'fnlwgt' in X_adult.columns:
        X_adult = X_adult.drop(columns=['fnlwgt'])
    
    num_cols_adult = X_adult.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_adult = X_adult.select_dtypes(exclude=[np.number]).columns.tolist()
    
    preprocessor_adult = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_adult),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols_adult)
        ]
    )
    
    sample_adult = np.random.choice(len(df_adult), size=min(25000, len(df_adult)), replace=False)
    pipe_adult_lr = Pipeline([('prep', preprocessor_adult), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
    pipe_adult_lr.fit(X_adult.iloc[sample_adult], y_adult.iloc[sample_adult])
    joblib.dump(pipe_adult_lr, 'models/adult_income_lr.pkl')
    
    metadata['adult_income'] = {
        'numerical_features': {
            col: {
                'min': float(df_adult[col].min()),
                'max': float(df_adult[col].max()),
                'mean': float(df_adult[col].mean()),
                'median': float(df_adult[col].median()),
                'default': float(df_adult[col].median())
            } for col in num_cols_adult
        },
        'categorical_features': {
            col: sorted([str(x) for x in df_adult[col].dropna().unique().tolist()])
            for col in cat_cols_adult
        },
        'features_order': X_adult.columns.tolist()
    }

with open('models/feature_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("All models and metadata successfully generated and saved into ./models/")
