# 1. Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

import joblib

# 2. Load Dataset

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df['stroke'].value_counts())

# 3. Separate Features and Target

X = df.drop(columns=['stroke', 'id'])
y = df['stroke']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# 3. Separate Features and Target

X = df.drop(columns=['stroke', 'id'])
y = df['stroke']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

numerical_features = [
    'age',
    'hypertension',
    'heart_disease',
    'avg_glucose_level',
    'bmi'
]

categorical_features = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

# Numerical Preprocessing

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Preprocessing

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine Preprocessing Pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Complete ML Pipeline

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

# Model Training

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# 9. Model Predictions

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# 10. Model Evaluation

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

# Threshold Tuning

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")

    print("Precision:",
          precision_score(y_test, y_pred_threshold, zero_division=0))

    print("Recall:",
          recall_score(y_test, y_pred_threshold, zero_division=0))

    print("F1 Score:",
          f1_score(y_test, y_pred_threshold, zero_division=0))

# 12. Save Trained Model and Threshold

best_threshold = 0.7

model_artifact = {
    "model": model,
    "threshold": best_threshold
}

joblib.dump(model_artifact, "stroke_model.pkl")

print("\nModel saved successfully!")