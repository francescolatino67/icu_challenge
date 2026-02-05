#-*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: venv (3.10.11)
#     language: python
#     name: python3
# ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.utils import resample

# Define the path to the dataset
file_path = r'..\icu_challenge\Dataset_ICU_Barbieri_Mollura.csv'

# Load the dataset
df = pd.read_csv(file_path)

# 1. Understand the dataset structure
print("--- First 5 rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Shape of the dataset ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

# 2. Find correlations between variables
print("\n--- Correlation Analysis ---")
plt.figure(figsize=(20, 16))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of ICU Dataset Variables')
plt.show()

# --- Data Preprocessing ---
columns_to_drop = [
    'NIMAP_first', 'MAP_last', 'NIMAP_last', 'NIMAP_lowest', 'NIMAP_highest', 'MAP_median', 'NIMAP_median',
    'SysABP_median', 'GCS_last', 'GCS_median', 'SaO2_last', 'SaO2_median', 'DiasABP_median', 'HR_last', 'HR_median',
    'NISysABP_median', 'Temp_median', 'AST_first', 'AST_last', 'Weight', 'Weight_last', 'ALP_last', 'Albumin_last',
    'BUN_last', 'Bilirubin_last', 'Cholesterol_last', 'Creatinine_last', 'Platelets_last', 'TroponinI_last',
    'TroponinT_last', 'MechVentLast8Hour',
]
columns_to_drop = sorted(list(set(columns_to_drop)))
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

def remove_high_missing_cols(df, threshold=50):
    total = len(df)
    cols_to_drop = [col for col in df.columns if (df[col].isnull().sum() / total * 100) > threshold]
    return df.drop(columns=cols_to_drop)

df_cleaned_noNANs = remove_high_missing_cols(df_cleaned)

# Define X and y
if 'recordid' in df_cleaned_noNANs.columns:
    X = df_cleaned_noNANs.drop(columns=['In-hospital_death', 'recordid'])
else:
    X = df_cleaned_noNANs.drop(columns=['In-hospital_death'])

y = df_cleaned_noNANs['In-hospital_death']

# Impute missing values with median
X = X.fillna(X.median())

# Feature scaling before splitting
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)
X = pd.DataFrame(X_scaled_full, columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# # --- Feature Scaling ---
# scaler = StandardScaler()
# # Fit on the FULL training set
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# --- Model Training and Evaluation ---

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "k-Nearest Neighbors": KNeighborsClassifier(), 
    "Gaussian Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
}

results = {}

print("\n\n=======================================================")
print(" STARTING MODEL EVALUATION WITH THRESHOLD TUNING (F2)")
print(" Goal: Maximize Recall (Sensitivity) to reduce False Negatives")
print("=======================================================")

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\n\n-----------------------------------")
    print(f"--- {model_name} ---")
    print(f"-----------------------------------")

    # Get probabilities (positive class)
    if model_name in ["k-Nearest Neighbors", "Logistic Regression"]:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else: # Tree-based models and Naive Bayes
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # --- Threshold Tuning Logic ---
    # We calculate Precision, Recall, and Thresholds for the curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F2-Score: Weights Recall higher than Precision (Beta=2)
    numerator = (1 + 2**2) * precision * recall
    denominator = (2**2 * precision) + recall
    # Handle division by zero
    f2_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # Locate the index of the largest F2 score
    # Note: precision_recall_curve appends 1.0 to precision and 0.0 to recall as the last element.
    # thresholds array is 1 element shorter than precision/recall arrays.
    # We ignore the last element of f2_scores for finding the max index relative to thresholds.
    ix = np.argmax(f2_scores[:-1])
    best_thresh = thresholds[ix]
    best_f2 = f2_scores[ix]
    
    print(f"Optimal Threshold (Max F2-Score): {best_thresh:.4f}")
    print(f"Expected F2-Score at this threshold: {best_f2:.4f}")

    # Generate predictions using the NEW customized threshold
    y_pred = (y_pred_proba >= best_thresh).astype(int)

    # Store results
    results[model_name] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba, "roc_auc": 0} # roc_auc calc below

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix ({model_name}) [Tuned]:")
    print(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Not-Died', 'Predicted Died'],
                yticklabels=['Actual Not-Died', 'Actual Died'])
    plt.title(f'{model_name} - Confusion Matrix (Threshold={best_thresh:.2f})')
    plt.show()

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Not-Died (0)', 'Died (1)'])
    print(f"\nClassification Report ({model_name}):")
    print(report)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy ({model_name}): {accuracy:.4f}")

    # ROC Curve data (Standard calculation, threshold invariant)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    results[model_name]["fpr"] = fpr
    results[model_name]["tpr"] = tpr
    results[model_name]["roc_auc"] = roc_auc


# --- Combined ROC Curve ---
print("\n\n-----------------------------------")
print("--- Model Comparison ---")
print("-----------------------------------")
plt.figure(figsize=(12, 10))
colors = {'Decision Tree': 'darkorange', 'Random Forest': 'green', 'k-Nearest Neighbors': 'purple', 'Gaussian Naive Bayes': 'red', 'Logistic Regression': 'blue'}

for model_name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], color=colors[model_name], lw=2, label=f'{model_name} ROC (area = {res["roc_auc"]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves')
plt.legend(loc="lower right")
plt.show()
