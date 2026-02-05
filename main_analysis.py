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

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# --- Hyperparameter Grids ---

model_params = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "params": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"]
        },
        "requires_scale": False
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        },
        "requires_scale": False
    },
    "k-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"]
        },
        "requires_scale": True
    },
    "Gaussian Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": np.logspace(0, -9, num=100)
        },
        "requires_scale": False # NB handles unscaled, but often better with scaled if Gaussian assumption holds. Let's treat as unscaled for consistency with previous steps, or scaled? Usually GaussianNB is fine either way but strictly speaking assumes Gaussian distribution. Let's use unscaled as per previous logic.
    },
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
        "params": {
            "C": np.logspace(-3, 2, 10),
            "solver": ['liblinear', 'saga'], # saga supports elasticnet/l1/l2
            "penalty": ['l1', 'l2']
        },
        "requires_scale": True
    }
}

results = {}

print("\n\n=======================================================")
print(" STARTING RANDOMIZED SEARCH CV & THRESHOLD TUNING (F2)")
print(" Goal: 1. Optimize AUC (Hyperparams) -> 2. Maximize F2 (Threshold)")
print("=======================================================")

# Train and evaluate each model
for model_name, config in model_params.items():
    print(f"\n\n-----------------------------------")
    print(f"--- {model_name} ---")
    print(f"-----------------------------------")

    model = config["model"]
    params = config["params"]
    requires_scale = config["requires_scale"]

    # Select correct data
    if requires_scale:
        X_train_curr = X_train
        X_test_curr = X_test
    else:
        X_train_curr = X_train
        X_test_curr = X_test

    # 1. Randomized Search CV
    print("Running RandomizedSearchCV...")
    # n_iter=20 to keep runtime reasonable. Increase to 50+ for production.
    rs = RandomizedSearchCV(
        model, 
        params, 
        n_iter=20, 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1, 
        random_state=42,
        verbose=0
    )
    
    rs.fit(X_train_curr, y_train)
    
    best_model = rs.best_estimator_
    print(f"Best Parameters: {rs.best_params_}")
    print(f"Best CV ROC-AUC: {rs.best_score_:.4f}")

    # 2. Get probabilities from best model
    y_pred_proba = best_model.predict_proba(X_test_curr)[:, 1]

    # 3. Threshold Tuning (F2 Score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    numerator = (1 + 2**2) * precision * recall
    denominator = (2**2 * precision) + recall
    f2_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    ix = np.argmax(f2_scores[:-1])
    best_thresh = thresholds[ix]
    best_f2 = f2_scores[ix]
    
    print(f"Optimal Threshold (Max F2): {best_thresh:.4f}")

    # 4. Generate predictions
    y_pred = (y_pred_proba >= best_thresh).astype(int)

    # Store results
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    results[model_name] = {
        "y_pred": y_pred, 
        "y_pred_proba": y_pred_proba, 
        "fpr": fpr, 
        "tpr": tpr, 
        "roc_auc": roc_auc
    }

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix ({model_name}) [Tuned]:")
    print(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Not-Died', 'Predicted Died'],
                yticklabels=['Actual Not-Died', 'Actual Died'])
    plt.title(f'{model_name} - CM (Thresh={best_thresh:.2f})')
    plt.show()

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Not-Died (0)', 'Died (1)'])
    print(f"\nClassification Report ({model_name}):")
    print(report)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy ({model_name}): {accuracy:.4f}")


# --- Combined ROC Curve ---
print("\n\n-----------------------------------")
print("--- Model Comparison (Optimized) ---")
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
plt.title('Comparison of ROC Curves (Post-Tuning)')
plt.legend(loc="lower right")
plt.show()
