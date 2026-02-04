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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.utils import resample

# Define the path to the dataset
file_path = r'c:\Users\franc\Desktop\PhD\courses\AI Methods for Bioengineering Challenges\challenge\icu_challenge\Dataset_ICU_Barbieri_Mollura.csv'

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

# # Find and print highly correlated pairs
# print("Highly correlated attributes (correlation > 0.7 or < -0.7):")
# # Get the absolute value of the correlation matrix
# corr_abs = correlation_matrix.abs()
# # Get the upper triangle of the correlation matrix
# upper_tri = corr_abs.where(pd.np.triu(pd.np.ones(corr_abs.shape), k=1).astype(pd.np.bool))
# # Find index of feature columns with correlation greater than 0.7
# to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
# # Get the pairs of highly correlated features
# highly_correlated_pairs = []
# for col in to_drop_corr:
#     for row in upper_tri.index:
#         if upper_tri.loc[row, col] > 0.7:
#             highly_correlated_pairs.append((row, col, upper_tri.loc[row, col]))

# for pair in highly_correlated_pairs:
#     print(f"{pair[0]:<20} | {pair[1]:<20} | {pair[2]:.2f}")


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

X = df_cleaned_noNANs.drop(columns=['In-hospital_death', 'recordid'])
y = df_cleaned_noNANs['In-hospital_death']
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Downsample the training data ---
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data['In-hospital_death'] == 0]
minority = train_data[train_data['In-hospital_death'] == 1]
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
downsampled_train_data = pd.concat([majority_downsampled, minority])
downsampled_train_data = downsampled_train_data.sample(frac=1, random_state=42)
X_train_balanced = downsampled_train_data.drop(columns='In-hospital_death')
y_train_balanced = downsampled_train_data['In-hospital_death']

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)


# --- Model Training and Evaluation ---

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\n\n-----------------------------------")
    print(f"--- {model_name} ---")
    print(f"-----------------------------------")

    # Use scaled data for kNN and Logistic Regression
    if model_name in ["k-Nearest Neighbors", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else: # Use original data for tree-based models and Naive Bayes
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Store results
    results[model_name] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix ({model_name}):")
    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Not-Died', 'Predicted Died'], yticklabels=['Actual Not-Died', 'Actual Died'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Not-Died (0)', 'Died (1)'])
    print(f"\nClassification Report ({model_name}):")
    print(report)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy ({model_name}): {accuracy:.4f}")

    # ROC Curve data
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
