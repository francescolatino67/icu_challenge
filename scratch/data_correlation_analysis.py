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

# Define the path to the dataset
file_path = r'c:\Users\franc\Desktop\PhD\courses\AI Methods for Bioengineering Challenges\challenge\icu_challenge\Dataset_ICU_Barbieri_Mollura.csv'

# Load the dataset
df = pd.read_csv(file_path)

# 1. Understand the dataset structure
print("--- First 5 rows ---")
print(df.head())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Shape of the dataset ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

# 2. Find correlations between variables
plt.figure(figsize=(20, 16))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of ICU Dataset Variables')
plt.show()

# Find and print highly correlated pairs
print("Highly correlated attributes (correlation > 0.7 or < -0.7):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            print(f"{correlation_matrix.columns[i]:<20} | {correlation_matrix.columns[j]:<20} | {correlation_matrix.iloc[i, j]:.2f}")

# ---
# Step 3: Remove Highly Correlated Variables ---
# Based on the analysis above, we will remove variables to reduce redundancy (multicollinearity).
# The strategy is to keep the most clinically relevant or fundamental variable in a correlated group.

# Define the list of columns to remove based on our rules:
# - Drop MAP/NIMAP in favor of Systolic/Diastolic.
# - Drop median/last summary stats in favor of first/lowest/highest.
# - Drop AST in favor of the more liver-specific ALT.
# - Drop redundant Weight columns.
# - Drop '_last' lab values in favor of '_first' (baseline).
# - Drop redundant ventilation metric.

columns_to_drop = [
    # Redundant Blood Pressure (keeping Systolic/Diastolic)
    'NIMAP_first', 'MAP_last', 'NIMAP_last', 'NIMAP_lowest', 'NIMAP_highest', 'MAP_median', 'NIMAP_median',
    'SysABP_median', # Correlated with MAP_median

    # Redundant Summary Statistics (keeping _first, _lowest, _highest)
    'GCS_last', 'GCS_median',
    'SaO2_last', 'SaO2_median',
    'DiasABP_median',
    'HR_last', 'HR_median',
    'NISysABP_median',
    'Temp_median', # from 'Temp_median' vs 'Temp_highest'

    # Redundant Clinical Concepts
    'AST_first', 'AST_last', # Keeping ALT
    'Weight', 'Weight_last', # Keeping Weight_first

    # Redundant Lab Tests (keeping _first)
    'ALP_last',
    'Albumin_last',
    'BUN_last',
    'Bilirubin_last',
    'Cholesterol_last', # Also has 1.0 correlation
    'Creatinine_last',
    'Platelets_last',
    'TroponinI_last',
    'TroponinT_last',

    # Redundant Ventilation
    'MechVentLast8Hour', # Keeping MechVentDuration
]

# Remove duplicates from the list
columns_to_drop = sorted(list(set(columns_to_drop)))

# Make sure the dataframe 'df' is available from previous cells
if 'df' in locals() or 'df' in globals():
    # Drop the columns
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    print("Based on clinical significance and high correlation, the following variables have been removed:")
    for col in columns_to_drop:
        print(f"- {col}")

    print(f"Original number of columns: {df.shape[1]}")
    print(f"Number of columns removed: {len(columns_to_drop)}")
    print(f"New number of columns: {df_cleaned.shape[1]}")

    # Display the first few rows of the cleaned dataframe
    # df_cleaned.head() # This will be the output of the cell
else:
    print("DataFrame 'df' not found. Please run the cell that loads the data first.")
