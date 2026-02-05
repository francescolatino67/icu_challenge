import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.inspection import permutation_importance

# Define the path to the dataset
file_path = r'..\icu_challenge\Dataset_ICU_Barbieri_Mollura.csv'

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file name and path.")
    exit()

# --- Data Preprocessing: Clinical "First 30 Minutes" Filtering ---
# GOAL: Retain ONLY variables available immediately at admission (Triage/Monitor).
# EXCLUDE: Laboratory results (require time to process).

print("\n--- Applying Clinical Filter (First 30 Minutes: Monitor + Demographics ONLY) ---")

# 1. Define Immediate Variables (Demographics + Unit)
demographics_and_units = ['Age', 'Gender', 'Height', 'Weight', 'CCU', 'CSRU', 'SICU', 'In-hospital_death', 'recordid']

# 2. Identify potential admission columns ('_first')
all_first_cols = [c for c in df.columns if c.endswith('_first')]

# 3. Define Laboratory Keywords to EXCLUDE
# These measurements typically take > 30 mins (Blood work, Gases, Chemistries)
lab_keywords = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', # Chemistry
    'HCT', 'K', 'Lactate', 'Mg', 'Na', # Blood Gas / Electrolytes
    'Platelets', 'TroponinI', 'TroponinT', 'WBC', 'Glucose' # Heme / Cardiac / Others
]

# Split '_first' columns into Vitals (Keep) and Labs (Drop)
vitals_to_keep = []
labs_dropped = []

for col in all_first_cols:
    is_lab = False
    for lab_key in lab_keywords:
        if lab_key in col:
            is_lab = True
            break
    
    if is_lab:
        labs_dropped.append(col)
    else:
        vitals_to_keep.append(col)

print(f"\nExcluding {len(labs_dropped)} laboratory features (unlikely available in 30 mins):")
print(labs_dropped)
print(f"\nRetaining {len(vitals_to_keep)} bedside/monitor features:")
print(vitals_to_keep)

# Combine Demographics + Vitals
final_cols = demographics_and_units + vitals_to_keep
df_immediate = df[final_cols].copy()

print(f"\nFinal Feature Count: {df_immediate.shape[1]}")

# --- Feature Engineering: One-Hot Encoding Clinical Scores ---
# CLINICAL NOTE: Only GCS_first remains as a score available in 30 mins.
# We use dummy_na=True to preserve 'NaN' information.

cols_to_encode = ['GCS_first']
cols_to_encode = [c for c in cols_to_encode if c in df_immediate.columns]

if cols_to_encode:
    print(f"\nOne-Hot Encoding with NaN preservation: {cols_to_encode}")
    df_encoded = pd.get_dummies(df_immediate, columns=cols_to_encode, drop_first=True, dummy_na=True)
else:
    df_encoded = df_immediate.copy()

# --- Data Preprocessing: Objective Correlation Pruning ---
print("\nComputing correlation matrix for admission variables...")
numeric_df = df_encoded.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

# Identify Highly Correlated Pairs (> 0.95)
threshold = 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop_corr = [column for column in upper.columns if any(upper[column] > threshold)]

print(f"\nHighly correlated columns detected (> {threshold}): {len(cols_to_drop_corr)}")
print(f"Dropping: {cols_to_drop_corr}")

df_cleaned = df_encoded.drop(columns=cols_to_drop_corr, errors='ignore')

# --- Remove columns with excessive missingness ---
def remove_high_missing_cols(df, threshold=95): 
    total = len(df)
    cols_to_drop = [col for col in df.columns if (df[col].isnull().sum() / total * 100) > threshold]
    if cols_to_drop:
        print(f"Dropping columns with >{threshold}% missing values: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)

df_cleaned_noNANs = remove_high_missing_cols(df_cleaned)

# Define X and y
if 'recordid' in df_cleaned_noNANs.columns:
    X = df_cleaned_noNANs.drop(columns=['In-hospital_death', 'recordid'])
else:
    X = df_cleaned_noNANs.drop(columns=['In-hospital_death'])

y = df_cleaned_noNANs['In-hospital_death']

# Train-Test Split (Preserving NaNs)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Hyperparameter Grids & Model Pipelines ---

model_params = {
    "Decision Tree": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        "params": {
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 10],
            "classifier__min_samples_leaf": [1, 5]
        }
    },
    "Random Forest": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 4]
        }
    },
    "k-Nearest Neighbors": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()), 
            ('classifier', KNeighborsClassifier())
        ]),
        "params": {
            "classifier__n_neighbors": [5, 9, 15],
            "classifier__weights": ['uniform', 'distance']
        }
    },
    "Hist Gradient Boosting": {
        "model": HistGradientBoostingClassifier(random_state=42, class_weight='balanced'),
        "params": {
            "learning_rate": [0.01, 0.1],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [20, 40]
        }
    },
    "Logistic Regression": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)), 
            ('scaler', StandardScaler()), 
            ('classifier', LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'))
        ]),
        "params": {
            "classifier__C": np.logspace(-3, 2, 10),
            "classifier__penalty": ['l1', 'l2'],
            "classifier__solver": ['liblinear', 'saga'] 
        }
    }
}

results = {}
best_estimators = {} 
best_auc = 0
best_model_name = ""
best_model_obj = None

print("\n\n=======================================================")
print(" STARTING MODEL TRAINING (TRIAGE MODE)")
print(" Strategy: Native Support (HGB) vs Missing Indicator (RF, DT, LR, KNN)")
print("=======================================================")

# Train and evaluate individual models
for model_name, config in model_params.items():
    print(f"\n\n-----------------------------------")
    print(f"--- {model_name} ---")
    print(f"-----------------------------------")

    model = config["model"]
    params = config["params"]

    print("Running RandomizedSearchCV...")
    rs = RandomizedSearchCV(
        model, 
        params, 
        n_iter=10, 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1, 
        random_state=42,
        verbose=0
    )
    
    rs.fit(X_train, y_train)
    
    best_model = rs.best_estimator_
    best_estimators[model_name] = best_model
    
    print(f"Best Parameters: {rs.best_params_}")
    print(f"Best CV ROC-AUC: {rs.best_score_:.4f}")

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Threshold Tuning (F2 Score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    numerator = (1 + 2**2) * precision * recall
    denominator = (2**2 * precision) + recall
    f2_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    ix = np.argmax(f2_scores[:-1])
    best_thresh = thresholds[ix]
    
    print(f"Optimal Threshold (Max F2): {best_thresh:.4f}")

    y_pred = (y_pred_proba >= best_thresh).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Track Best Model
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model_name = model_name
        best_model_obj = best_model

    results[model_name] = {
        "y_pred": y_pred, 
        "y_pred_proba": y_pred_proba, 
        "fpr": fpr, 
        "tpr": tpr, 
        "roc_auc": roc_auc
    }

# --- IMPLEMENTING SOFT VOTING ENSEMBLE ---
if "Random Forest" in best_estimators and "Logistic Regression" in best_estimators:
    print("\n\n-----------------------------------")
    print("--- Voting Ensemble (RF + LR) ---")
    print("-----------------------------------")
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', best_estimators['Random Forest']), 
            ('lr', best_estimators['Logistic Regression'])
        ],
        voting='soft'
    )
    
    ensemble_model.fit(X_train, y_train)
    
    y_pred_proba_ens = ensemble_model.predict_proba(X_test)[:, 1]
    
    # Tune Threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_ens)
    numerator = (1 + 2**2) * precision * recall
    denominator = (2**2 * precision) + recall
    f2_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    ix = np.argmax(f2_scores[:-1])
    best_thresh_ens = thresholds[ix]
    
    print(f"Optimal Ensemble Threshold (Max F2): {best_thresh_ens:.4f}")
    
    y_pred_ens = (y_pred_proba_ens >= best_thresh_ens).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_ens)
    roc_auc = auc(fpr, tpr)
    
    # Check if Ensemble is best
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model_name = "Ensemble (RF+LR)"
        best_model_obj = ensemble_model

    results["Ensemble (RF+LR)"] = {
        "y_pred": y_pred_ens,
        "y_pred_proba": y_pred_proba_ens,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }
    
    conf_matrix = confusion_matrix(y_test, y_pred_ens)
    print("Confusion Matrix (Ensemble) [Tuned]:")
    print(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Not-Died', 'Predicted Died'],
                yticklabels=['Actual Not-Died', 'Actual Died'])
    plt.title(f'Ensemble (RF+LR) - CM (Thresh={best_thresh_ens:.2f})')
    plt.show()

    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, y_pred_ens, target_names=['Not-Died (0)', 'Died (1)']))
    
    print(f"\nAccuracy (Ensemble): {accuracy_score(y_test, y_pred_ens):.4f}")


# --- Combined ROC Curve ---
print("\n\n-----------------------------------")
print("--- Model Comparison (Optimized) ---")
print("-----------------------------------")
plt.figure(figsize=(12, 10))
colors = {
    'Decision Tree': 'darkorange',
    'Random Forest': 'green', 
    'k-Nearest Neighbors': 'purple',
    'Hist Gradient Boosting': 'teal',
    'Logistic Regression': 'blue',
    'Ensemble (RF+LR)': 'black'
}

for model_name, res in results.items():
    color = colors.get(model_name, 'gray')
    plt.plot(res["fpr"], res["tpr"], color=color, lw=2, label=f'{model_name} ROC (area = {res["roc_auc"]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves (All Models NaN-Aware)')
plt.legend(loc="lower right")
plt.show()

# --- FEATURE IMPORTANCE ANALYSIS (BEST MODEL) ---
print(f"\n\n=======================================================")
print(f" BEST MODEL IDENTIFIED: {best_model_name}")
print(f" ROC AUC: {best_auc:.4f}")
print("=======================================================")

print(f"\nCalculating Permutation Importance for {best_model_name}...")
print("This explains 'weights' by shuffling each feature and measuring AUC drop.")

# Permutation importance works for ALL models (Pipeline, Ensemble, etc.)
# It uses the Test Set to ensure we are measuring generalization importance.
r = permutation_importance(
    best_model_obj, 
    X_test, 
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance Mean': r.importances_mean,
    'Importance Std': r.importances_std
})

# Sort by importance
importances_df = importances_df.sort_values(by='Importance Mean', ascending=False).head(20)

print("\nTop 10 Important Features:")
print(importances_df[['Feature', 'Importance Mean']].head(10))

# Plotting Boxplot
plt.figure(figsize=(12, 8))
# Re-create raw data structure for boxplot
top_indices = r.importances_mean.argsort()[::-1][:20]
top_features = X.columns[top_indices]
top_importances = r.importances[top_indices].T

plt.boxplot(top_importances, vert=False, labels=top_features)
plt.title(f'Permutation Feature Importance ({best_model_name})\n(Metric: Decrease in ROC AUC)')
plt.xlabel('Decrease in ROC AUC score')
plt.tight_layout()
plt.show()