import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Added for directory creation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, VotingClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge
# REQUIRED: Must be imported before IterativeImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, accuracy_score, recall_score, precision_score, precision_recall_curve, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# --- CONFIGURATION ---
FILE_PATH = 'Dataset_ICU_Barbieri_Mollura.csv'
RANDOM_STATE = 42
RESULTS_DIR = 'results' # Directory to save results

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- LOAD DATA ---
try:
    df = pd.read_csv(FILE_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{FILE_PATH}' not found.")
    exit()

# ==============================================================================
# 1. DATA PREPROCESSING & CLINICAL FILTERING
# ==============================================================================
print("\n[1] Applying Clinical Filter (First 30 Minutes)...")

# A. Define Variables to Keep
demographics_and_units = ['Age', 'Gender', 'Height', 'Weight', 'CCU', 'CSRU', 'SICU', 'In-hospital_death']
all_first_cols = [c for c in df.columns if c.endswith('_first')]

# B. Define Laboratory Keywords to EXCLUDE
lab_keywords = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', # Chemistry
    'HCT', 'K', 'Lactate', 'Mg', 'Na', # Electrolytes
    'Platelets', 'TroponinI', 'TroponinT', 'WBC', 'Glucose', 'PaO2', 'PaCO2', 'pH', 'FiO2', 'HCO3' # Gases/Heme
]

vitals_to_keep = []
for col in all_first_cols:
    if not any(lab_key in col for lab_key in lab_keywords):
        vitals_to_keep.append(col)

# C. Construct Final Dataset
final_cols = demographics_and_units + vitals_to_keep
df_triage = df[final_cols].copy()
print(f"Features Retained: {df_triage.shape[1]-1} (Excluding Target)")

# D. Feature Engineering (GCS)
# We use dummy_na=False because we want the Imputer to FILL these later.
if 'GCS_first' in df_triage.columns:
    df_triage = pd.get_dummies(df_triage, columns=['GCS_first'], drop_first=True, dummy_na=False)

# E. Correlation Analysis & Pruning
print("\n[2] Computing Correlation Matrix...")
X_temp = df_triage.drop(columns=['In-hospital_death'])
corr_matrix = X_temp.corr().abs()

# PLOT 1: Correlation Matrix (Before Pruning)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix (Triage Variables - Before Pruning)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix_pre_pruning.png')) # Save plot
plt.show()

# Pruning > 0.95
print("Pruning High Correlations (>0.95)...")
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_triage = df_triage.drop(columns=to_drop)
print(f"Dropped {len(to_drop)} redundant features: {to_drop}")

# F. Train/Test Split
X = df_triage.drop(columns=['In-hospital_death'])
y = df_triage['In-hospital_death']

# Clean split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
print(f"Train Size: {X_train.shape[0]} | Test Size: {X_test.shape[0]}")


# ==============================================================================
# 2. PHASE 1: BASELINE MODEL SELECTION & TUNING
# Strategy: Median Imputation + Missing Indicators
# ==============================================================================
print("\n" + "="*60)
print(" PHASE 1: BASELINE MODEL SELECTION (Median + Missing Flags)")
print("="*60)

# Define Model Pipelines (All use SimpleImputer(add_indicator=True))
model_configs = {
    "Decision Tree": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
        ]),
        "params": {
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 5]
        }
    },
    "Random Forest": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
        ]),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 4]
        }
    },
    "Hist Gradient Boosting": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('classifier', HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
        ]),
        "params": {
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [None, 10, 20]
        }
    },
    "Logistic Regression": {
        "model": Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)), 
            ('scaler', StandardScaler()), 
            ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, class_weight='balanced'))
        ]),
        "params": {
            "classifier__C": np.logspace(-3, 2, 10),
            "classifier__solver": ['liblinear', 'saga'] 
        }
    }
}

best_auc = 0
best_model_name = ""
best_model_estimator = None 
phase1_metrics_data = [] # To store dicts for DataFrame
phase1_roc_data = {}

# Hyperparameter Tuning Loop
for name, config in model_configs.items():
    print(f"Tuning {name}...")
    rs = RandomizedSearchCV(
        config["model"], config["params"], 
        n_iter=10, cv=5, scoring='roc_auc', 
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    rs.fit(X_train, y_train)
    
    # Store results
    y_prob = rs.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = rs.best_estimator_.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_prob)
    acc_score = accuracy_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    
    # Save Metrics for Plotting
    phase1_metrics_data.append({'Model': name, 'Metric': 'Accuracy', 'Value': acc_score})
    phase1_metrics_data.append({'Model': name, 'Metric': 'Precision', 'Value': prec_score})
    phase1_metrics_data.append({'Model': name, 'Metric': 'Recall', 'Value': rec_score})
    
    # Save ROC Data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    phase1_roc_data[name] = (fpr, tpr, auc_score)
    
    print(f"  -> Best AUC: {auc_score:.4f} | Recall: {rec_score:.4f} | Accuracy: {acc_score:.4f}")
    
    if auc_score > best_auc:
        best_auc = auc_score
        best_model_name = name
        best_model_estimator = rs.best_estimator_

print(f"\n🏆 CHAMPION MODEL (BASELINE): {best_model_name} (AUC={best_auc:.4f})")

# Save Phase 1 Metrics to CSV
metrics_df = pd.DataFrame(phase1_metrics_data)
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'phase1_metrics.csv'), index=False)

# PLOT 2: Phase 1 Metrics Comparison (Bar Chart)
plt.figure(figsize=(12, 6))
sns.barplot(data=metrics_df, x='Model', y='Value', hue='Metric', palette='viridis')
plt.title('Phase 1: Model Performance Comparison (Default Threshold)')
plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'phase1_metrics_comparison.png')) # Save plot
plt.show()

# PLOT 3: Phase 1 ROC Curves Comparison
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'orange', 'purple']
for (name, (fpr, tpr, roc_auc)), color in zip(phase1_roc_data.items(), colors):
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', lw=2, color=color)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Phase 1: ROC Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'phase1_roc_curves.png')) # Save plot
plt.show()

# Feature Importance
print(f"\n[Analysis] Calculating Feature Importance for {best_model_name}...")
r = permutation_importance(
    best_model_estimator, X_test, y_test,
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring='roc_auc'
)
sorted_idx = r.importances_mean.argsort()[::-1][:15]

# Save Feature Importance to CSV
fi_df = pd.DataFrame({
    'Feature': X_test.columns[sorted_idx],
    'Importance Mean': r.importances_mean[sorted_idx],
    'Importance Std': r.importances_std[sorted_idx]
})
fi_df.to_csv(os.path.join(RESULTS_DIR, f'feature_importance_{best_model_name.replace(" ", "_")}.csv'), index=False)

plt.figure(figsize=(10, 6))
plt.boxplot(r.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
plt.title(f"Permutation Importance ({best_model_name})")
plt.xlabel("Decrease in ROC AUC Score")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'feature_importance_{best_model_name.replace(" ", "_")}.png')) # Save plot
plt.show()

# ==============================================================================
# 3. PHASE 2: ADVANCED IMPUTATION BENCHMARK
# Strategy: Compare different regressors for IterativeImputer with Best Classifier
# Metrics: Reconstruction Error (R2, MAE) AND Classification Performance
# ==============================================================================
print("\n" + "="*60)
print(f" PHASE 2: IMPUTATION STRATEGY BENCHMARK")
print(f" Engine: {best_model_name}")
print("="*60)

# Define Imputation Models to Compare
IMPUTATION_MODELS = [
    ("BayesianRidge", None),  # Default
    ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
    ("RandomForest", RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)),
    ("ExtraTrees", ExtraTreesRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)),
    ("KNN", KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)),
]

imputation_results = {}
imputation_quality_data = [] # To store R2/MAE

# 1. Validation Logic for Imputation Quality
# We mask known values in the Test set to calculate MAE/R2 of the imputation itself
X_test_val = X_test.copy().astype(float) # Cast to float to avoid object errors
# Create mask for 10% of observed values
np.random.seed(RANDOM_STATE)
mask = np.random.choice([True, False], size=X_test_val.shape, p=[0.1, 0.9])
# Only mask values that are NOT already NaN
mask = mask & (~np.isnan(X_test_val.values))
X_test_masked = X_test_val.mask(mask)
ground_truth = X_test_val.values[mask]

# Baseline Stats (Classification)
def get_metrics_tuned(pipeline, X, y):
    y_prob = pipeline.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    f2_scores = np.divide((1 + 4) * precision * recall, (4 * precision) + recall, out=np.zeros_like(precision), where=((4 * precision) + recall)!=0)
    ix = np.argmax(f2_scores[:-1])
    thresh = thresholds[ix]
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "auc": roc_auc_score(y, y_prob),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "y_prob": y_prob
    }

base_stats = get_metrics_tuned(best_model_estimator, X_test, y_test)
imputation_results["Baseline (Median)"] = base_stats
# Note: Baseline Median imputation MAE/R2 is not calculated via iterative logic, but we can impute median for plot
imp_median = SimpleImputer(strategy='median')
imp_median.fit(X_train)
X_test_median = imp_median.transform(X_test_masked)
preds_median = X_test_median[mask]
mae_median = mean_absolute_error(ground_truth, preds_median)
r2_median = r2_score(ground_truth, preds_median)
imputation_quality_data.append({'Model': 'Baseline (Median)', 'Metric': 'MAE', 'Value': mae_median})
imputation_quality_data.append({'Model': 'Baseline (Median)', 'Metric': 'R2 Score', 'Value': r2_median})

print(f"Baseline (Median): AUC={base_stats['auc']:.4f} | R2_Imp={r2_median:.4f} | MAE_Imp={mae_median:.4f}")

# Loop through Advanced Imputers
for imp_name, imp_estimator in IMPUTATION_MODELS:
    print(f"Evaluating Imputer: {imp_name}...")
    
    # A. Train Imputer
    advanced_imputer = IterativeImputer(
        estimator=imp_estimator, max_iter=10, random_state=RANDOM_STATE, initial_strategy='median'
    )
    advanced_imputer.fit(X_train)
    
    # B. Calculate Imputation Quality (R2/MAE) on Masked Data
    X_test_imputed_val = advanced_imputer.transform(X_test_masked)
    preds_val = X_test_imputed_val[mask]
    mae_val = mean_absolute_error(ground_truth, preds_val)
    r2_val = r2_score(ground_truth, preds_val)
    
    imputation_quality_data.append({'Model': imp_name, 'Metric': 'MAE', 'Value': mae_val})
    imputation_quality_data.append({'Model': imp_name, 'Metric': 'R2 Score', 'Value': r2_val})
    print(f"  -> Imputation Quality: R2={r2_val:.4f} | MAE={mae_val:.4f}")

    # C. Train & Evaluate Downstream Classifier
    # Reconstruct Pipeline
    steps = []
    steps.append(('imputer', advanced_imputer)) # Use the fitted imputer (or re-fit in pipeline)
    if 'scaler' in best_model_estimator.named_steps:
        steps.append(('scaler', StandardScaler()))
    classifier_instance = best_model_estimator.named_steps['classifier']
    steps.append(('classifier', classifier_instance))
    
    pipeline_adv = Pipeline(steps)
    
    try:
        # Fit classification pipeline (refits imputer on X_train naturally)
        pipeline_adv.fit(X_train, y_train)
        stats = get_metrics_tuned(pipeline_adv, X_test, y_test)
        imputation_results[imp_name] = stats
        print(f"  -> Classification: AUC={stats['auc']:.4f} | Recall={stats['recall']:.4f}")
        
    except Exception as e:
        print(f"  -> Failed: {e}")

# Save Imputation Quality Metrics to CSV
imp_qual_df = pd.DataFrame(imputation_quality_data)
imp_qual_df.to_csv(os.path.join(RESULTS_DIR, 'imputation_quality_metrics.csv'), index=False)

# PLOT 4: Imputation Quality Comparison (R2 and MAE)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MAE Plot
sns.barplot(data=imp_qual_df[imp_qual_df['Metric']=='MAE'], x='Model', y='Value', ax=axes[0], palette='magma')
axes[0].set_title('Imputation Error (MAE) - Lower is Better')
axes[0].tick_params(axis='x', rotation=45)

# R2 Plot
sns.barplot(data=imp_qual_df[imp_qual_df['Metric']=='R2 Score'], x='Model', y='Value', ax=axes[1], palette='viridis')
axes[1].set_title('Imputation Accuracy (R2 Score) - Higher is Better')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'imputation_quality_comparison.png')) # Save plot
plt.show()

# ==============================================================================
# 4. FINAL CLASSIFICATION COMPARISON
# ==============================================================================
print("\n" + "="*80)
print(f" FINAL RESULTS: IMPUTATION STRATEGY CLASSIFICATION IMPACT ({best_model_name})")
print("="*80)
print(f"{'Strategy':<20} | {'AUC':<10} | {'Recall (F2)':<15} | {'Accuracy':<10} | {'Diff vs Base'}")
print("-" * 80)

final_results_data = [] # For CSV

for name, res in imputation_results.items():
    diff = res['auc'] - base_stats['auc']
    print(f"{name:<20} | {res['auc']:<10.4f} | {res['recall']:<15.4f} | {res['accuracy']:<10.4f} | {diff:+.4f}")
    final_results_data.append({
        'Strategy': name,
        'AUC': res['auc'],
        'Recall (F2)': res['recall'],
        'Accuracy': res['accuracy'],
        'Diff vs Base': diff
    })

# Save Final Classification Results to CSV
final_results_df = pd.DataFrame(final_results_data)
final_results_df.to_csv(os.path.join(RESULTS_DIR, 'final_classification_comparison.csv'), index=False)

# PLOT 5: Classification ROC Comparison
plt.figure(figsize=(12, 10))
colors = ['black', 'blue', 'green', 'purple', 'red', 'orange']
for (name, res), color in zip(imputation_results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})', linewidth=2, color=color)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Impact of Imputation Strategies on Mortality Prediction ({best_model_name})')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'final_roc_comparison.png')) # Save plot
plt.show()