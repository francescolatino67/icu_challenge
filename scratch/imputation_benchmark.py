"""
Virtual Sensor Benchmark - Random Sparse Validation
====================================================
Tests the imputer's ability to adapt to ANY combination of inputs.

Use case: Doctor enters 1, 2, or 5 random parameters (what they have)
and the model must do its best to predict lab values.

Validation Strategy:
- Train imputer on Training Set (learn maximum correlations)
- Test on 3 scenarios with different data availability levels:
  - Scenario A (Poor): 1-2 random bedside params visible
  - Scenario B (Medium): 4-5 random params visible  
  - Scenario C (Rich): All bedside params visible

Target: Reconstruct hidden lab values (Lactate, WBC, Creatinine)
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def remove_high_missing_cols(df, threshold=50):
    """Remove columns with missing values above threshold percentage."""
    total = len(df)
    cols_to_drop = [col for col in df.columns 
                    if (df[col].isnull().sum() / total * 100) > threshold]
    return df.drop(columns=cols_to_drop)


def add_missing_flags(df, columns=None):
    """Create binary flag columns for missing values."""
    df = df.copy()
    if columns is None:
        columns = [col for col in df.columns if df[col].isnull().any()]
    for col in columns:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isnull().astype(int)
    return df


def create_omni_mask(df, fixed_cols, variable_pool, n_visible, random_state=None):
    """
    Create an omni-directional mask for validation:
    - fixed_cols: ALWAYS visible (demographics, ICU type)
    - variable_pool: n_visible are randomly kept visible, rest become TARGETS
    
    This allows testing imputation in any direction within the pool.
    
    Returns df_masked and ground_truth dict (for hidden pool variables).
    """
    rng = np.random.RandomState(random_state)
    df_masked = df.copy()
    ground_truth = {}
    
    for idx in df.index:
        # Fixed cols are always visible (never masked)
        
        # For variable pool: randomly select n_visible to keep, hide the rest
        available_pool = [c for c in variable_pool if c in df.columns and pd.notna(df.loc[idx, c])]
        
        if len(available_pool) > n_visible:
            # Keep n_visible random vars, hide the rest
            keep_cols = list(rng.choice(available_pool, size=min(n_visible, len(available_pool)), replace=False))
            hide_cols = [c for c in available_pool if c not in keep_cols]
            
            for col in hide_cols:
                # Save ground truth before masking
                if col not in ground_truth:
                    ground_truth[col] = {}
                ground_truth[col][idx] = df.loc[idx, col]
                df_masked.loc[idx, col] = np.nan
        else:
            # If we have fewer vars than n_visible, no masking needed for this row
            pass
    
    return df_masked, ground_truth


def evaluate_reconstruction(df_imputed, ground_truth, target_cols):
    """Calculate R² and MAE for target reconstruction."""
    results = {}
    
    for col in target_cols:
        if col not in ground_truth or len(ground_truth[col]) == 0:
            continue
            
        indices = list(ground_truth[col].keys())
        y_true = np.array([ground_truth[col][i] for i in indices])
        y_pred = df_imputed.loc[indices, col].values
        
        # Handle any remaining NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid_mask.sum() < 2:
            continue
            
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        results[col] = {
            'R2': r2_score(y_true_valid, y_pred_valid),
            'MAE': mean_absolute_error(y_true_valid, y_pred_valid),
            'n_samples': len(y_true_valid)
        }
    
    return results


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, '..', 'Dataset_ICU_Barbieri_Mollura.csv')
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    MAX_ITER = 10
    
    # ============================================================
    # RAPID ASSESSMENT MODEL (0-30 min window)
    # Omni-directional: given any subset, predict the rest
    # ============================================================
    
    # FIXED_COLS: Static data (always present at minute 0)
    FIXED_COLS = [
        'Age', 
        'Gender', 
        'Weight',
        # ICU Types (one-hot encoded)
        'CCU',    # Cardiac Care Unit
        'CSRU',   # Cardiac Surgery Recovery Unit
        'SICU'    # Surgical ICU
    ]
    
    # GROUP 1: Immediate Vital Signs (0-5 min - Bedside)
    # From multiparametric monitor and direct clinical assessment
    GROUP_1_VITAL_SIGNS = [
        'HR_first',          # Heart Rate
        'RespRate_first',    # Respiratory Rate
        'Temp_first',        # Temperature
        'GCS_first',         # Glasgow Coma Scale
        'SaO2_first',        # Oxygen Saturation
        'NISysABP_first',    # Systolic BP (Non-Invasive)
        'NIDiasABP_first',   # Diastolic BP (Non-Invasive)
        'NIMAP_first'        # Mean Arterial Pressure (Non-Invasive)
    ]
    
    # GROUP 2: Rapid Labs / Blood Gas (15-30 min - Point of Care)
    # From arterial catheter or rapid ABG
    GROUP_2_RAPID_LABS = [
        # Invasive Hemodynamics (arterial catheter)
        'SysABP_first',
        'DiasABP_first',
        'MAP_first',
        
        # Gas Exchange and Acid-Base Balance (ABG)
        'pH_first',
        'PaO2_first',
        'PaCO2_first',
        'FiO2_first',        # Administered O2
        'HCO3_first',        # Bicarbonates
        
        # Rapid Metabolism and Electrolytes
        'Lactate_first',     # CRITICAL TARGET
        'Glucose_first',
        'Na_first',          # Sodium
        'K_first'            # Potassium
    ]
    
    # VARIABLE_POOL: All dynamic variables the model can impute omni-directionally
    VARIABLE_POOL = GROUP_1_VITAL_SIGNS + GROUP_2_RAPID_LABS
    
    # Scenarios: (name, n_visible_from_pool)
    # Fixed are ALWAYS visible, we vary how many POOL vars are given
    SCENARIOS = [
        ("A: Minimo (2 var)", 2),      # e.g., only HR + Pressure
        ("B: Scarso (4 var)", 4),      # e.g., HR + BP + pH
        ("C: Parziale (6 var)", 6),    # Half the pool
        ("D: Buono (10 var)", 10),     # Most vital signs
    ]
    
    # MODELS TO COMPARE (Tournament)
    MODELS = [
        ("BayesianRidge", None),  # Default
        ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)),
        ("KNN", KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)),
    ]
    
    print("="*70)
    print("VIRTUAL SENSOR - Random Sparse Validation Benchmark")
    print("="*70)
    
    # ============================================================
    # STEP 1: LOAD AND PREPROCESS
    # ============================================================
    print("\n[STEP 1] Loading and Preprocessing...")
    
    df = pd.read_csv(input_csv)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    df = remove_high_missing_cols(df, threshold=50)
    df = df.drop(columns=[c for c in ['recordid', 'Height'] if c in df.columns])
    
    # Filter to existing columns
    fixed_cols = [c for c in FIXED_COLS if c in df.columns]
    variable_pool = [c for c in VARIABLE_POOL if c in df.columns]
    
    print(f"\n  FIXED columns (always visible): {len(fixed_cols)}")
    print(f"    → {fixed_cols}")
    print(f"\n  VARIABLE POOL (omni-directional): {len(variable_pool)}")
    print(f"    Group 1 (Vitals): {[c for c in GROUP_1_VITAL_SIGNS if c in df.columns]}")
    print(f"    Group 2 (Labs):   {[c for c in GROUP_2_RAPID_LABS if c in df.columns]}")
    
    # Add missing flags
    df = add_missing_flags(df)
    
    # Get feature columns (non-flag numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not c.endswith('_is_missing')]
    
    print(f"\n  Total features for model: {len(feature_cols)}")
    
    # ============================================================
    # STEP 2: TRAIN/TEST SPLIT
    # ============================================================
    print("\n[STEP 2] Train/Test Split...")
    
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  Train: {len(df_train)} patients")
    print(f"  Test: {len(df_test)} patients")
    
    # ============================================================
    # STEP 3: TOURNAMENT - TRAIN AND EVALUATE EACH MODEL
    # ============================================================
    print("\n[STEP 3] Tournament - Training and Evaluating Models...")
    print("="*70)
    
    all_model_results = []
    trained_imputers = {}  # Store trained imputers for saving later
    
    for model_name, estimator in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")
        
        # Create imputer
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        # Train on training data
        print(f"  Training on {len(df_train)} patients...")
        start_time = time.time()
        imputer.fit(df_train[feature_cols])
        train_time = time.time() - start_time
        print(f"  Training completed in {train_time:.1f}s")
        
        # Store trained imputer
        trained_imputers[model_name] = imputer
        
        # Evaluate on each scenario
        scenario_scores = []
        
        for scenario_name, n_visible in SCENARIOS:
            print(f"\n  Testing: {scenario_name}")
            
            # Create omni-directional mask on test set
            # N visible pool vars as input, rest as targets
            df_masked, ground_truth = create_omni_mask(
                df_test[feature_cols].copy(),
                fixed_cols=fixed_cols,
                variable_pool=variable_pool,
                n_visible=n_visible,
                random_state=RANDOM_STATE
            )
            
            # Transform (impute) using trained model
            start_time = time.time()
            imputed_values = imputer.transform(df_masked)
            inference_time = time.time() - start_time
            
            df_imputed = pd.DataFrame(imputed_values, columns=feature_cols, index=df_test.index)
            
            # Evaluate reconstruction on hidden pool variables
            col_results = evaluate_reconstruction(df_imputed, ground_truth, list(ground_truth.keys()))
            
            # Aggregate results
            r2_vals = [r['R2'] for r in col_results.values()]
            mae_vals = [r['MAE'] for r in col_results.values()]
            
            r2_mean = np.mean(r2_vals) if r2_vals else 0
            mae_mean = np.mean(mae_vals) if mae_vals else 0
            
            scenario_scores.append({
                'scenario': scenario_name,
                'n_visible': n_visible,
                'R2': r2_mean,
                'MAE': mae_mean,
                'inference_time': inference_time
            })
            
            print(f"    R² = {r2_mean:.4f} | MAE = {mae_mean:.4f} | Time = {inference_time:.2f}s")
        
        # Calculate overall R² (average across scenarios)
        overall_r2 = np.mean([s['R2'] for s in scenario_scores])
        overall_mae = np.mean([s['MAE'] for s in scenario_scores])
        
        all_model_results.append({
            'model_name': model_name,
            'overall_R2': overall_r2,
            'overall_MAE': overall_mae,
            'train_time': train_time,
            'scenario_scores': scenario_scores
        })
        
        print(f"\n  OVERALL: R² = {overall_r2:.4f} | MAE = {overall_mae:.4f}")
    
    # ============================================================
    # STEP 4: RANKING AND TOP 3 SELECTION
    # ============================================================
    print("\n" + "="*70)
    print("TOURNAMENT RESULTS - FINAL RANKING")
    print("="*70)
    
    # Sort by R² descending
    ranked_results = sorted(all_model_results, key=lambda x: x['overall_R2'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<20} {'R² (avg)':<12} {'MAE (avg)':<12} {'Train Time':<12}")
    print("-"*70)
    
    for rank, r in enumerate(ranked_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<4} {r['model_name']:<20} {r['overall_R2']:<12.4f} {r['overall_MAE']:<12.4f} {r['train_time']:<12.1f}s")
    
    # ============================================================
    # STEP 5: SAVE TOP 3 MODELS
    # ============================================================
    print("\n" + "="*70)
    print("SAVING TOP 3 MODELS")
    print("="*70)
    
    output_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    top_3 = ranked_results[:3]
    
    for rank, result in enumerate(top_3, 1):
        model_name = result['model_name']
        imputer = trained_imputers[model_name]
        
        # Create safe filename
        safe_name = model_name.replace(" ", "_").lower()
        filename = f"imputer_rank{rank}_{safe_name}.pkl"
        filepath = os.path.join(output_dir, filename)
        
        # Save using joblib
        joblib.dump(imputer, filepath)
        print(f"  Rank {rank}: {model_name}")
        print(f"         → Saved as: {filepath}")
        print(f"         → R² = {result['overall_R2']:.4f}")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
