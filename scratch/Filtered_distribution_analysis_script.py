import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 0. Config
# ==================================================

FILE_PATH = "Dataset_ICU_Barbieri_Mollura.csv"
SEP = ","                         # "," | ";" | "\t"

plt.style.use("ggplot")

# Subplot size
N_ROWS = 4
N_COLS = 4
PER_FIG = N_ROWS * N_COLS

# ==================================================
# 1. DATA LOADING
# ==================================================

df = pd.read_csv(FILE_PATH, sep=SEP)

print("\n--- INFO DATAFRAME ORIGINALE ---")
print(df.info())

# ==================================================
# 2. FILTERING (correlation + NaNs)
# ==================================================

cols_to_drop = [
    # ---- correlate ----
    "ALP_last",
    "AST_first",
    "AST_last",
    "Albumin_last",
    "BUN_last",
    "Bilirubin_last",
    "Cholesterol_last",
    "Creatinine_last",
    "DiasABP_median",
    "GCS_last",
    "GCS_median",
    "HR_last",
    "HR_median",
    "MAP_last",
    "MAP_median",
    "MechVentLast8Hour",
    "NIMAP_first",
    "NIMAP_highest",
    "NIMAP_last",
    "NIMAP_lowest",
    "NIMAP_median",
    "NISysABP_median",
    "Platelets_last",
    "SaO2_last",
    "SaO2_median",
    "SysABP_median",
    "Temp_median",
    "TroponinI_last",
    "TroponinT_last",
    "Weight",
    "Weight_last",
    # ---- high NaN ----
    "TroponinI_first",
    "Cholesterol_first",
    "TroponinT_first",
    "RespRate_first",
    "RespRate_last",
    "RespRate_lowest",
    "RespRate_highest",
    "RespRate_median",
    "Albumin_first",
    "ALP_first",
    "Bilirubin_first",
    "ALT_first",
    "ALT_last",
    "SaO2_first",
    "SaO2_lowest",
    "SaO2_highest",
    "Height",
    "recordid"
]

# filters
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

print("\n--- Removed columns ---")
print(cols_to_drop)

# ==================================================
# 3. Reduced DF
# ==================================================

cols_to_keep = [c for c in df.columns if c not in cols_to_drop]
df_reduced = df[cols_to_keep].copy()

print("\n--- INFO Reduced DF ---")
print(df_reduced.info())

# Numerical variables of reduced df
num_cols = df_reduced.select_dtypes(include=[np.number]).columns

print("\n--- % MISSING (first 20 variables of df_reduced) ---")
print(df_reduced.isna().mean().sort_values(ascending=False).head(20))

print("\n--- Description of Numerical variables (df_reduced) ---")
print(df_reduced[num_cols].describe().T)

num_cols_no_target = [c for c in num_cols if c != "In-hospital_death"]


# ===========================================================================
# 7. Comparison alive vs deaths (hystogram and boxplot) for filtered vairables
# ===========================================================================

target = "In-hospital_death"

if target in df_reduced.columns:
    print("\n--- Death vs Alive Comparison (Filtered Hyst) ---")
    for start in range(0, len(num_cols_no_target), PER_FIG):
        subset = num_cols_no_target[start:start+PER_FIG]

        fig, axes = plt.subplots(N_ROWS, N_COLS,
                                 figsize=(5 * N_COLS, 3 * N_ROWS))
        axes = axes.flatten()

        for ax, col in zip(axes, subset):
            alive = df_reduced[df_reduced[target] == 0][col].dropna()
            dead  = df_reduced[df_reduced[target] == 1][col].dropna()

            ax.hist(alive, bins=30, alpha=0.5,
                    label="Vivi", edgecolor="black")
            ax.hist(dead,  bins=30, alpha=0.5,
                    label="Morti", edgecolor="black")
            ax.set_title(col, fontsize=9)

        for j in range(len(subset), len(axes)):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        fig.suptitle("Distribution Alive vs Death (filtered)", fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\n--- Death vs Alive Comparison (Filtered Boxplot) ---")
    for start in range(0, len(num_cols_no_target), PER_FIG):
        subset = num_cols_no_target[start:start+PER_FIG]

        fig, axes = plt.subplots(N_ROWS, N_COLS,
                                 figsize=(5 * N_COLS, 3 * N_ROWS))
        axes = axes.flatten()

        for ax, col in zip(axes, subset):
            alive = df_reduced[df_reduced[target] == 0][col].dropna()
            dead  = df_reduced[df_reduced[target] == 1][col].dropna()

            ax.boxplot([alive, dead], labels=["Alive", "Death"])
            ax.set_title(col, fontsize=9)

        for j in range(len(subset), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Boxplot by status (Alive vs Death, filtered)", fontsize=16)
        plt.tight_layout()
        plt.show()
else:
    print("\nATTENZIONE: 'In-hospital_death' not available in df_reduced.")

