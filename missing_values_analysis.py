import pandas as pd

df = pd.read_csv('Dataset_ICU_Barbieri_Mollura.csv')
total = len(df)

# Calcolo missing per ogni colonna
results = []
for col in df.columns:
    missing = df[col].isnull().sum()
    pct = (missing / total * 100)
    results.append((col, missing, pct))

# Ordino per percentuale decrescente
results.sort(key=lambda x: x[2], reverse=True)

print(f'Totale righe: {total}\n')
for col, missing, pct in results:
    print(f'{col}: {missing} ({pct:.2f}%)')

# Conteggio colonne con >50% missing
cols_over_50 = [r for r in results if r[2] > 50]
total_cols = len(results)
print(f'\n--- RIEPILOGO ---')
print(f'Colonne totali: {total_cols}')
print(f'Colonne con >50% missing: {len(cols_over_50)} ({len(cols_over_50)/total_cols*100:.2f}%)')
