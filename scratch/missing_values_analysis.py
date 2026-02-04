import pandas as pd


def remove_high_missing_cols(df, threshold=50):
    """
    Remove columns with missing values above threshold percentage.
    Returns the cleaned dataframe.
    """
    total = len(df)
    cols_to_drop = []
    
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = (missing / total * 100)
        if pct > threshold:
            cols_to_drop.append(col)
    
    return df.drop(columns=cols_to_drop)


if __name__ == "__main__":
    df = pd.read_csv('Dataset_ICU_Barbieri_Mollura.csv')
    total = len(df)

    # Calculate missing values for each column
    results = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        pct = (missing / total * 100)
        results.append((col, missing, pct))

    # Sort by percentage descending
    results.sort(key=lambda x: x[2], reverse=True)

    print(f'Total rows: {total}\n')
    for col, missing, pct in results:
        print(f'{col}: {missing} ({pct:.2f}%)')

    # Count columns with >50% missing
    cols_over_50 = [r for r in results if r[2] > 50]
    total_cols = len(results)
    print(f'\n--- SUMMARY ---')
    print(f'Total columns: {total_cols}')
    print(f'Columns with >50% missing: {len(cols_over_50)} ({len(cols_over_50)/total_cols*100:.2f}%)')

    # Remove columns with >50% missing
    df = remove_high_missing_cols(df)
    print(f'\nRemaining columns after removal: {len(df.columns)}')

