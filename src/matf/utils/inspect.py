import pandas as pd
from pathlib import Path
from typing import Optional


def inspect_parquet(file_path: str | Path, num_rows: int = 5, verbose: bool = True) -> pd.DataFrame:
    """
    Load and inspect a parquet file.

    Args:
        file_path: Path to the parquet file.
        num_rows: Number of rows to display from the head.
        verbose: If True, prints information to the console.

    Returns:
        DataFrame loaded from the parquet file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)

    if verbose:
        print("=" * 80)
        print(f"Inspecting: {file_path}")
        print("=" * 80)
        print(f"\nShape: {df.shape}")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")

        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")

        print(f"\nFirst {num_rows} rows:")
        print(df.head(num_rows))

        if 'observed' in df.columns:
            true_count = df['observed'].sum()  # True counts as 1
            false_count = len(df) - true_count
            print(f"\n'observed' counts -> True: {true_count}, False: {false_count}")
        else:
            print("\nNo column named 'observed' found.")

        print("\nUnique values per column (top-level summary):")
        for col in df.columns:
            try:
                unique_count = df[col].nunique()
                print(f"{col}: {unique_count}")
            except Exception:
                print(f"{col}: (unable to compute unique count)")

        print("\nDone.")

    return df
