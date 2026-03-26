"""
Analyze pre-computed likelihoods from FT150M model.

Loads CSV with branch_length, ll_per_site, ppl columns and computes
statistics stratified by the same branch length buckets.
"""

import pandas as pd
import numpy as np

# Configuration
CSV_PATH = "/scratch/users/stephen.lu/projects/protevo/paper/ctmc_eval/ctmc_ft150M_ll_rodriguezCC.csv"

# Same branch length buckets as evaluate_test_nll.py
BUCKETS = [
    (0.000, 0.010, "Very short [0.000, 0.010)"),
    (0.010, 0.050, "Short [0.010, 0.050)"),
    (0.050, 0.150, "Medium [0.050, 0.150)"),
    (0.150, float('inf'), "Long [0.150, ∞)")
]


def main():
    print("=" * 80)
    print("FT150M Model Evaluation: Branch Length Bucket Analysis")
    print("=" * 80)

    # Load data
    print(f"\n[1/2] Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")

    # Compute NLL (negate ll_per_site)
    df['nll'] = -df['ll_per_site']

    # Overall statistics
    print(f"\n[2/2] Computing statistics...")
    print("\n" + "=" * 80)
    print("Overall Results")
    print("=" * 80)
    print(f"Total transitions: {len(df):,}")
    print(f"\nMean NLL: {df['nll'].mean():.6f}")
    print(f"Std NLL: {df['nll'].std():.6f}")
    print(f"Min NLL: {df['nll'].min():.6f}")
    print(f"Max NLL: {df['nll'].max():.6f}")
    print(f"\nMean branch length: {df['branch_length'].mean():.4f}")
    print(f"Std branch length: {df['branch_length'].std():.4f}")

    # Bucket analysis
    print("\n" + "=" * 80)
    print("Branch Length Bucket Analysis")
    print("=" * 80)

    for bucket_min, bucket_max, bucket_name in BUCKETS:
        # Filter rows in this bucket
        if bucket_max == float('inf'):
            bucket_df = df[df['branch_length'] >= bucket_min]
        else:
            bucket_df = df[(df['branch_length'] >= bucket_min) & (df['branch_length'] < bucket_max)]

        if len(bucket_df) == 0:
            print(f"\n{bucket_name}")
            print(f"  Count: 0 transitions (0.0%)")
            print(f"  No data in this range")
            continue

        percentage = 100 * len(bucket_df) / len(df)

        print(f"\n{bucket_name}")
        print(f"  Count: {len(bucket_df):,} transitions ({percentage:.1f}%)")
        print(f"  Mean NLL: {bucket_df['nll'].mean():.6f}")
        print(f"  Std NLL: {bucket_df['nll'].std():.6f}")
        print(f"  Min NLL: {bucket_df['nll'].min():.6f}")
        print(f"  Max NLL: {bucket_df['nll'].max():.6f}")
        print(f"  Mean branch length: {bucket_df['branch_length'].mean():.4f}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
