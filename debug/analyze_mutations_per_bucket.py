"""
Analyze average number of mutations per branch length bucket.

Reads test data and computes mutation counts (Hamming distance) between
parent and child sequences, stratified by branch length buckets.
"""

from pathlib import Path
import numpy as np

# Configuration
TEST_FILE = Path("./data/test/v1rodriguezCC.txt")

# Same branch length buckets
BUCKETS = [
    (0.000, 0.010, "Very short [0.000, 0.010)"),
    (0.010, 0.050, "Short [0.010, 0.050)"),
    (0.050, 0.150, "Medium [0.050, 0.150)"),
    (0.150, float('inf'), "Long [0.150, ∞)")
]


def count_mutations(parent, child):
    """Count number of mutations (Hamming distance) between two sequences."""
    if len(parent) != len(child):
        return None

    mutations = sum(1 for p, c in zip(parent, child) if p != c)
    return mutations


def main():
    print("=" * 80)
    print("Mutation Count Analysis by Branch Length Bucket")
    print("=" * 80)

    # Load test data
    print(f"\n[1/2] Loading test data from {TEST_FILE}")
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found at {TEST_FILE}")
        return

    transitions = []
    with open(TEST_FILE, 'r') as f:
        # Skip header
        header = f.readline()

        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            parent = parts[0]
            child = parts[1]
            branch_length = float(parts[2])

            mutations = count_mutations(parent, child)
            if mutations is not None:
                transitions.append({
                    'parent': parent,
                    'child': child,
                    'branch_length': branch_length,
                    'mutations': mutations,
                    'seq_length': len(parent)
                })

    print(f"  Loaded {len(transitions):,} transitions")

    # Overall statistics
    print(f"\n[2/2] Computing statistics...")

    all_mutations = [t['mutations'] for t in transitions]
    all_seq_lengths = [t['seq_length'] for t in transitions]
    all_branch_lengths = [t['branch_length'] for t in transitions]

    print("\n" + "=" * 80)
    print("Overall Results")
    print("=" * 80)
    print(f"Total transitions: {len(transitions):,}")
    print(f"\nMean mutations: {np.mean(all_mutations):.3f}")
    print(f"Std mutations: {np.std(all_mutations):.3f}")
    print(f"Min mutations: {np.min(all_mutations)}")
    print(f"Max mutations: {np.max(all_mutations)}")
    print(f"\nMean sequence length: {np.mean(all_seq_lengths):.1f}")
    print(f"Mean branch length: {np.mean(all_branch_lengths):.6f}")

    # Bucket analysis
    print("\n" + "=" * 80)
    print("Branch Length Bucket Analysis")
    print("=" * 80)

    for bucket_min, bucket_max, bucket_name in BUCKETS:
        # Filter transitions in this bucket
        if bucket_max == float('inf'):
            bucket_transitions = [t for t in transitions if t['branch_length'] >= bucket_min]
        else:
            bucket_transitions = [
                t for t in transitions
                if bucket_min <= t['branch_length'] < bucket_max
            ]

        if len(bucket_transitions) == 0:
            print(f"\n{bucket_name}")
            print(f"  Count: 0 transitions (0.0%)")
            print(f"  No data in this range")
            continue

        bucket_mutations = [t['mutations'] for t in bucket_transitions]
        bucket_seq_lengths = [t['seq_length'] for t in bucket_transitions]
        bucket_branch_lengths = [t['branch_length'] for t in bucket_transitions]

        percentage = 100 * len(bucket_transitions) / len(transitions)

        print(f"\n{bucket_name}")
        print(f"  Count: {len(bucket_transitions):,} transitions ({percentage:.1f}%)")
        print(f"  Mean mutations: {np.mean(bucket_mutations):.3f}")
        print(f"  Std mutations: {np.std(bucket_mutations):.3f}")
        print(f"  Min mutations: {int(np.min(bucket_mutations))}")
        print(f"  Max mutations: {int(np.max(bucket_mutations))}")
        print(f"  Mean sequence length: {np.mean(bucket_seq_lengths):.1f}")
        print(f"  Mean branch length: {np.mean(bucket_branch_lengths):.6f}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
