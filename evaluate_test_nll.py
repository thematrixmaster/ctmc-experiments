"""
Evaluate trained WAG model on test set with per-transition NLL analysis.

Computes NLL for each transition and analyzes performance across branch length buckets.
"""

import torch
import numpy as np
from pathlib import Path
import pickle

from tqdm import tqdm

# Import from run_experiment_5
from run_experiment_5 import (
    WAGModel, encode_sequence, DEVICE, N_AA,
    load_transitions_from_file, TEST_DIR
)

# Configuration
CHECKPOINT_PATH = "./exp5_cache/exp5_wag_checkpoint.pkl"
TEST_FILE = TEST_DIR / "v1rodriguezCC.txt"

# Branch length buckets (nice round numbers)
BUCKETS = [
    (0.000, 0.010, "Very short [0.000, 0.010)"),
    (0.010, 0.050, "Short [0.010, 0.050)"),
    (0.050, 0.150, "Medium [0.050, 0.150)"),
    (0.150, float('inf'), "Long [0.150, ∞)")
]


def compute_transition_nll(model, parent_seq, child_seq, branch_length):
    """
    Compute NLL for a single transition (mean over valid sites).

    Args:
        model: Trained WAGModel
        parent_seq: Parent sequence string
        child_seq: Child sequence string
        branch_length: Branch length (float)

    Returns:
        nll: Negative log-likelihood per site (float)
        n_valid_sites: Number of valid sites (int)
    """
    # Encode sequences
    parent_enc = encode_sequence(parent_seq)
    child_enc = encode_sequence(child_seq)

    # Create mask for valid sites
    mask = (parent_enc >= 0) & (child_enc >= 0)
    n_valid_sites = mask.sum().item()

    if n_valid_sites == 0:
        return float('nan'), 0

    # Build Q matrix and compute P(t)
    with torch.no_grad():
        Q = model.build_Q()
        P = torch.matrix_exp(branch_length * Q)

        # Get transition probabilities for valid sites
        parent_valid = parent_enc[mask]
        child_valid = child_enc[mask]

        # Move to device
        parent_valid = parent_valid.to(DEVICE)
        child_valid = child_valid.to(DEVICE)

        # Get probabilities
        probs = P[parent_valid, child_valid]

        # Compute log-likelihood
        log_probs = torch.log(probs + 1e-10)
        total_log_likelihood = log_probs.sum().item()

        # NLL per site
        nll = -total_log_likelihood / n_valid_sites

    return nll, n_valid_sites


def main():
    print("=" * 80)
    print("Test Set Evaluation: Per-Transition NLL Analysis")
    print("=" * 80)

    # Load checkpoint
    print(f"\n[1/4] Loading trained model from {CHECKPOINT_PATH}")
    if not Path(CHECKPOINT_PATH).exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first using: python run_experiment_5.py")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    pi = checkpoint['pi'].to(DEVICE)

    model = WAGModel(pi).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best validation NLL: {checkpoint['best_val_nll']:.6f}")

    # Load test data
    print(f"\n[2/4] Loading test data from {TEST_FILE}")
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found at {TEST_FILE}")
        return

    test_transitions = load_transitions_from_file(TEST_FILE)
    print(f"  Loaded {len(test_transitions):,} test transitions")

    # Compute per-transition NLL
    print(f"\n[3/4] Computing per-transition NLL...")
    results = []
    total_valid_sites = 0

    for i, (parent, child, branch_length) in tqdm(enumerate(test_transitions), total=len(test_transitions)):
        # if (i + 1) % 5000 == 0:
        #     print(f"  Processed {i+1:,}/{len(test_transitions):,} transitions...")

        nll, n_valid = compute_transition_nll(model, parent, child, branch_length)

        if not np.isnan(nll):
            results.append({
                'transition_id': i,
                'branch_length': branch_length,
                'nll': nll,
                'n_valid_sites': n_valid
            })
            total_valid_sites += n_valid

    print(f"  Completed! Processed {len(results):,} valid transitions")

    # Compute overall statistics
    print(f"\n[4/4] Computing statistics...")

    all_nlls = [r['nll'] for r in results]
    all_branch_lengths = [r['branch_length'] for r in results]

    # Weight by number of sites for overall mean
    weighted_nll = sum(r['nll'] * r['n_valid_sites'] for r in results) / total_valid_sites

    # Print overall results
    print("\n" + "=" * 80)
    print("Test Set Evaluation Results")
    print("=" * 80)
    print(f"Total transitions: {len(results):,}")
    print(f"Total valid sites: {total_valid_sites:,}")
    print(f"\nOverall Test NLL (weighted by sites): {weighted_nll:.6f}")
    print(f"Mean NLL (unweighted): {np.mean(all_nlls):.6f}")
    print(f"Std NLL: {np.std(all_nlls):.6f}")
    print(f"Min NLL: {np.min(all_nlls):.6f}")
    print(f"Max NLL: {np.max(all_nlls):.6f}")

    # Bucket analysis
    print("\n" + "=" * 80)
    print("Branch Length Bucket Analysis")
    print("=" * 80)

    for bucket_min, bucket_max, bucket_name in BUCKETS:
        # Filter transitions in this bucket
        bucket_results = [
            r for r in results
            if bucket_min <= r['branch_length'] < bucket_max
        ]

        if len(bucket_results) == 0:
            print(f"\n{bucket_name}")
            print(f"  Count: 0 transitions (0.0%)")
            print(f"  No data in this range")
            continue

        bucket_nlls = [r['nll'] for r in bucket_results]
        bucket_sites = sum(r['n_valid_sites'] for r in bucket_results)
        bucket_weighted_nll = sum(r['nll'] * r['n_valid_sites'] for r in bucket_results) / bucket_sites

        percentage = 100 * len(bucket_results) / len(results)

        print(f"\n{bucket_name}")
        print(f"  Count: {len(bucket_results):,} transitions ({percentage:.1f}%)")
        print(f"  Valid sites: {bucket_sites:,}")
        print(f"  Mean NLL (weighted): {bucket_weighted_nll:.6f}")
        print(f"  Mean NLL (unweighted): {np.mean(bucket_nlls):.6f}")
        print(f"  Std NLL: {np.std(bucket_nlls):.6f}")
        print(f"  Min NLL: {np.min(bucket_nlls):.6f}")
        print(f"  Max NLL: {np.max(bucket_nlls):.6f}")
        print(f"  Mean branch length: {np.mean([r['branch_length'] for r in bucket_results]):.4f}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
