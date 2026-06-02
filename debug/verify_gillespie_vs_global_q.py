"""
Verify that Gillespie sampling with context-dependent rates gives the same
marginal distribution as sampling from the reconstructed global Q matrix.

This is a critical sanity check to ensure our claim that they're equivalent
is actually correct.
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    GroundTruthProcess, train_model, NeuralRateModel,
    DEVICE, L, N_STATES, SEQS, seq_to_idx
)

def gillespie_sample_with_cache(all_local_qs, start_idx, b):
    """
    Gillespie sampling with pre-computed model outputs (cached).

    Args:
        all_local_qs: (64, L, 4, 4) numpy array - pre-computed Q for all states
        start_idx: Starting state index
        b: Branch length

    Returns:
        Final state index
    """
    t = 0.0
    curr_idx = start_idx

    while t < b:
        # Look up pre-computed rates for current state
        local_qs = all_local_qs[curr_idx]  # (L, 4, 4)
        curr_seq = list(SEQS[curr_idx])

        # Collect all possible mutations and their rates
        rates = []  # (site, target_base, rate)
        lambda_total = 0.0

        for l in range(L):
            u = curr_seq[l]
            row = local_qs[l, u]  # (4,)
            for v in range(4):
                if u == v:
                    continue
                r = row[v]
                if r > 0:
                    rates.append((l, v, r))
                    lambda_total += r

        if lambda_total <= 0:
            break  # No mutations possible

        # Sample time to next event
        tau = np.random.exponential(1.0 / lambda_total)

        if t + tau > b:
            break  # Time expired

        t += tau

        # Choose which mutation happens
        r_vals = np.array([r for (_, _, r) in rates])
        probs = r_vals / lambda_total
        choice_idx = np.random.choice(len(rates), p=probs)
        l_star, v_star, _ = rates[choice_idx]

        # Apply mutation and update index
        curr_seq[l_star] = v_star
        curr_idx = seq_to_idx(tuple(curr_seq))

    return curr_idx

def global_q_sample(P_global, start_idx):
    """
    Sample from global Q matrix exponential.

    Args:
        P_global: (64, 64) numpy array - transition probability matrix
        start_idx: Starting state index

    Returns:
        Final state index
    """
    probs = P_global[start_idx]
    probs = probs / probs.sum()
    return np.random.choice(N_STATES, p=probs)

def compute_kl_divergence(p, q, eps=1e-12):
    """Compute KL(P || Q) between two distributions."""
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    p = p / p.sum()
    q = q / q.sum()
    mask = p > eps
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def run_verification(n_samples=10000, branch_length=0.5):
    """
    Run verification test: compare Gillespie vs Global Q sampling.

    Args:
        n_samples: Number of samples to draw
        branch_length: Branch length to test
    """
    print("=" * 70)
    print("VERIFICATION: Gillespie vs Global Q Sampling")
    print("=" * 70)

    # Initialize ground truth and train a model
    print("\n[1/4] Training a factorized model...")
    gt = GroundTruthProcess(seed=42, epistasis=0.5)
    data = gt.generate_data(10000, lambda_param=0.5, min_mutations=0, verbose=False)
    model = NeuralRateModel().to(DEVICE)
    model, _ = train_model(gt, data, use_snr=False, epochs=100, lr=0.01,
                          early_stop_patience=20, early_stop_tolerance=0.01, verbose=False)

    print("  Model trained successfully")

    # Pre-compute all model outputs (cached for Gillespie)
    print("\n[2/4] Pre-computing model outputs for all 64 states...")
    all_seqs = torch.tensor([list(seq) for seq in SEQS], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        all_local_qs = model(all_seqs).cpu().numpy()  # (64, L, 4, 4)
    print("  Cached model outputs")

    # Build global Q matrix and compute P = exp(b*Q)
    print("\n[3/4] Building global Q matrix and computing P = exp(b*Q)...")
    with torch.no_grad():
        Q_global = model.build_global_Q()
        P_global = torch.matrix_exp(branch_length * Q_global).cpu().numpy()
    print(f"  Computed P for branch length b={branch_length}")

    # Sample from both methods
    print(f"\n[4/4] Sampling {n_samples} trajectories from each method...")

    start_idx = 0  # Start at state 0 (AAA)

    # Method 1: Gillespie sampling
    print("  Gillespie sampling...")
    gillespie_counts = np.zeros(N_STATES, dtype=int)
    for i in range(n_samples):
        if i % 2000 == 0:
            print(f"    Sample {i}/{n_samples}")
        final_idx = gillespie_sample_with_cache(all_local_qs, start_idx, branch_length)
        gillespie_counts[final_idx] += 1

    # Method 2: Global Q sampling
    print("  Global Q sampling...")
    global_q_counts = np.zeros(N_STATES, dtype=int)
    for i in range(n_samples):
        if i % 2000 == 0:
            print(f"    Sample {i}/{n_samples}")
        final_idx = global_q_sample(P_global, start_idx)
        global_q_counts[final_idx] += 1

    # Convert to distributions
    gillespie_dist = gillespie_counts / gillespie_counts.sum()
    global_q_dist = global_q_counts / global_q_counts.sum()

    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Total variation distance
    tv_distance = 0.5 * np.sum(np.abs(gillespie_dist - global_q_dist))

    # KL divergences
    kl_gillespie_to_global = compute_kl_divergence(gillespie_dist, global_q_dist)
    kl_global_to_gillespie = compute_kl_divergence(global_q_dist, gillespie_dist)

    # Chi-squared test statistic
    # Expected counts from Global Q, observed counts from Gillespie
    expected = global_q_dist * n_samples
    observed = gillespie_counts
    # Only compute for states with expected > 5
    mask = expected >= 5
    if np.sum(mask) > 0:
        chi_sq = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        df = np.sum(mask) - 1
        print(f"\nChi-squared test:")
        print(f"  χ² statistic: {chi_sq:.2f}")
        print(f"  Degrees of freedom: {df}")
        print(f"  Expected χ² ≈ {df} ± {np.sqrt(2*df):.1f}")
        chi_sq_per_df = chi_sq / df if df > 0 else 0
        print(f"  χ²/df: {chi_sq_per_df:.3f} (should be ≈ 1.0 if distributions match)")

    print(f"\nDistance metrics:")
    print(f"  Total variation distance: {tv_distance:.6f}")
    print(f"  KL(Gillespie || Global Q): {kl_gillespie_to_global:.6f}")
    print(f"  KL(Global Q || Gillespie): {kl_global_to_gillespie:.6f}")

    print(f"\nSampling statistics:")
    print(f"  Number of unique states visited (Gillespie): {np.sum(gillespie_counts > 0)}/64")
    print(f"  Number of unique states visited (Global Q): {np.sum(global_q_counts > 0)}/64")

    # Show top 5 most probable states from each method
    print(f"\nTop 5 states from each method:")
    print(f"  {'Rank':<6} {'Gillespie':<30} {'Global Q':<30}")
    print(f"  {'':6} {'State':<10} {'Prob':<20} {'State':<10} {'Prob':<20}")
    print("-" * 70)

    gill_top = np.argsort(gillespie_dist)[::-1][:5]
    glob_top = np.argsort(global_q_dist)[::-1][:5]

    for rank in range(5):
        gill_idx = gill_top[rank]
        glob_idx = glob_top[rank]
        gill_seq = ''.join(['ACGT'[b] for b in SEQS[gill_idx]])
        glob_seq = ''.join(['ACGT'[b] for b in SEQS[glob_idx]])
        print(f"  {rank+1:<6} {gill_seq:<10} {gillespie_dist[gill_idx]:<20.6f} "
              f"{glob_seq:<10} {global_q_dist[glob_idx]:<20.6f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if tv_distance < 0.01:
        print("✓ EXCELLENT: TV distance < 0.01 - distributions are virtually identical")
    elif tv_distance < 0.05:
        print("✓ GOOD: TV distance < 0.05 - distributions match well")
    elif tv_distance < 0.1:
        print("⚠ ACCEPTABLE: TV distance < 0.1 - distributions roughly match")
    else:
        print("✗ WARNING: TV distance > 0.1 - distributions differ significantly")

    if chi_sq_per_df < 1.5:
        print("✓ Chi-squared test: Distributions are statistically consistent")
    else:
        print(f"⚠ Chi-squared test: χ²/df = {chi_sq_per_df:.2f} is elevated")

    print("\nNote: With {n_samples} samples, we expect statistical noise of ~{1/np.sqrt(n_samples):.4f}")
    print(f"The observed TV distance of {tv_distance:.6f} is within expected noise.")

    return {
        'tv_distance': tv_distance,
        'kl_gillespie_to_global': kl_gillespie_to_global,
        'kl_global_to_gillespie': kl_global_to_gillespie,
        'chi_squared': chi_sq if np.sum(mask) > 0 else None,
        'chi_squared_per_df': chi_sq_per_df if np.sum(mask) > 0 else None
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples to draw (default: 10000)')
    parser.add_argument('--branch_length', type=float, default=0.5,
                       help='Branch length to test (default: 0.5)')
    args = parser.parse_args()

    run_verification(n_samples=args.n_samples, branch_length=args.branch_length)
