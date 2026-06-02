"""
Run Experiment 4: Sampling Method Comparison (EXACT KL DIVERGENCE)

Compare site-independent vs global Q sampling for the factorized model
against ground truth across branch lengths and epistasis values.

Configuration:
- Ground truth: Various epistasis values [0.0, 0.25, 0.5, 0.75, 1.0], seed=42
- Training: 2.5M samples, lambda=0.5
- Branch lengths: 30 logarithmically-spaced values from 0.01 to 10.0
- Metrics: EXACT KL divergence (no sampling - computed from full distributions)

Two sampling methods compared:
1. Matrix-Exp (Site-Independent): Product of local site transition probabilities (approximation)
2. Gillespie (Global Q): Full 64x64 matrix exponential from factorized model (exact)

Two evaluation modes:
1. Single starting state (AAA)
2. Uniform average over all 64 starting states

Key improvements:
- No sampling noise (exact distributions)
- Both single-state and uniform-average KL
- Very fast execution (~seconds instead of hours)
- Uses cached models for efficiency
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from main import (
    GroundTruthProcess, train_model, NeuralRateModel,
    save_factorized_model, load_factorized_model,
    get_cache_key,
    DEVICE, L, N_STATES, SEQS, seq_to_idx, PLOTS_DIR, CACHE_DIR
)

def compute_site_independent_distribution(model, start_idx, b):
    """
    Compute transition distribution using site-independent approximation.

    Uses local Q matrices from the model at the starting state,
    computes P = exp(b*Q) for each site independently, then computes
    the product distribution. This is an approximation that ignores
    correlations between sites.

    Returns:
        probs: (N_STATES,) array of transition probabilities
    """
    start_seq = list(SEQS[start_idx])
    start_tens = torch.tensor([start_seq], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        q_preds = model(start_tens)  # (1, L, 4, 4)

    # Compute local transition matrices for each site
    small_ps = []
    for l in range(L):
        P_l = torch.matrix_exp(b * q_preds[0, l])  # (4, 4)
        small_ps.append(P_l.cpu().numpy())

    # Compute product distribution over all sequences
    probs = np.zeros(N_STATES)
    for end_idx in range(N_STATES):
        end_seq = SEQS[end_idx]
        # Product of independent site probabilities
        prob = 1.0
        for l in range(L):
            start_base = start_seq[l]
            end_base = end_seq[l]
            prob *= small_ps[l][start_base, end_base]
        probs[end_idx] = prob

    # Normalize
    probs = probs / probs.sum()
    return probs

def compute_global_q_distribution(model, start_idx, b):
    """
    Compute transition distribution using full global Q matrix (context-dependent).

    Builds the full 64x64 global Q matrix from the factorized model,
    computes P = exp(b*Q), then returns the exact transition distribution.
    This is mathematically equivalent to Gillespie sampling but much faster.

    Returns:
        probs: (N_STATES,) array of transition probabilities
    """
    with torch.no_grad():
        # Build full 64x64 global Q matrix
        Q_global = model.build_global_Q()  # (N_STATES, N_STATES)

        # Compute full transition matrix P = exp(b * Q)
        P_global = torch.matrix_exp(b * Q_global)  # (N_STATES, N_STATES)

        # Get transition probabilities from starting state
        probs = P_global[start_idx].cpu().numpy()
        probs = probs / probs.sum()  # Normalize

    return probs

def compute_ground_truth_distribution(gt, start_idx, b):
    """
    Compute ground truth transition distribution using exact matrix exponential.

    Returns:
        probs: (N_STATES,) array of transition probabilities
    """
    P = gt.get_P(b)  # (N_STATES, N_STATES)
    probs = P[start_idx]
    probs = probs / probs.sum()  # Normalize
    return probs

def compute_kl_divergence(p, q, eps=1e-12):
    """
    Compute KL(P || Q) for two probability distributions.

    KL(P || Q) = sum_x P(x) log(P(x) / Q(x))

    Args:
        p: (N_STATES,) probability distribution
        q: (N_STATES,) probability distribution
        eps: Small constant for numerical stability

    Returns:
        KL divergence value
    """
    # Add epsilon for numerical stability
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)

    # Renormalize
    p = p / p.sum()
    q = q / q.sum()

    # Compute KL divergence (only sum where p > 0)
    mask = p > eps
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return kl

def run_experiment_4_single(epistasis, seed=42, n_samples_train=2500000, lambda_param=0.5):
    """
    Run experiment 4 for a single epistasis value.

    Args:
        epistasis: Epistasis strength
        seed: Random seed
        n_samples_train: Number of training samples
        lambda_param: Lambda parameter for data generation

    Returns:
        results: Dictionary with KL divergence results
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4: Epistasis = {epistasis}")
    print("=" * 70)

    # ========== Setup ==========
    start_idx = 0  # State 0 = "AAA"

    # Branch lengths (log spacing, dense near 0)
    branch_lengths = np.logspace(np.log10(0.01), np.log10(10.0), 30)

    print(f"Configuration:")
    print(f"  Ground truth: epistasis={epistasis}, seed={seed}")
    print(f"  Training samples: {n_samples_train}")
    print(f"  Branch lengths: {len(branch_lengths)} values from {branch_lengths[0]:.3f} to {branch_lengths[-1]:.1f}")
    print(f"  Starting state: {SEQS[start_idx]} (state {start_idx})")
    print(f"  Using EXACT KL divergence (no sampling noise)")
    print("=" * 70)

    # ========== Initialize Ground Truth ==========
    print("\n[1/5] Initializing ground truth...")
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    print(f"  Ground truth Q matrix: {Q_true.shape}, mean rate: {gt.Q_global[gt.Q_global > 0].mean():.4f}")

    # ========== Check if we need to train any models ==========
    key = get_cache_key(epistasis, seed, "factorized", n_samples_train)
    cache_path = os.path.join(CACHE_DIR, f"factorized_{key}.pt")

    need_training = not os.path.exists(cache_path)

    # ========== Generate Training Data ==========
    if need_training:
        print("\n[2/5] Generating training data...")
        print(f"  Generating {n_samples_train} samples...")
        data = gt.generate_data(n_samples_train, lambda_param=lambda_param, min_mutations=0, verbose=True)
    else:
        print("\n[2/5] Skipping data generation (all models cached)...")
        data = None

    # ========== Train/Load Factorized Model ==========
    print("\n[3/5] Training/loading factorized model...")

    # Standard factorized model
    if os.path.exists(cache_path):
        print(f"  Loading cached factorized model...")
        save_dict = torch.load(cache_path, weights_only=False, map_location=DEVICE)
        model_factorized = NeuralRateModel().to(DEVICE)
        model_factorized.load_state_dict(save_dict['model_state'])
        Q_factorized = model_factorized.build_global_Q()
        error_factorized = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)
        print(f"  Loaded from cache. Error: {error_factorized.item():.4f}")
    else:
        print(f"  Training factorized model...")
        model_factorized, _ = train_model(gt, data, use_snr=False, epochs=1000, lr=0.01,
                                         early_stop_patience=50, early_stop_tolerance=0.001, verbose=True)
        Q_factorized = model_factorized.build_global_Q()
        error_factorized = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)
        print(f"  Factorized model error: {error_factorized.item():.4f}")

        # Save to cache
        save_factorized_model(model_factorized, epistasis, seed, n_samples_train)

    # ========== Compute Exact KL Divergences ==========
    print("\n[4/5] Computing exact KL divergences...")
    print(f"  Testing {len(branch_lengths)} branch lengths...")

    results = {
        'branch_lengths': branch_lengths,
        'single_state': {
            'factorized_site_indep': [],
            'factorized_global_q': []
        },
        'uniform_avg': {
            'factorized_site_indep': [],
            'factorized_global_q': []
        }
    }

    for i, b in enumerate(branch_lengths):
        if i % 5 == 0:
            print(f"  Branch length {i+1}/{len(branch_lengths)}: b={b:.4f}")

        # ========== Single Starting State (state 0) ==========
        # Ground truth distribution
        p_true = compute_ground_truth_distribution(gt, start_idx, b)

        # Factorized model distributions
        p_fact_site_indep = compute_site_independent_distribution(model_factorized, start_idx, b)
        p_fact_global_q = compute_global_q_distribution(model_factorized, start_idx, b)

        # Compute KL divergences for single state
        results['single_state']['factorized_site_indep'].append(compute_kl_divergence(p_fact_site_indep, p_true))
        results['single_state']['factorized_global_q'].append(compute_kl_divergence(p_fact_global_q, p_true))

        # ========== Uniform Average Over All Starting States ==========
        kl_fact_site_indep_avg = 0.0
        kl_fact_global_q_avg = 0.0

        for start_idx_avg in range(N_STATES):
            # Ground truth
            p_true_avg = compute_ground_truth_distribution(gt, start_idx_avg, b)

            # Factorized
            p_fact_site_indep_avg = compute_site_independent_distribution(model_factorized, start_idx_avg, b)
            p_fact_global_q_avg = compute_global_q_distribution(model_factorized, start_idx_avg, b)

            # Accumulate KL divergences
            kl_fact_site_indep_avg += compute_kl_divergence(p_fact_site_indep_avg, p_true_avg)
            kl_fact_global_q_avg += compute_kl_divergence(p_fact_global_q_avg, p_true_avg)

        # Average over all starting states
        results['uniform_avg']['factorized_site_indep'].append(kl_fact_site_indep_avg / N_STATES)
        results['uniform_avg']['factorized_global_q'].append(kl_fact_global_q_avg / N_STATES)

    # ========== Print Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY: KL Divergence at Key Branch Lengths (Single State)")
    print("=" * 70)
    print(f"{'Branch':<10} {'Matrix-Exp':<15} {'Gillespie':<15}")
    print("-" * 70)

    key_indices = [0, 7, 14, 21, 29]  # Roughly 0.01, 0.05, 0.25, 1.0, 5.0
    for idx in key_indices:
        b = branch_lengths[idx]
        print(f"{b:<10.3f} {results['single_state']['factorized_site_indep'][idx]:<15.6f} "
              f"{results['single_state']['factorized_global_q'][idx]:<15.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY: KL Divergence at Key Branch Lengths (Uniform Average)")
    print("=" * 70)
    print(f"{'Branch':<10} {'Matrix-Exp':<15} {'Gillespie':<15}")
    print("-" * 70)

    for idx in key_indices:
        b = branch_lengths[idx]
        print(f"{b:<10.3f} {results['uniform_avg']['factorized_site_indep'][idx]:<15.6f} "
              f"{results['uniform_avg']['factorized_global_q'][idx]:<15.6f}")

    # ========== Plot Results ==========
    print("\n[5/5] Generating plot...")

    # Single plot: Uniform Average only
    plt.figure(figsize=(3.25, 2.5))

    # Blue color for factorized model
    color_blue = '#0173B2'  # Blue

    # Plot uniform average results - two lines only
    plt.plot(branch_lengths, results['uniform_avg']['factorized_site_indep'],
             '-', linewidth=1.5, color=color_blue, label='Matrix-Exp')
    plt.plot(branch_lengths, results['uniform_avg']['factorized_global_q'],
             ':', linewidth=2.0, color=color_blue, label='Gillespie')

    plt.xscale('log')
    plt.xlabel('Branch Length', fontsize=10)
    plt.ylabel('KL Divergence', fontsize=10)
    plt.legend(fontsize=7, loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=0.3)

    # Save plot with epistasis-specific filename
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path_pdf = os.path.join(PLOTS_DIR, f'exp4_epistasis_{epistasis:.2f}.pdf')
    plot_path_png = os.path.join(PLOTS_DIR, f'exp4_epistasis_{epistasis:.2f}.png')
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    print(f"\nSaved {plot_path_pdf}")
    print(f"Saved {plot_path_png}")
    plt.close()

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4 COMPLETED (Epistasis = {epistasis})!")
    print("=" * 70)

    return results

def run_experiment_4():
    """
    Run experiment 4 for multiple epistasis values.

    Tests how sampling method comparison changes with epistasis strength.
    Runs a single replicate per epistasis value.
    """
    epistasis_values = [1.0, 0.75, 0.5, 0.25, 0.0]
    all_results = {}

    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SAMPLING METHOD COMPARISON ACROSS EPISTASIS VALUES")
    print("=" * 80)
    print(f"Testing {len(epistasis_values)} epistasis values: {epistasis_values}")
    print(f"Single seed: 42")
    print("=" * 80)

    for i, eps in enumerate(epistasis_values):
        print(f"\n{'#' * 80}")
        print(f"# Running {i+1}/{len(epistasis_values)}: Epistasis = {eps}")
        print(f"{'#' * 80}")

        results = run_experiment_4_single(epistasis=eps, seed=42)
        all_results[eps] = results

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"Generated {len(epistasis_values)} plots:")
    for eps in epistasis_values:
        print(f"  - plots/exp4_epistasis_{eps:.2f}.pdf")

    return all_results

if __name__ == "__main__":
    results = run_experiment_4()
