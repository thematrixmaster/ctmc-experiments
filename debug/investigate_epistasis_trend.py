"""
Investigate why Full MLE error increases with epistasis.

This script will:
1. Check if ground truth Q matrices become "harder" with more epistasis
2. Verify data quality across epistasis levels
3. Test if there's a fundamental identifiability issue
4. Examine transition coverage
"""

import sys
sys.path.append('/accounts/projects/yss/stephen.lu/ctmc')

from main import *
import matplotlib.pyplot as plt
import numpy as np

def analyze_ground_truth_difficulty(epistasis_levels=[0.0, 0.3, 0.7, 1.0], seed=42):
    """Check if Q matrices become harder to estimate with more epistasis."""
    print("=" * 80)
    print("ANALYSIS 1: Ground Truth Q Matrix Properties vs Epistasis")
    print("=" * 80)

    for eps in epistasis_levels:
        gt = GroundTruthProcess(seed=seed, epistasis=eps)
        Q = gt.Q_global

        # Compute various metrics
        frobenius_norm = np.linalg.norm(Q, 'fro')
        spectral_norm = np.linalg.norm(Q, 2)
        condition_number = np.linalg.cond(Q)

        # Off-diagonal variance (measure of heterogeneity)
        off_diag_entries = []
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                    off_diag_entries.append(Q[i, j])

        rate_mean = np.mean(off_diag_entries)
        rate_std = np.std(off_diag_entries)
        rate_cv = rate_std / rate_mean if rate_mean > 0 else 0  # Coefficient of variation

        print(f"\nEpistasis = {eps:.1f}:")
        print(f"  Frobenius norm:      {frobenius_norm:.4f}")
        print(f"  Spectral norm:       {spectral_norm:.4f}")
        print(f"  Condition number:    {condition_number:.2e}")
        print(f"  Off-diag rate mean:  {rate_mean:.4f}")
        print(f"  Off-diag rate std:   {rate_std:.4f}")
        print(f"  Coefficient of var:  {rate_cv:.4f}")
        print(f"  Min rate:            {np.min(off_diag_entries):.4f}")
        print(f"  Max rate:            {np.max(off_diag_entries):.4f}")
        print(f"  Range/mean ratio:    {(np.max(off_diag_entries) - np.min(off_diag_entries)) / rate_mean:.4f}")


def analyze_data_quality(epistasis_levels=[0.0, 0.3, 0.7, 1.0], n_samples=10000, seed=42):
    """Check if data quality degrades with epistasis."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Data Quality vs Epistasis")
    print("=" * 80)

    for eps in epistasis_levels:
        gt = GroundTruthProcess(seed=seed, epistasis=eps)
        data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)

        # Analyze data statistics
        branch_lengths = [d[2] for d in data]
        mutations = [hamming(SEQS[seq_to_idx(tuple(d[0].tolist()))],
                            SEQS[seq_to_idx(tuple(d[1].tolist()))])
                    for d in data]

        # Transition coverage
        transition_counts = np.zeros((N_STATES, N_STATES))
        for start_seq, end_seq, b in data:
            i = seq_to_idx(tuple(start_seq.tolist()))
            j = seq_to_idx(tuple(end_seq.tolist()))
            transition_counts[i, j] += 1

        # Count how many Hamming-1 transitions are observed
        hamming1_transitions = []
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                    hamming1_transitions.append((i, j))

        observed_transitions = sum(1 for (i, j) in hamming1_transitions if transition_counts[i, j] > 0)
        total_transitions = len(hamming1_transitions)
        coverage = observed_transitions / total_transitions

        # Average counts per observed transition
        counts_per_observed = [transition_counts[i, j] for (i, j) in hamming1_transitions if transition_counts[i, j] > 0]

        print(f"\nEpistasis = {eps:.1f}:")
        print(f"  Samples:                  {len(data)}")
        print(f"  Mean branch length:       {np.mean(branch_lengths):.4f}")
        print(f"  Mean mutations:           {np.mean(mutations):.4f}")
        print(f"  Transition coverage:      {coverage:.2%} ({observed_transitions}/{total_transitions})")
        print(f"  Avg counts/transition:    {np.mean(counts_per_observed):.2f}")
        print(f"  Min counts (if observed): {np.min(counts_per_observed):.0f}")
        print(f"  Max counts:               {np.max(counts_per_observed):.0f}")


def test_identifiability(epistasis_levels=[0.0, 1.0], n_samples=10000, seed=42):
    """Test if there's a fundamental identifiability issue."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Identifiability Test (Oracle Experiment)")
    print("=" * 80)
    print("What if we give the Full MLE model PERFECT initialization?")
    print()

    for eps in epistasis_levels:
        gt = GroundTruthProcess(seed=seed, epistasis=eps)
        data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)

        # Create model and initialize with TRUE parameters
        model = FullMLEModel().to(DEVICE)

        # Initialize parameters to ground truth
        Q_true = gt.Q_global
        with torch.no_grad():
            for (i, j), idx in model.param_indices.items():
                true_rate = Q_true[i, j]
                # Inverse of softplus: softplus(x) = log(exp(x) + 1) ≈ x for large x
                # For small values: x = log(exp(rate) - 1)
                if true_rate > 0:
                    model.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

        # Verify initialization
        Q_init = model.build_Q().detach().cpu().numpy()
        init_error = np.linalg.norm(Q_init - Q_true, 'fro') / np.linalg.norm(Q_true, 'fro')

        print(f"Epistasis = {eps:.1f}:")
        print(f"  Initial error (should be ~0): {init_error:.6f}")

        # Now train for just 10 epochs with small LR
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in data], dtype=torch.long).to(DEVICE)
        ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in data], dtype=torch.long).to(DEVICE)
        bs = torch.tensor([d[2] for d in data], dtype=torch.float32).to(DEVICE)

        bucket_factor = 100
        bs_bucketed = torch.round(bs * bucket_factor) / bucket_factor
        unique_bs = torch.unique(bs_bucketed)

        for epoch in range(10):
            optimizer.zero_grad()
            Q = model.build_Q()

            log_prob_total = 0
            for b_val in unique_bs:
                mask = (bs_bucketed == b_val)
                if mask.sum() == 0:
                    continue

                P = torch.matrix_exp(b_val * Q)
                batch_starts = starts_idx[mask]
                batch_ends = ends_idx[mask]
                probs = P[batch_starts, batch_ends]
                log_prob_total += torch.sum(torch.log(probs + 1e-9))

            loss = -log_prob_total / len(data)
            loss.backward()
            optimizer.step()

            if epoch % 3 == 0:
                Q_current = model.build_Q().detach().cpu().numpy()
                error = np.linalg.norm(Q_current - Q_true, 'fro') / np.linalg.norm(Q_true, 'fro')
                print(f"  Epoch {epoch:2d}: error = {error:.6f}, loss = {loss.item():.6f}")

        print(f"  → Does training DEGRADE the perfect initialization? {error > init_error}")
        print()


def visualize_rate_distributions(epistasis_levels=[0.0, 0.3, 0.7, 1.0], seed=42):
    """Visualize how rate distributions change with epistasis."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Rate Distribution Visualization")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, eps in enumerate(epistasis_levels):
        gt = GroundTruthProcess(seed=seed, epistasis=eps)
        Q = gt.Q_global

        # Extract off-diagonal Hamming-1 rates
        rates = []
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                    rates.append(Q[i, j])

        axes[idx].hist(rates, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'Epistasis = {eps:.1f}')
        axes[idx].set_xlabel('Rate')
        axes[idx].set_ylabel('Count')
        axes[idx].axvline(np.mean(rates), color='red', linestyle='--', label=f'Mean = {np.mean(rates):.3f}')
        axes[idx].axvline(np.median(rates), color='blue', linestyle='--', label=f'Median = {np.median(rates):.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/accounts/projects/yss/stephen.lu/ctmc/plots/rate_distributions_vs_epistasis.pdf', dpi=150, bbox_inches='tight')
    print("Plot saved to plots/rate_distributions_vs_epistasis.pdf")
    print()


if __name__ == '__main__':
    # Run all analyses
    analyze_ground_truth_difficulty()
    analyze_data_quality()
    test_identifiability()
    visualize_rate_distributions()

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
