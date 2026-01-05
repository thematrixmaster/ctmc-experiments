"""
Verify that the new epistasis implementation preserves mutation rates.
"""

import sys
sys.path.append('/accounts/projects/yss/stephen.lu/ctmc')

from main import *
import matplotlib.pyplot as plt

print("=" * 80)
print("Verifying Fixed Epistasis Implementation")
print("=" * 80)

epistasis_levels = [0.0, 0.3, 0.7, 1.0]
seed = 42

print("\n1. Rate Statistics Across Epistasis Levels:")
print("-" * 80)
print(f"{'Epistasis':<12} {'Frob Norm':<12} {'Mean Rate':<12} {'Std Rate':<12} {'Min Rate':<12} {'Max Rate':<12}")
print("-" * 80)

all_stats = []
for eps in epistasis_levels:
    gt = GroundTruthProcess(seed=seed, epistasis=eps)
    Q = gt.Q_global

    # Extract off-diagonal Hamming-1 rates
    rates = []
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                rates.append(Q[i, j])

    rates = np.array(rates)
    frobenius_norm = np.linalg.norm(Q, 'fro')

    stats = {
        'epistasis': eps,
        'frob_norm': frobenius_norm,
        'mean': np.mean(rates),
        'std': np.std(rates),
        'min': np.min(rates),
        'max': np.max(rates),
        'cv': np.std(rates) / np.mean(rates)
    }
    all_stats.append(stats)

    print(f"{eps:<12.1f} {stats['frob_norm']:<12.2f} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f}")

print("\n2. Mean Rate Stability Check:")
print("-" * 80)
baseline_mean = all_stats[0]['mean']
print(f"Baseline mean rate (eps=0.0): {baseline_mean:.4f}")
print()
print(f"{'Epistasis':<12} {'Mean Rate':<12} {'% Change':<12} {'Status':<20}")
print("-" * 80)
for stats in all_stats:
    pct_change = 100 * (stats['mean'] - baseline_mean) / baseline_mean
    status = "✓ GOOD" if abs(pct_change) < 5 else "✗ BAD (mean shifted!)"
    print(f"{stats['epistasis']:<12.1f} {stats['mean']:<12.4f} {pct_change:>+11.2f}% {status:<20}")

print("\n3. Frobenius Norm Stability Check:")
print("-" * 80)
baseline_norm = all_stats[0]['frob_norm']
print(f"Baseline Frobenius norm (eps=0.0): {baseline_norm:.2f}")
print()
print(f"{'Epistasis':<12} {'Frob Norm':<12} {'% Change':<12} {'Status':<20}")
print("-" * 80)
for stats in all_stats:
    pct_change = 100 * (stats['frob_norm'] - baseline_norm) / baseline_norm
    status = "✓ GOOD" if abs(pct_change) < 10 else "✗ BAD (norm shifted!)"
    print(f"{stats['epistasis']:<12.1f} {stats['frob_norm']:<12.2f} {pct_change:>+11.2f}% {status:<20}")

print("\n4. Coefficient of Variation (Heterogeneity Measure):")
print("-" * 80)
print("Expected: CV should INCREASE with epistasis (more context-dependency)")
print()
print(f"{'Epistasis':<12} {'CV (std/mean)':<15} {'Status':<20}")
print("-" * 80)
for stats in all_stats:
    status = "✓ GOOD" if stats['cv'] >= stats['epistasis'] * 0.3 else "→ Low variation"
    print(f"{stats['epistasis']:<12.1f} {stats['cv']:<15.4f} {status:<20}")

print("\n5. Mutation Rate Verification (Data-Based):")
print("-" * 80)
print("Testing: Does expected mutations/site ≈ branch length?")
print()
print(f"{'Epistasis':<12} {'b':<8} {'E[mut]':<12} {'E[mut/site]':<15} {'Expected':<12}")
print("-" * 80)

for eps in epistasis_levels:
    gt = GroundTruthProcess(seed=seed, epistasis=eps)

    # Test with b=1.0
    b_test = 1.0
    n_samples = 5000

    mutation_counts = []
    for _ in range(n_samples):
        start_idx = np.random.randint(0, N_STATES)
        P = gt.get_P(b_test)
        end_idx = np.random.choice(N_STATES, p=P[start_idx])
        n_mutations = hamming(SEQS[start_idx], SEQS[end_idx])
        mutation_counts.append(n_mutations)

    mean_mutations = np.mean(mutation_counts)
    expected_per_site = mean_mutations / L

    status = "✓" if abs(expected_per_site - b_test) < 0.2 else "✗"
    print(f"{eps:<12.1f} {b_test:<8.1f} {mean_mutations:<12.2f} {expected_per_site:<15.2f} {b_test:<12.1f} {status}")

print("\n6. Visualization:")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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

    axes[idx].hist(rates, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[idx].set_title(f'Epistasis = {eps:.1f}', fontsize=14)
    axes[idx].set_xlabel('Rate', fontsize=12)
    axes[idx].set_ylabel('Count', fontsize=12)

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    axes[idx].axvline(mean_rate, color='red', linestyle='--', linewidth=2,
                     label=f'Mean = {mean_rate:.3f}')
    axes[idx].axvline(mean_rate - std_rate, color='orange', linestyle=':', linewidth=1.5,
                     label=f'±1 std')
    axes[idx].axvline(mean_rate + std_rate, color='orange', linestyle=':', linewidth=1.5)

    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/accounts/projects/yss/stephen.lu/ctmc/plots/fixed_epistasis_verification.pdf',
            dpi=150, bbox_inches='tight')
print("Plot saved to: plots/fixed_epistasis_verification.pdf")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\nSummary:")
print("✓ Mean rate should remain stable (within ±5%)")
print("✓ Frobenius norm should remain stable (within ±10%)")
print("✓ CV should increase with epistasis (more heterogeneity)")
print("✓ Mutation rate should match branch length")
