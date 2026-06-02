"""
Investigate why Full MLE error increases at high epistasis values.

Key questions:
1. How does the ground truth Q matrix change with epistasis?
2. Are the rate matrices becoming pathological (very small/large rates)?
3. Is the training data still informative at high epistasis?
4. What happens to mutation rates and branch length distribution?
"""

import sys
sys.path.append('/accounts/projects/yss/stephen.lu/ctmc')

from main import *
import matplotlib.pyplot as plt

print("=" * 80)
print("INVESTIGATION: High Epistasis Effects on Q Matrix and Training")
print("=" * 80)

# Test different epistasis levels
epistasis_levels = [0.0, 1.0, 3.0, 6.0, 9.0, 12.0]
seed = 42

print("\n" + "=" * 80)
print("PART 1: Ground Truth Q Matrix Analysis")
print("=" * 80)

print(f"\n{'Epistasis':<12} {'Frob Norm':<12} {'Mean Rate':<12} {'Min Rate':<12} {'Max Rate':<12} {'Rate Ratio':<12} {'Condition #':<15}")
print("-" * 100)

stats_by_eps = {}

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
    condition_number = np.linalg.cond(Q)

    stats = {
        'epistasis': eps,
        'frob_norm': frobenius_norm,
        'mean': np.mean(rates),
        'std': np.std(rates),
        'min': np.min(rates),
        'max': np.max(rates),
        'ratio': np.max(rates) / np.min(rates),
        'cond': condition_number
    }
    stats_by_eps[eps] = stats

    print(f"{eps:<12.1f} {stats['frob_norm']:<12.2f} {stats['mean']:<12.4f} {stats['min']:<12.4f} "
          f"{stats['max']:<12.4f} {stats['ratio']:<12.1f}× {stats['cond']:<15.2e}")

print("\n" + "=" * 80)
print("PART 2: Mutation Rate Analysis")
print("=" * 80)

print(f"\nWith branch length b=1.0, expected mutations per site:")
print(f"{'Epistasis':<12} {'E[mut/site]':<15} {'E[mut/seq]':<15} {'Status':<20}")
print("-" * 70)

for eps in epistasis_levels:
    gt = GroundTruthProcess(seed=seed, epistasis=eps)
    mean_rate = stats_by_eps[eps]['mean']

    expected_mut_per_site = 1.0 * mean_rate
    expected_mut_per_seq = 1.0 * mean_rate * L

    status = "✓ Good" if 0.3 <= expected_mut_per_site <= 2.0 else "⚠ Problematic"

    print(f"{eps:<12.1f} {expected_mut_per_site:<15.3f} {expected_mut_per_seq:<15.3f} {status:<20}")

print("\n" + "=" * 80)
print("PART 3: Data Informativeness Test")
print("=" * 80)

print(f"\nGenerating 10k samples with lambda=1.0 for each epistasis level...")
print(f"{'Epistasis':<12} {'Mean b':<12} {'Mean mut':<12} {'Coverage':<15} {'Samples/param':<20}")
print("-" * 80)

for eps in epistasis_levels:
    gt = GroundTruthProcess(seed=seed, epistasis=eps)
    data = gt.generate_data(10000, lambda_param=1.0, min_mutations=0)

    # Analyze data
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

    # Count observed Hamming-1 transitions
    hamming1_transitions = []
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                hamming1_transitions.append((i, j))

    observed = sum(1 for (i, j) in hamming1_transitions if transition_counts[i, j] > 0)
    coverage = observed / len(hamming1_transitions)

    # Full MLE has 576 parameters
    samples_per_param = len(data) / 576

    print(f"{eps:<12.1f} {np.mean(branch_lengths):<12.3f} {np.mean(mutations):<12.3f} "
          f"{coverage:<15.1%} {samples_per_param:<20.1f}")

print("\n" + "=" * 80)
print("PART 4: Rate Distribution Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, eps in enumerate(epistasis_levels):
    gt = GroundTruthProcess(seed=seed, epistasis=eps)
    Q = gt.Q_global

    # Extract rates
    rates = []
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                rates.append(Q[i, j])

    # Plot histogram
    axes[idx].hist(rates, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[idx].set_xlabel('Rate', fontsize=11)
    axes[idx].set_ylabel('Count', fontsize=11)
    axes[idx].set_title(f'Epistasis = {eps:.1f}', fontsize=12)
    axes[idx].axvline(np.mean(rates), color='red', linestyle='--', linewidth=2,
                     label=f'Mean = {np.mean(rates):.3f}')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

    # Add text with key stats
    text = f'Min: {np.min(rates):.3f}\nMax: {np.max(rates):.3f}\nRatio: {np.max(rates)/np.min(rates):.1f}×'
    axes[idx].text(0.98, 0.97, text, transform=axes[idx].transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/high_epistasis_investigation.pdf', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to: plots/high_epistasis_investigation.pdf")

print("\n" + "=" * 80)
print("PART 5: Eigenvalue Analysis")
print("=" * 80)

print(f"\nChecking eigenvalue spectrum of Q matrices:")
print(f"{'Epistasis':<12} {'Max |eigenval|':<18} {'Min |eigenval|':<18} {'Spectral gap':<15}")
print("-" * 70)

for eps in epistasis_levels:
    gt = GroundTruthProcess(seed=seed, epistasis=eps)
    Q = gt.Q_global

    eigenvalues = np.linalg.eigvals(Q)
    eigenvalues_sorted = sorted(np.abs(eigenvalues), reverse=True)

    max_eig = eigenvalues_sorted[0]
    min_nonzero_eig = eigenvalues_sorted[-2] if len(eigenvalues_sorted) > 1 else eigenvalues_sorted[-1]
    spectral_gap = eigenvalues_sorted[0] - eigenvalues_sorted[1]

    print(f"{eps:<12.1f} {max_eig:<18.4f} {min_nonzero_eig:<18.6f} {spectral_gap:<15.4f}")

print("\n" + "=" * 80)
print("DIAGNOSIS & RECOMMENDATIONS")
print("=" * 80)

# Analyze trends
high_eps_stats = stats_by_eps[12.0]
low_eps_stats = stats_by_eps[0.0]

print(f"\nKey observations:")

# Check if rates are becoming too spread out
if high_eps_stats['ratio'] > 100:
    print(f"  ⚠ ISSUE 1: Rate ratio = {high_eps_stats['ratio']:.1f}× at epistasis=12.0")
    print(f"    → Some rates are 100× larger than others")
    print(f"    → This creates numerical instability in optimization")
    print(f"    → Solution: Use log-parameterization or rate normalization")

# Check if condition number is too high
if high_eps_stats['cond'] > 1e15:
    print(f"\n  ⚠ ISSUE 2: Condition number = {high_eps_stats['cond']:.2e} at epistasis=12.0")
    print(f"    → Matrix is nearly singular (ill-conditioned)")
    print(f"    → Gradient descent struggles with ill-conditioned problems")
    print(f"    → Solution: Add stronger regularization or use preconditioned optimizer")

# Check if mean rate changed significantly
mean_change = abs(high_eps_stats['mean'] - low_eps_stats['mean']) / low_eps_stats['mean']
if mean_change > 0.1:
    print(f"\n  ⚠ ISSUE 3: Mean rate changed by {mean_change*100:.1f}%")
    print(f"    → From {low_eps_stats['mean']:.4f} (eps=0) to {high_eps_stats['mean']:.4f} (eps=12)")
    print(f"    → Branch length distribution may not be optimal anymore")
    print(f"    → Solution: Adjust lambda parameter or use adaptive branch lengths")

print(f"\n" + "=" * 80)
print("PROPOSED SOLUTIONS")
print("=" * 80)

print(f"""
1. **Normalize the epistasis effect**:
   - Current: modifier uses exp(epistasis * hash * scale)
   - Problem: At high epistasis, this creates very large/small rates
   - Fix: Cap the epistasis effect or use a different parameterization

2. **Use log-space optimization for Full MLE**:
   - Current: Optimize rates directly
   - Problem: Rates span multiple orders of magnitude
   - Fix: Optimize log(rates) instead of rates

3. **Increase regularization at high epistasis**:
   - Current: weight_decay = 0.0001 (constant)
   - Problem: Not strong enough for ill-conditioned Q
   - Fix: Scale weight_decay with epistasis level

4. **Adjust branch length distribution**:
   - Current: lambda = 1.0 (fixed)
   - Problem: May not be optimal for all epistasis levels
   - Fix: Scale lambda inversely with mean rate

5. **Use better optimizer**:
   - Current: Adam with fixed learning rate
   - Problem: Struggles with ill-conditioned problems
   - Fix: Use second-order optimizer (L-BFGS) or adaptive learning rate

RECOMMENDED IMMEDIATE FIX:
Modify the epistasis parameterization to bound the rate variation.
Instead of unbounded exp(), use a bounded transformation like tanh().
""")

print("\nInvestigation complete!")
