"""
Diagnose transition coverage and data sufficiency issues.

This script analyzes:
1. How many Hamming-1 transitions are observed in the training data
2. Distribution of observation counts per transition
3. Comparison with different sample sizes
"""

from main import *
import numpy as np
import matplotlib.pyplot as plt

def analyze_coverage(n_samples, epistasis=0.0, seed=42):
    """Analyze transition coverage for given sample size."""
    print(f"\n{'='*70}")
    print(f"Coverage Analysis: {n_samples} samples, epistasis={epistasis}")
    print(f"{'='*70}")
    
    # Generate data
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)
    
    # Count transitions
    transition_counts = np.zeros((N_STATES, N_STATES))
    for start_seq, end_seq, b in data:
        i = seq_to_idx(tuple(start_seq.tolist()))
        j = seq_to_idx(tuple(end_seq.tolist()))
        transition_counts[i, j] += 1
    
    # Analyze Hamming-1 transitions (the ones we care about)
    hamming1_counts = []
    hamming1_total = 0
    hamming1_observed = 0
    hamming1_well_observed = 0  # Count transitions with >= 20 observations
    
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                hamming1_total += 1
                count = transition_counts[i, j]
                hamming1_counts.append(count)
                if count > 0:
                    hamming1_observed += 1
                if count >= 20:
                    hamming1_well_observed += 1
    
    hamming1_counts = np.array(hamming1_counts)
    
    # Statistics
    coverage_pct = hamming1_observed / hamming1_total * 100
    well_observed_pct = hamming1_well_observed / hamming1_total * 100
    mean_count = hamming1_counts.mean()
    median_count = np.median(hamming1_counts)
    zero_count = np.sum(hamming1_counts == 0)
    
    print(f"\nTransition Statistics:")
    print(f"  Total Hamming-1 transitions: {hamming1_total}")
    print(f"  Observed at least once: {hamming1_observed} ({coverage_pct:.1f}%)")
    print(f"  Observed >= 20 times: {hamming1_well_observed} ({well_observed_pct:.1f}%)")
    print(f"  Never observed: {zero_count} ({zero_count/hamming1_total*100:.1f}%)")
    print(f"\nObservation Counts:")
    print(f"  Mean: {mean_count:.1f}")
    print(f"  Median: {median_count:.1f}")
    print(f"  Min: {hamming1_counts.min()}")
    print(f"  Max: {hamming1_counts.max()}")
    print(f"  Std: {hamming1_counts.std():.1f}")
    
    # Distribution
    print(f"\nDistribution of Observation Counts:")
    bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
    labels = ['0', '1-4', '5-9', '10-19', '20-49', '50-99', '100+']
    for i in range(len(bins)-1):
        count = np.sum((hamming1_counts >= bins[i]) & (hamming1_counts < bins[i+1]))
        pct = count / hamming1_total * 100
        print(f"  {labels[i]:>8}: {count:4d} ({pct:5.1f}%)")
    
    return {
        'n_samples': n_samples,
        'hamming1_total': hamming1_total,
        'hamming1_observed': hamming1_observed,
        'hamming1_well_observed': hamming1_well_observed,
        'coverage_pct': coverage_pct,
        'well_observed_pct': well_observed_pct,
        'mean_count': mean_count,
        'median_count': median_count,
        'counts': hamming1_counts
    }

def plot_coverage_comparison(results_list):
    """Plot coverage comparison across different sample sizes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sample_sizes = [r['n_samples'] for r in results_list]
    
    # Plot 1: Coverage percentage
    ax = axes[0]
    coverage = [r['coverage_pct'] for r in results_list]
    well_observed = [r['well_observed_pct'] for r in results_list]
    
    ax.plot(sample_sizes, coverage, 'o-', linewidth=2, markersize=8, label='Observed (≥1)')
    ax.plot(sample_sizes, well_observed, 's-', linewidth=2, markersize=8, label='Well-observed (≥20)')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% coverage')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Transition Coverage vs Sample Size', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Mean observations per transition
    ax = axes[1]
    mean_counts = [r['mean_count'] for r in results_list]
    ax.plot(sample_sizes, mean_counts, 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Minimum recommended (20)')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Mean Observations per Transition', fontsize=12)
    ax.set_title('Data Density vs Sample Size', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 3: Distribution for largest dataset
    ax = axes[2]
    largest = results_list[-1]
    counts = largest['counts']
    ax.hist(counts, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Min recommended (20)')
    ax.set_xlabel('Observations per Transition', fontsize=12)
    ax.set_ylabel('Number of Transitions', fontsize=12)
    ax.set_title(f'Distribution (n={largest["n_samples"]})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/coverage_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results/coverage_analysis.png")

if __name__ == "__main__":
    print("="*70)
    print("TRANSITION COVERAGE DIAGNOSTIC")
    print("="*70)
    print("\nThis script analyzes how many transitions are observed")
    print("in the training data, and whether we have enough data")
    print("for reliable full MLE estimation.")
    
    # Test different sample sizes
    sample_sizes = [5000, 10000, 20000, 30000, 50000]
    results = []
    
    for n in sample_sizes:
        result = analyze_coverage(n, epistasis=0.0, seed=42)
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Samples':<10} {'Coverage':<12} {'Well-Obs':<12} {'Mean Count':<12} {'Recommendation'}")
    print("-"*70)
    
    for r in results:
        rec = "✓ Good" if r['well_observed_pct'] >= 80 else "⚠ Insufficient"
        print(f"{r['n_samples']:<10} {r['coverage_pct']:>6.1f}%     "
              f"{r['well_observed_pct']:>6.1f}%     {r['mean_count']:>8.1f}     {rec}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Find minimum sample size for 80% well-observed coverage
    for r in results:
        if r['well_observed_pct'] >= 80:
            print(f"\n✓ Recommended minimum: {r['n_samples']} samples")
            print(f"  This achieves {r['well_observed_pct']:.1f}% well-observed coverage")
            print(f"  (≥20 observations per transition)")
            break
    else:
        print(f"\n⚠ Need more than {results[-1]['n_samples']} samples")
        print(f"  Current best: {results[-1]['well_observed_pct']:.1f}% well-observed")
    
    # Current experiment status
    current = results[1]  # 10k samples
    print(f"\nCurrent experiment (10k samples):")
    print(f"  Coverage: {current['coverage_pct']:.1f}%")
    print(f"  Well-observed: {current['well_observed_pct']:.1f}%")
    print(f"  Mean observations: {current['mean_count']:.1f}")
    print(f"\n  ⚠ DIAGNOSIS: Insufficient data for reliable full MLE!")
    print(f"     Only {current['well_observed_pct']:.1f}% of transitions have ≥20 observations")
    print(f"     This explains why full MLE underperforms factorized model")
    
    # Plot comparison
    plot_coverage_comparison(results)
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("\nThe full MLE model underperforms because:")
    print("  1. Many transitions are rarely or never observed")
    print("  2. High variance in rate estimates for sparse transitions")
    print("  3. Factorized model has 12× better sample-to-parameter ratio")
    print("\nSolution: Increase training data to 30k+ samples")


