#!/usr/bin/env python3
"""
Quick verification of the data scarcity hypothesis.

This script runs a fast test to confirm:
1. Current setup (10k samples) has insufficient data
2. More data (30k samples) fixes the issue

Expected runtime: ~5-10 minutes
"""

from main import *
import numpy as np

def quick_test(n_samples, epistasis=0.0, seed=42):
    """Quick test with given sample size."""
    print(f"\nTesting with {n_samples} samples...")
    
    # Generate data
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)
    
    # Calculate coverage
    transition_counts = np.zeros((N_STATES, N_STATES))
    for start_seq, end_seq, b in data:
        i = seq_to_idx(tuple(start_seq.tolist()))
        j = seq_to_idx(tuple(end_seq.tolist()))
        transition_counts[i, j] += 1
    
    hamming1_counts = []
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                hamming1_counts.append(transition_counts[i, j])
    
    hamming1_counts = np.array(hamming1_counts)
    coverage = np.sum(hamming1_counts > 0) / len(hamming1_counts) * 100
    well_observed = np.sum(hamming1_counts >= 20) / len(hamming1_counts) * 100
    
    print(f"  Coverage: {coverage:.1f}% observed, {well_observed:.1f}% well-observed (≥20)")
    print(f"  Mean observations per transition: {hamming1_counts.mean():.1f}")
    
    # Train factorized (fast)
    print(f"  Training factorized model...", end='', flush=True)
    model_fact, _ = train_model(gt, data, use_snr=False, epochs=500, lr=0.01,
                                early_stop_patience=50, early_stop_tolerance=0.001, 
                                verbose=False)
    Q_fact = model_fact.build_global_Q()
    error_fact = torch.norm(Q_fact - Q_true) / torch.norm(Q_true)
    print(f" Error: {error_fact.item():.4f}")
    
    # Train full MLE (slower)
    print(f"  Training full MLE model...", end='', flush=True)
    model_full, _ = train_full_mle_model(gt, data, epochs=300, lr=0.01, 
                                         verbose=False, early_stop_patience=50,
                                         early_stop_tolerance=0.001)
    Q_full = model_full.build_Q()
    error_full = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
    print(f" Error: {error_full.item():.4f}")
    
    # Comparison
    gap = error_full.item() - error_fact.item()
    print(f"\n  Gap (Full MLE - Factorized): {gap:+.4f}")
    
    if gap > 0.05:
        status = "⚠ Full MLE significantly worse"
    elif gap > 0.02:
        status = "⚠ Full MLE slightly worse"
    elif gap < -0.02:
        status = "✓ Full MLE better"
    else:
        status = "✓ Similar performance"
    
    print(f"  Status: {status}")
    
    return {
        'n_samples': n_samples,
        'coverage': coverage,
        'well_observed': well_observed,
        'mean_count': hamming1_counts.mean(),
        'error_fact': error_fact.item(),
        'error_full': error_full.item(),
        'gap': gap,
        'status': status
    }

if __name__ == "__main__":
    print("="*70)
    print("QUICK VERIFICATION: Data Scarcity Hypothesis")
    print("="*70)
    print("\nThis script tests whether increasing data fixes the issue.")
    print("Testing at epistasis=0.0 where both models should be similar.")
    print("\nExpected:")
    print("  - 10k samples: Full MLE worse (insufficient data)")
    print("  - 30k samples: Full MLE matches factorized (sufficient data)")
    
    results = []
    
    # Test 1: Current setup (10k samples)
    print(f"\n{'='*70}")
    print("TEST 1: Current Setup (10k samples)")
    print(f"{'='*70}")
    r1 = quick_test(10000, epistasis=0.0, seed=42)
    results.append(r1)
    
    # Test 2: Proposed fix (30k samples)
    print(f"\n{'='*70}")
    print("TEST 2: Proposed Fix (30k samples)")
    print(f"{'='*70}")
    r2 = quick_test(30000, epistasis=0.0, seed=42)
    results.append(r2)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Samples':<10} {'Coverage':<12} {'Well-Obs':<12} {'Factorized':<12} {'Full MLE':<12} {'Gap':<10} {'Status'}")
    print("-"*90)
    
    for r in results:
        print(f"{r['n_samples']:<10} {r['coverage']:>6.1f}%     {r['well_observed']:>6.1f}%     "
              f"{r['error_fact']:<12.4f} {r['error_full']:<12.4f} {r['gap']:>+8.4f}  {r['status']}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    improvement = r1['gap'] - r2['gap']
    
    if r2['gap'] < 0.02 and r1['gap'] > 0.05:
        print("\n✓ HYPOTHESIS CONFIRMED!")
        print(f"  - With 10k samples: Full MLE is {r1['gap']:.3f} worse")
        print(f"  - With 30k samples: Full MLE is {r2['gap']:.3f} worse")
        print(f"  - Improvement: {improvement:.3f}")
        print(f"\n  Increasing data from 10k to 30k fixes the issue!")
        print(f"  The full MLE was underperforming due to data scarcity.")
    elif r2['gap'] < r1['gap']:
        print(f"\n⚠ PARTIAL IMPROVEMENT")
        print(f"  - Gap reduced from {r1['gap']:.3f} to {r2['gap']:.3f}")
        print(f"  - Improvement: {improvement:.3f}")
        print(f"\n  More data helps, but may need even more samples")
        print(f"  or additional fixes (lower LR, regularization).")
    else:
        print(f"\n✗ HYPOTHESIS NOT CONFIRMED")
        print(f"  - Gap did not improve with more data")
        print(f"  - May indicate other issues beyond data scarcity")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    
    if r2['gap'] < 0.02:
        print("\n1. Update run_experiment_0.py to use n_samples=30000")
        print("2. Re-run full experiment with 3 replicates")
        print("3. Verify that full MLE beats factorized at high epistasis")
    else:
        print("\n1. Try even more data (50k samples)")
        print("2. Add L2 regularization (weight_decay=1e-4)")
        print("3. Reduce learning rate (lr=0.005)")
        print("4. Run test_fixes.py for detailed comparison")


