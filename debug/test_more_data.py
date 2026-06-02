"""
Quick test: Does more data help the full MLE model?

Just trains full MLE at epistasis=0.0 with different sample sizes.
Runtime: ~10-15 minutes

Expected: With 30k samples, full MLE should match factorized (~0.075)
"""

from main import *
import time

def test_full_mle_with_data(n_samples, epistasis=0.0, seed=42):
    """Test full MLE with given sample size."""
    print(f"\n{'='*70}")
    print(f"Testing Full MLE with {n_samples} samples (epistasis={epistasis})")
    print(f"{'='*70}")
    
    # Generate data
    print(f"\n1. Generating ground truth and data...")
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    
    start_time = time.time()
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)
    gen_time = time.time() - start_time
    print(f"   Generated {len(data)} samples in {gen_time:.1f}s")
    
    # Quick coverage stats
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
    observed = np.sum(hamming1_counts > 0)
    well_observed = np.sum(hamming1_counts >= 20)
    
    print(f"\n2. Data coverage statistics:")
    print(f"   Hamming-1 transitions: 576 total")
    print(f"   Observed (≥1):  {observed} ({observed/576*100:.1f}%)")
    print(f"   Well-obs (≥20): {well_observed} ({well_observed/576*100:.1f}%)")
    print(f"   Mean count: {hamming1_counts.mean():.1f}")
    print(f"   Samples per parameter: {n_samples/576:.1f}")
    
    # Train full MLE
    print(f"\n3. Training Full MLE model...")
    print(f"   Parameters: lr=0.01, max_epochs=500, patience=50")
    
    start_time = time.time()
    model_full, history = train_full_mle_model(
        gt, data,
        epochs=500,
        lr=0.01,
        verbose=True,  # Show progress
        early_stop_patience=50,
        early_stop_tolerance=0.001
    )
    train_time = time.time() - start_time
    
    Q_full = model_full.build_Q()
    error_full = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
    
    print(f"\n4. Results:")
    print(f"   Training time: {train_time/60:.1f} minutes")
    print(f"   Epochs trained: {len(history['errors'])}")
    print(f"   Final error: {error_full.item():.4f}")
    
    if history['errors']:
        best_error = min(history['errors'])
        print(f"   Best error: {best_error:.4f}")
    
    return {
        'n_samples': n_samples,
        'error': error_full.item(),
        'best_error': min(history['errors']) if history['errors'] else error_full.item(),
        'epochs': len(history['errors']),
        'train_time': train_time,
        'coverage': observed/576*100,
        'well_observed': well_observed/576*100,
        'mean_count': hamming1_counts.mean()
    }

if __name__ == "__main__":
    print("="*70)
    print("QUICK TEST: Does More Data Help Full MLE?")
    print("="*70)
    print("\nGoal: Verify that full MLE matches factorized (~0.075) with more data")
    print("Testing at epistasis=0.0 (no epistasis, both models should be similar)")
    
    # Reference: factorized model error from logs
    FACTORIZED_ERROR = 0.075  # From experiment logs at epistasis=0.0
    
    print(f"\nReference (from previous experiment):")
    print(f"  Factorized model error: {FACTORIZED_ERROR:.4f}")
    print(f"  Full MLE error (10k):   0.1724 (WORSE)")
    
    print(f"\nWe will test full MLE with:")
    print(f"  - 10k samples (baseline, should match previous ~0.172)")
    print(f"  - 30k samples (proposed fix, should match factorized ~0.075)")
    
    results = []
    
    # Test 1: 10k samples (baseline - should match previous result)
    print(f"\n\n{'#'*70}")
    print("# TEST 1: Baseline (10k samples)")
    print(f"{'#'*70}")
    r1 = test_full_mle_with_data(10000, epistasis=0.0, seed=42)
    results.append(r1)
    
    print(f"\nBaseline check:")
    if abs(r1['best_error'] - 0.172) < 0.02:
        print(f"  ✓ Matches previous result (~0.172)")
    else:
        print(f"  ⚠ Different from previous (was 0.172, now {r1['best_error']:.4f})")
    
    # Test 2: 30k samples (should improve)
    print(f"\n\n{'#'*70}")
    print("# TEST 2: With More Data (30k samples)")
    print(f"{'#'*70}")
    r2 = test_full_mle_with_data(30000, epistasis=0.0, seed=42)
    results.append(r2)
    
    # Analysis
    print(f"\n\n{'='*70}")
    print("FINAL ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n{'Samples':<10} {'Coverage':<12} {'Well-Obs':<12} {'Best Error':<12} {'vs Factorized':<15}")
    print("-"*70)
    
    for r in results:
        gap = r['best_error'] - FACTORIZED_ERROR
        if gap < 0.02:
            status = "✓ Matches"
        elif gap < 0.05:
            status = "~ Close"
        else:
            status = "✗ Worse"
        
        print(f"{r['n_samples']:<10} {r['coverage']:>6.1f}%     {r['well_observed']:>6.1f}%     "
              f"{r['best_error']:<12.4f} {gap:>+6.4f} ({status})")
    
    # Conclusion
    improvement = r1['best_error'] - r2['best_error']
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    print(f"\nError change with 3× more data:")
    print(f"  10k samples: {r1['best_error']:.4f}")
    print(f"  30k samples: {r2['best_error']:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement/r1['best_error']*100:.1f}% reduction)")
    
    if r2['best_error'] < FACTORIZED_ERROR + 0.02:
        print(f"\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"\nWith 30k samples, full MLE achieves {r2['best_error']:.4f},")
        print(f"which matches the factorized model ({FACTORIZED_ERROR:.4f})!")
        print(f"\nThis confirms the hypothesis:")
        print(f"  - The issue WAS insufficient training data")
        print(f"  - Increasing from 10k to 30k samples fixes it")
        print(f"\nRecommendation: Re-run full experiment with n_samples=30000")
        
    elif improvement > 0.03:
        print(f"\n⚠ PARTIAL SUCCESS")
        print(f"\nFull MLE improved by {improvement:.4f}, but still worse than factorized.")
        print(f"Gap to factorized: {r2['best_error'] - FACTORIZED_ERROR:.4f}")
        print(f"\nPossible next steps:")
        print(f"  1. Try even more data (50k samples)")
        print(f"  2. Try lower learning rate (0.005 instead of 0.01)")
        print(f"  3. Add L2 regularization (weight_decay=1e-4)")
        
    else:
        print(f"\n✗ NO IMPROVEMENT")
        print(f"\nMore data did not help. The issue may not be data scarcity.")
        print(f"\nPossible issues to investigate:")
        print(f"  1. Training dynamics (learning rate too high?)")
        print(f"  2. Optimization (stuck in bad local minimum?)")
        print(f"  3. Model architecture (parameterization issue?)")
        print(f"\nTry running test_fixes.py for comprehensive debugging")
    
    # Time estimate for full experiment
    if r2['best_error'] < FACTORIZED_ERROR + 0.02:
        total_time = r2['train_time'] * 3 * 4  # 3 replicates, 4 epistasis levels
        print(f"\n{'='*70}")
        print("TIME ESTIMATE FOR FULL EXPERIMENT")
        print(f"{'='*70}")
        print(f"  Full MLE training time: ~{r2['train_time']/60:.1f} min per condition")
        print(f"  Total conditions: 3 replicates × 4 epistasis = 12")
        print(f"  Estimated total: ~{total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"  (Plus factorized model training, which is faster)")


