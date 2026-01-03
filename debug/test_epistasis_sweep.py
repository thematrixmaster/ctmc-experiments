"""
Test epistasis sweep with early stopping.

Expected behavior:
- Low epistasis (0.0): Factorized << Full MLE (factorized is correct model)
- High epistasis (1.0): Factorized >> Full MLE (can't capture context-dependence)
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

# Parameters - MUCH MORE DATA TO TEST COVERAGE HYPOTHESIS
epistasis_levels = [0.0, 0.3, 0.7, 1.0]
seed = 42
n_samples = 50000  # 6x more data - should give ~70-80% transition coverage
lr = 0.001         # Best from early stopping test
patience = 100
max_epochs = 2000

# Full MLE parameters (gradient-based with tuned early stopping)
# FIXED: Re-enable early stopping and reduce LR to prevent overfitting
full_mle_max_epochs = 500      # More epochs, let early stopping decide when to stop
full_mle_lr = 0.005            # Reduced from 0.01 (was too aggressive, caused overfitting)
full_mle_bucket_decimals = 2   # 2 decimals = ~100 unique values
full_mle_patience = 100        # Re-enabled: tolerate noisy error, wait for plateau
full_mle_tolerance = 0.005     # Increased tolerance (was 0.0001, too strict for noisy training)

print("=" * 70)
print("EPISTASIS SWEEP: Factorized (Early Stop) vs Full MLE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Samples per condition: {n_samples}")
print(f"  Full MLE max epochs: {full_mle_max_epochs} (early stop: patience={full_mle_patience})")
print(f"  Full MLE bucket decimals: {full_mle_bucket_decimals}")
print(f"  Factorized max epochs: {max_epochs} (early stop: patience={patience})")
print(f"\nEstimated time per epistasis level:")
print(f"  Full MLE: ~25 minutes (300 epochs × 5sec, no early stopping)")
print(f"  Factorized: ~2-5 minutes")
print(f"  Total per level: ~27-30 minutes")
print(f"  TOTAL FOR ALL {len(epistasis_levels)} LEVELS: ~110-120 minutes (~2 hours)")
print(f"\nNOTE: Early stopping DISABLED - was stopping too early before convergence!")
print("=" * 70)

results = []

for eps in epistasis_levels:
    print(f"\n{'='*70}")
    print(f"EPISTASIS = {eps}")
    print(f"{'='*70}")

    # Create ground truth
    gt = GroundTruthProcess(seed=seed, epistasis=eps)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

    # Load/generate data (NO mutation filter for unbiased samples)
    data = load_dataset(eps, seed, n_samples)
    if data is None:
        print(f"Generating {n_samples} samples (no mutation filter)...")
        data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)
        save_dataset(data, eps, seed, n_samples)
    else:
        print(f"Loaded {len(data)} samples from cache")

    # Quick data quality check
    transition_counts = np.zeros((N_STATES, N_STATES))
    for start_seq, end_seq, b in data:
        i = seq_to_idx(tuple(start_seq.tolist()))
        j = seq_to_idx(tuple(end_seq.tolist()))
        transition_counts[i, j] += 1
    n_hamming1 = sum(1 for i in range(N_STATES) for j in range(N_STATES)
                     if i != j and hamming(SEQS[i], SEQS[j]) == 1)
    n_observed = sum(1 for i in range(N_STATES) for j in range(N_STATES)
                    if i != j and hamming(SEQS[i], SEQS[j]) == 1 and transition_counts[i, j] > 0)
    print(f"   Transition coverage: {n_observed}/{n_hamming1} ({n_observed/n_hamming1:.1%})")

    # 1. Full MLE (gradient descent with early stopping to prevent overfitting)
    print(f"\n1. Full MLE (256×256, gradient descent with early stopping)...")
    model_full, history_full = train_full_mle_model(
        gt, data,
        epochs=full_mle_max_epochs,
        lr=full_mle_lr,
        bucket_decimals=full_mle_bucket_decimals,
        early_stop_patience=full_mle_patience,
        early_stop_tolerance=full_mle_tolerance,
        verbose=True
    )
    with torch.no_grad():
        Q_full = model_full.build_Q()
        error_full_final = torch.norm(Q_full - Q_true) / torch.norm(Q_true)

    # Show convergence stats
    epochs_trained = len(history_full['errors'])
    error_full_best = min(history_full['errors'])
    print(f"   Epochs trained: {epochs_trained}")
    print(f"   Best error: {error_full_best:.4f}")
    print(f"   Final error: {error_full_final.item():.4f}")

    # CRITICAL FIX: Use best error for fair comparison
    error_full = error_full_best
    if history_full['epoch_times']:
        avg_epoch_time = np.mean(history_full['epoch_times'])
        print(f"   Avg time/epoch: {avg_epoch_time:.2f}s")

    # 2. Factorized with early stopping
    print(f"\n2. Factorized model (with early stopping)...")
    model, history = train_model(
        gt, data,
        use_snr=False,
        epochs=max_epochs,
        lr=lr,
        early_stop_patience=patience,
        early_stop_tolerance=0.001,
        verbose=False  # Less verbose
    )

    with torch.no_grad():
        Q_factorized = model.build_global_Q()
        error_factorized = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)

    best_error = min(history['errors']) if history['errors'] else error_factorized.item()
    stopped_epoch = len(history['epochs']) if history['epochs'] else max_epochs

    print(f"   Best error: {best_error:.4f}")
    print(f"   Stopped at epoch: {stopped_epoch}")

    # 3. Comparison
    gap = best_error - error_full.item()
    print(f"\n3. Comparison:")
    print(f"   Full MLE:      {error_full.item():.4f}")
    print(f"   Factorized:    {best_error:.4f}")
    print(f"   Gap:           {gap:+.4f} ", end='')

    if gap < -0.1:
        print("(Factorized BETTER - expected at low epistasis)")
    elif gap > 0.1:
        print("(Full MLE BETTER - expected at high epistasis)")
    else:
        print("(Similar performance)")

    results.append({
        'epistasis': eps,
        'error_full': error_full.item(),
        'error_factorized': best_error,
        'gap': gap,
        'stopped_epoch': stopped_epoch
    })

# Summary plot
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Epistasis':<12} {'Full MLE':<12} {'Factorized':<12} {'Gap':<10} {'Epochs'}")
print("-" * 70)
for r in results:
    print(f"{r['epistasis']:<12.1f} {r['error_full']:<12.4f} {r['error_factorized']:<12.4f} "
          f"{r['gap']:<+10.4f} {r['stopped_epoch']}")

# Plot
plt.figure(figsize=(10, 6))

eps_vals = [r['epistasis'] for r in results]
errors_full = [r['error_full'] for r in results]
errors_fact = [r['error_factorized'] for r in results]

plt.plot(eps_vals, errors_full, 'o-', linewidth=2, markersize=10,
         color='orange', label='Full MLE (256×256)')
plt.plot(eps_vals, errors_fact, 's-', linewidth=2, markersize=10,
         color='blue', label='Factorized (early stop)')

plt.xlabel('Epistasis Strength', fontsize=12)
plt.ylabel('Relative Frobenius Error', fontsize=12)
plt.title('Model Error vs Epistasis (with Early Stopping)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(eps_vals)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/epistasis_sweep_early_stop.png', dpi=150)
print(f"\nSaved results/epistasis_sweep_early_stop.png")

# Analysis
print(f"\n{'='*70}")
print("EXPECTED BEHAVIOR CHECK")
print(f"{'='*70}")

# Check crossover point
low_eps_better = results[0]['gap'] < -0.1  # Factorized better at eps=0
high_eps_worse = results[-1]['gap'] > 0.1  # Full MLE better at eps=1

print(f"At epistasis=0.0: Factorized better than Full MLE? {low_eps_better}")
print(f"At epistasis=1.0: Full MLE better than Factorized? {high_eps_worse}")

if low_eps_better and high_eps_worse:
    print(f"\n✓ EXPECTED BEHAVIOR: Models cross over as epistasis increases")
elif low_eps_better:
    print(f"\n⚠ Factorized better at low epistasis (good), but gap not clear at high epistasis")
else:
    print(f"\n✗ UNEXPECTED: Check model training or early stopping settings")

# Find crossover point
for i in range(len(results) - 1):
    if results[i]['gap'] < 0 and results[i+1]['gap'] > 0:
        print(f"\nCrossover between epistasis {results[i]['epistasis']} and {results[i+1]['epistasis']}")
        break
