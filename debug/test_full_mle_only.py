"""
Test ONLY Full MLE with unbiased data (min_mutations=0).

Goal: Verify Full MLE can achieve low error with enough unbiased data.
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

print("=" * 70)
print("TESTING FULL MLE WITH UNBIASED DATA")
print("=" * 70)

epistasis = 0.0
seed = 99  # Use different seed to avoid cached data
sample_sizes = [10000, 50000, 100000]

for n_samples in sample_sizes:
    print(f"\n{'='*70}")
    print(f"N_SAMPLES = {n_samples} (min_mutations=0, unbiased)")
    print(f"{'='*70}")

    # Generate ground truth
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

    # Generate fresh data with NO mutation filter
    print(f"Generating {n_samples} samples...")
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)

    # Data statistics
    transition_counts = np.zeros((N_STATES, N_STATES))
    time_spent = np.zeros(N_STATES)
    n_no_mutation = 0

    for start_seq, end_seq, b in data:
        i = seq_to_idx(tuple(start_seq.tolist()))
        j = seq_to_idx(tuple(end_seq.tolist()))
        transition_counts[i, j] += 1
        time_spent[i] += b

        if i == j:
            n_no_mutation += 1

    print(f"\nData Statistics:")
    print(f"  Total samples:              {len(data)}")
    print(f"  No-mutation samples (i→i):  {n_no_mutation} ({n_no_mutation/len(data):.1%})")
    print(f"  States visited:             {np.sum(time_spent > 0)}/256")

    # Hamming-1 coverage
    n_hamming1 = sum(1 for i in range(N_STATES) for j in range(N_STATES)
                     if i != j and hamming(SEQS[i], SEQS[j]) == 1)
    n_observed = sum(1 for i in range(N_STATES) for j in range(N_STATES)
                    if i != j and hamming(SEQS[i], SEQS[j]) == 1 and transition_counts[i, j] > 0)
    coverage = n_observed / n_hamming1

    print(f"\nTransition Coverage:")
    print(f"  Possible Hamming-1:         {n_hamming1}")
    print(f"  Observed:                   {n_observed}")
    print(f"  Coverage:                   {coverage:.1%}")

    # Unobserved transitions
    unobserved_hamming1 = 0
    unobserved_nonzero = 0
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                if transition_counts[i, j] == 0:
                    unobserved_hamming1 += 1
                    if abs(gt.Q_global[i, j]) > 1e-6:
                        unobserved_nonzero += 1

    print(f"  Unobserved:                 {unobserved_hamming1}")
    print(f"  Unobserved but nonzero:     {unobserved_nonzero}")

    # Compute Full MLE
    print(f"\nFull MLE Estimation:")
    Q_full_np = estimate_full_mle_q(data, gt, regularization=1e-6)
    Q_full = torch.tensor(Q_full_np, dtype=torch.float32)
    error = torch.norm(Q_full - Q_true) / torch.norm(Q_true)

    print(f"  Error: {error.item():.4f}")

    # Check if error improves
    if n_samples == sample_sizes[0]:
        first_error = error.item()
    else:
        improvement = first_error - error.item()
        print(f"  Improvement from {sample_sizes[0]}: {improvement:+.4f}")

print(f"\n{'='*70}")
print("EXPECTED: Error should decrease significantly with more data")
print("IF NOT: There may be a bug in Full MLE estimation")
print(f"{'='*70}")
