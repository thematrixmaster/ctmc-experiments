"""
Diagnose why Full MLE error is stuck at ~1.1 even with 50K samples.

Check:
1. Data coverage across states
2. Transition count distribution
3. Time spent distribution
4. Effect of regularization
5. Comparison to ground truth structure
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

print("=" * 70)
print("DIAGNOSING FULL MLE PERFORMANCE")
print("=" * 70)

epistasis = 0.0
seed = 42

# Test with increasing sample sizes
sample_sizes = [5000, 10000, 20000, 50000]

for n_samples in sample_sizes:
    print(f"\n{'='*70}")
    print(f"N_SAMPLES = {n_samples}")
    print(f"{'='*70}")

    # Generate/load data
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

    data = load_dataset(epistasis, seed, n_samples)
    if data is None:
        print(f"Generating {n_samples} samples...")
        data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=1)
        save_dataset(data, epistasis, seed, n_samples)
    else:
        print(f"Loaded {len(data)} samples from cache")

    # Compute statistics
    transition_counts = np.zeros((N_STATES, N_STATES))
    time_spent = np.zeros(N_STATES)

    for start_seq, end_seq, b in data:
        i = seq_to_idx(tuple(start_seq.tolist()))
        j = seq_to_idx(tuple(end_seq.tolist()))
        transition_counts[i, j] += 1
        time_spent[i] += b

    # Coverage statistics
    states_visited = np.sum(time_spent > 0)

    # Count Hamming-1 transitions
    n_possible_transitions = 0
    n_observed_transitions = 0
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                n_possible_transitions += 1
                if transition_counts[i, j] > 0:
                    n_observed_transitions += 1

    coverage = n_observed_transitions / n_possible_transitions

    print(f"\nData Coverage:")
    print(f"  States visited:                {states_visited}/{N_STATES} ({states_visited/N_STATES:.1%})")
    print(f"  Possible Hamming-1 transitions: {n_possible_transitions}")
    print(f"  Observed transitions:           {n_observed_transitions}")
    print(f"  Coverage:                       {coverage:.1%}")

    # Time distribution
    print(f"\nTime Distribution:")
    print(f"  Total time:        {time_spent.sum():.2f}")
    print(f"  Mean time/state:   {time_spent[time_spent > 0].mean():.4f}")
    print(f"  Min time (>0):     {time_spent[time_spent > 0].min():.6f}")
    print(f"  Max time:          {time_spent.max():.4f}")

    # Count distribution for observed transitions
    observed_counts = transition_counts[transition_counts > 0]
    print(f"\nTransition Counts (observed only):")
    print(f"  Mean:   {observed_counts.mean():.2f}")
    print(f"  Median: {np.median(observed_counts):.0f}")
    print(f"  Min:    {observed_counts.min():.0f}")
    print(f"  Max:    {observed_counts.max():.0f}")

    # Compute Full MLE with different regularization
    print(f"\nFull MLE Error (different regularization):")

    for reg in [0.0, 1e-6, 1e-3, 1e-1]:
        Q_mle = np.zeros((N_STATES, N_STATES))

        for i in range(N_STATES):
            if time_spent[i] > 0:
                for j in range(N_STATES):
                    if i != j:
                        Q_mle[i, j] = (transition_counts[i, j] + reg) / (time_spent[i] + reg * N_STATES)

        # Set diagonal
        for i in range(N_STATES):
            Q_mle[i, i] = -np.sum(Q_mle[i, :])

        # Enforce sparsity
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i != j and hamming(SEQS[i], SEQS[j]) > 1:
                    Q_mle[i, j] = 0

        # Recompute diagonal
        for i in range(N_STATES):
            Q_mle[i, i] = -np.sum(Q_mle[i, :])

        Q_mle_torch = torch.tensor(Q_mle, dtype=torch.float32)
        error = torch.norm(Q_mle_torch - Q_true) / torch.norm(Q_true)

        print(f"  reg={reg:<10}: error={error.item():.4f}")

    # Check: are unobserved transitions actually zero in ground truth?
    print(f"\nUnobserved Transitions Analysis:")
    unobserved_hamming1 = 0
    nonzero_in_gt = 0

    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                if transition_counts[i, j] == 0:
                    unobserved_hamming1 += 1
                    if abs(gt.Q_global[i, j]) > 1e-6:
                        nonzero_in_gt += 1

    print(f"  Unobserved Hamming-1 transitions: {unobserved_hamming1}")
    print(f"  Of these, nonzero in ground truth: {nonzero_in_gt}")
    print(f"  This contributes to error!")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("If error doesn't decrease much with more data, the issue is likely:")
print("  1. Unobserved transitions that are nonzero in ground truth")
print("  2. States that are rarely visited")
print("  3. Imbalanced data distribution")
