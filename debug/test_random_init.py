"""
Test: Why does random factorized model beat empirical Full MLE?

Hypothesis: The factorization structure provides massive regularization,
even with random parameters.
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

print("=" * 70)
print("Testing Random Initialization vs Full MLE")
print("=" * 70)

epistasis = 0.0
seed = 42
n_samples = 5000

gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

# Load data
data = load_dataset(epistasis, seed, n_samples)
if data is None:
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=1)

print(f"\nGround truth: epistasis={epistasis}, samples={n_samples}")

# 1. Full MLE
print(f"\n1. Full MLE (empirical counts):")
Q_full_np = estimate_full_mle_q(data, gt)
Q_full = torch.tensor(Q_full_np, dtype=torch.float32)
error_full = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
print(f"   Error: {error_full.item():.4f}")

# 2. Random factorized models (try multiple)
print(f"\n2. Random factorized models (10 trials):")
errors_random = []
for trial in range(10):
    model = NeuralRateModel()  # Random initialization
    with torch.no_grad():
        Q_random = model.build_global_Q()
        error_random = torch.norm(Q_random - Q_true) / torch.norm(Q_true)
        errors_random.append(error_random.item())
        print(f"   Trial {trial+1}: Error = {error_random.item():.4f}")

mean_random = np.mean(errors_random)
std_random = np.std(errors_random)

# 3. Analysis
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")
print(f"Full MLE error:                   {error_full.item():.4f}")
print(f"Random factorized (mean ± std):   {mean_random:.4f} ± {std_random:.4f}")
print(f"Gap:                              {mean_random - error_full.item():+.4f}")

if mean_random < error_full.item():
    print(f"\n✓ CONFIRMED: Random factorized structure beats noisy empirical estimates!")
    print(f"  The factorization constraint provides ~{error_full.item()/mean_random:.1f}× regularization")
    print(f"  even before any training.")
else:
    print(f"\n✗ Unexpected: Full MLE is better than random")

# 4. Check sparsity/coverage
print(f"\n{'='*70}")
print("DATA COVERAGE")
print(f"{'='*70}")

# Count how many state pairs have observations
transition_counts = np.zeros((N_STATES, N_STATES))
for start_seq, end_seq, b in data:
    i = seq_to_idx(tuple(start_seq.tolist()))
    j = seq_to_idx(tuple(end_seq.tolist()))
    transition_counts[i, j] += 1

# Count Hamming-1 pairs (possible transitions)
n_hamming1_pairs = 0
for i in range(N_STATES):
    for j in range(N_STATES):
        if hamming(SEQS[i], SEQS[j]) == 1:
            n_hamming1_pairs += 1

n_observed = np.sum(transition_counts > 0)
coverage = n_observed / n_hamming1_pairs

print(f"Total possible Hamming-1 transitions: {n_hamming1_pairs}")
print(f"Observed transitions:                 {n_observed}")
print(f"Coverage:                             {coverage:.1%}")
print(f"\nWith {n_samples} samples across {N_STATES} states,")
print(f"many transitions have 0 or very few counts -> high variance estimates")
print(f"\nFactorized model: only {4 * 4 * 4} = 64 parameters")
print(f"Full MLE model:   ~{n_hamming1_pairs} parameters")
print(f"Regularization:   {n_hamming1_pairs/64:.1f}× fewer parameters")
