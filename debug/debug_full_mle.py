"""
Deep debug of Full MLE - why is error stuck at 1.09?

Check:
1. Is Q_mle a valid rate matrix?
2. Are the estimates reasonable?
3. Compare Q_mle to Q_true element-wise
4. Check for numerical issues
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

print("=" * 70)
print("DEBUGGING FULL MLE ESTIMATION")
print("=" * 70)

epistasis = 0.0
seed = 99
n_samples = 100000

# Generate data
gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
Q_true_np = gt.Q_global
Q_true = torch.tensor(Q_true_np, dtype=torch.float32)

print(f"Generating {n_samples} samples...")
data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)

# Compute Full MLE
Q_mle = estimate_full_mle_q(data, gt, regularization=1e-6)
Q_mle_torch = torch.tensor(Q_mle, dtype=torch.float32)

# Check 1: Is Q_mle valid?
print(f"\n1. Validity Checks:")
row_sums = Q_mle.sum(axis=1)
print(f"   Max |row_sum|: {np.abs(row_sums).max():.2e} (should be ~0)")
print(f"   All off-diagonal >= 0? {np.all(Q_mle >= 0) or np.all(Q_mle[np.eye(256, dtype=bool) == False] >= 0)}")

# Check 2: Compare norms
print(f"\n2. Matrix Norms:")
print(f"   ||Q_true||_F:  {torch.norm(Q_true).item():.4f}")
print(f"   ||Q_mle||_F:   {torch.norm(Q_mle_torch).item():.4f}")
print(f"   Ratio:         {torch.norm(Q_mle_torch).item() / torch.norm(Q_true).item():.4f}")

# Check 3: Error computation
diff = Q_mle_torch - Q_true
error_manual = torch.norm(diff) / torch.norm(Q_true)
error_func = torch.norm(Q_mle_torch - Q_true) / torch.norm(Q_true)

print(f"\n3. Error Computation:")
print(f"   ||Q_mle - Q_true||_F: {torch.norm(diff).item():.4f}")
print(f"   Error (manual):       {error_manual.item():.4f}")
print(f"   Error (function):     {error_func.item():.4f}")

# Check 4: Element-wise comparison
print(f"\n4. Element-wise Analysis (Hamming-1 only):")

# Collect differences for Hamming-1 transitions
diffs_hamming1 = []
relative_errors = []

for i in range(N_STATES):
    for j in range(N_STATES):
        if i != j and hamming(SEQS[i], SEQS[j]) == 1:
            true_val = Q_true_np[i, j]
            mle_val = Q_mle[i, j]
            diff_val = abs(mle_val - true_val)
            diffs_hamming1.append(diff_val)
            if abs(true_val) > 1e-6:
                relative_errors.append(diff_val / abs(true_val))

diffs_hamming1 = np.array(diffs_hamming1)
relative_errors = np.array(relative_errors)

print(f"   Mean absolute diff:     {diffs_hamming1.mean():.4f}")
print(f"   Median absolute diff:   {np.median(diffs_hamming1):.4f}")
print(f"   Max absolute diff:      {diffs_hamming1.max():.4f}")
print(f"   Mean relative error:    {relative_errors.mean():.2f}")
print(f"   Median relative error:  {np.median(relative_errors):.2f}")

# Check 5: Show some specific mismatches
print(f"\n5. Sample Mismatches (worst 5):")
worst_indices = np.argsort(diffs_hamming1)[-5:]

for idx in worst_indices[::-1]:
    # Find the (i, j) pair
    count = 0
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                if count == idx:
                    print(f"   [{i},{j}]: Q_true={Q_true_np[i,j]:.4f}, Q_mle={Q_mle[i,j]:.4f}, "
                          f"diff={abs(Q_mle[i,j] - Q_true_np[i,j]):.4f}")
                    break
                count += 1

# Check 6: Is the ground truth correct?
print(f"\n6. Ground Truth Verification:")
print(f"   Epistasis: {epistasis}")
print(f"   Should be context-independent at epistasis=0")

# Sample a few rates for same transition type
sample_transitions = []
for i in range(min(10, N_STATES)):
    s_i = SEQS[i]
    if s_i[0] == 1:  # C at site 0
        s_j = list(s_i)
        s_j[0] = 2  # Mutate to G
        j = seq_to_idx(tuple(s_j))
        sample_transitions.append(Q_true_np[i, j])

if len(sample_transitions) > 0:
    print(f"   Sample C→G rates at site 0: {sample_transitions[:5]}")
    print(f"   Std: {np.std(sample_transitions):.6f} (should be 0 at epistasis=0)")

print(f"\n{'='*70}")
print("DIAGNOSIS:")
print(f"{'='*70}")

if np.abs(row_sums).max() > 1e-3:
    print("⚠ Q_mle has invalid row sums - BUG in construction!")
elif error_func < 0.3:
    print("✓ Full MLE is working correctly - error is low")
elif diffs_hamming1.mean() > 0.5:
    print("⚠ Large element-wise differences - estimation quality issue")
else:
    print("? Error is high but individual estimates seem reasonable")
    print("  This might be expected with finite data")
