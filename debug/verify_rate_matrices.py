"""
Verify that all models satisfy rate matrix requirements:
1. Off-diagonal entries >= 0
2. Row sums = 0
3. Diagonal entries <= 0
"""

import torch
import numpy as np
from main import (
    GroundTruthProcess, NeuralRateModel, FullMLEModel, ContextIndependentRateModel,
    ALL_SEQS_BATCH, N_STATES, DEVICE
)

def check_rate_matrix(Q, name="Q"):
    """Check if Q is a valid rate matrix."""
    print(f"\n{name}:")

    # Check 1: Off-diagonal >= 0
    Q_np = Q.detach().cpu().numpy() if isinstance(Q, torch.Tensor) else Q
    off_diag_mask = ~np.eye(N_STATES, dtype=bool)
    off_diag = Q_np[off_diag_mask]
    min_off_diag = off_diag.min()
    negative_count = (off_diag < -1e-6).sum()

    print(f"  Off-diagonal entries >= 0? min={min_off_diag:.6f}, negative_count={negative_count}")
    if negative_count > 0:
        print(f"    ✗ FAILED: {negative_count} negative off-diagonal entries")
    else:
        print(f"    ✓ PASSED")

    # Check 2: Row sums = 0
    row_sums = Q_np.sum(axis=1)
    max_abs_row_sum = np.abs(row_sums).max()

    print(f"  Row sums = 0? max_abs={max_abs_row_sum:.2e}")
    if max_abs_row_sum > 1e-4:
        print(f"    ✗ FAILED: Row sums not zero")
        # Show worst rows
        worst_rows = np.argsort(np.abs(row_sums))[-5:]
        for row_idx in worst_rows[::-1]:
            print(f"      Row {row_idx}: sum={row_sums[row_idx]:.6f}")
    else:
        print(f"    ✓ PASSED")

    # Check 3: Diagonal <= 0
    diag = np.diag(Q_np)
    positive_diag_count = (diag > 1e-6).sum()

    print(f"  Diagonal entries <= 0? positive_count={positive_diag_count}")
    if positive_diag_count > 0:
        print(f"    ✗ FAILED: {positive_diag_count} positive diagonal entries")
    else:
        print(f"    ✓ PASSED")

    return negative_count == 0 and max_abs_row_sum < 1e-4 and positive_diag_count == 0

print("=" * 70)
print("RATE MATRIX VERIFICATION")
print("=" * 70)

# 1. Ground Truth
print("\n1. Ground Truth Process (epistasis=0.0)")
gt = GroundTruthProcess(seed=42, epistasis=0.0)
Q_true = gt.Q_torch
check_rate_matrix(Q_true, "Ground Truth Q")

print("\n2. Ground Truth Process (epistasis=1.0)")
gt_epi = GroundTruthProcess(seed=42, epistasis=1.0)
Q_true_epi = gt_epi.Q_torch
check_rate_matrix(Q_true_epi, "Ground Truth Q (epistasis=1.0)")

# 2. NeuralRateModel (context-dependent)
print("\n3. NeuralRateModel (initialized, not trained)")
model_neural = NeuralRateModel().to(DEVICE)
Q_neural = model_neural.build_global_Q()
check_rate_matrix(Q_neural, "NeuralRateModel Q (initialized)")

# 3. FullMLEModel
print("\n4. FullMLEModel (initialized, not trained)")
model_full = FullMLEModel().to(DEVICE)
Q_full = model_full.build_Q()
check_rate_matrix(Q_full, "FullMLEModel Q (initialized)")

# 4. ContextIndependentRateModel
print("\n5. ContextIndependentRateModel (initialized, not trained)")
model_indep = ContextIndependentRateModel().to(DEVICE)
Q_indep = model_indep.build_global_Q()
check_rate_matrix(Q_indep, "ContextIndependentRateModel Q (initialized)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("All models should satisfy rate matrix requirements:")
print("  1. Off-diagonal entries >= 0")
print("  2. Row sums = 0")
print("  3. Diagonal entries <= 0")
print("\nThese requirements are enforced by:")
print("  - Using softplus() activation for off-diagonal entries")
print("  - Setting diagonal = -row_sum")
