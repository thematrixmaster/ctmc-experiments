"""
Diagnose the relationship between loss and error.

Key question: Does minimizing loss actually minimize error?
Or is there a mismatch between the two objectives?
"""

import sys
sys.path.append('/accounts/projects/yss/stephen.lu/ctmc')

from main import *
import matplotlib.pyplot as plt

print("=" * 80)
print("Loss vs Error Diagnostic")
print("=" * 80)

# Generate test data
gt = GroundTruthProcess(seed=42, epistasis=0.0)
data = gt.generate_data(1000, lambda_param=1.0, min_mutations=0)

Q_true_torch = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)

starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in data], dtype=torch.long).to(DEVICE)
ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in data], dtype=torch.long).to(DEVICE)
bs = torch.tensor([d[2] for d in data], dtype=torch.float32).to(DEVICE)

bs_bucketed = torch.round(bs * 100) / 100
unique_bs = torch.unique(bs_bucketed)

def compute_loss_and_error(model):
    """Compute both loss and error for a model."""
    with torch.no_grad():
        Q = model.build_Q()

        # Compute error (Frobenius norm)
        error = (torch.norm(Q - Q_true_torch) / torch.norm(Q_true_torch)).item()

        # Compute loss (negative log-likelihood)
        log_prob_total = 0
        for b_val in unique_bs:
            mask = (bs_bucketed == b_val)
            if mask.sum() == 0:
                continue

            P = torch.matrix_exp(b_val * Q)
            batch_starts = starts_idx[mask]
            batch_ends = ends_idx[mask]
            probs = P[batch_starts, batch_ends]
            log_prob_total += torch.sum(torch.log(probs + 1e-9))

        loss = (-log_prob_total / len(data)).item()

        return loss, error

# Test 1: Loss/error at ground truth
model_truth = FullMLEModel().to(DEVICE)
with torch.no_grad():
    for (i, j), idx in model_truth.param_indices.items():
        true_rate = gt.Q_global[i, j]
        if true_rate > 0:
            model_truth.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

loss_truth, error_truth = compute_loss_and_error(model_truth)
print(f"\nAt ground truth Q:")
print(f"  Loss:  {loss_truth:.6f}")
print(f"  Error: {error_truth:.6f}")

# Test 2: Systematically perturb Q and track loss/error
print(f"\nSystematic perturbation test:")
print(f"{'Scale':<10} {'Loss':<15} {'Error':<15} {'Loss Δ':<15} {'Error Δ':<15}")
print("-" * 75)

scales = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
losses = []
errors = []

for scale in scales:
    model_pert = FullMLEModel().to(DEVICE)
    with torch.no_grad():
        # Initialize to ground truth
        for (i, j), idx in model_pert.param_indices.items():
            true_rate = gt.Q_global[i, j]
            if true_rate > 0:
                model_pert.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

        # Add random perturbation
        perturbation = torch.randn_like(model_pert.log_rates) * scale
        model_pert.log_rates.add_(perturbation)

    loss, error = compute_loss_and_error(model_pert)
    losses.append(loss)
    errors.append(error)

    loss_delta = loss - loss_truth
    error_delta = error - error_truth

    print(f"{scale:<10.3f} {loss:<15.6f} {error:<15.6f} {loss_delta:<15.6f} {error_delta:<15.6f}")

# Test 3: Does gradient descent on loss reduce error?
print(f"\n" + "=" * 80)
print("Gradient Descent Test: Does minimizing loss reduce error?")
print("=" * 80)

model_gd = FullMLEModel().to(DEVICE)
optimizer = optim.Adam(model_gd.parameters(), lr=0.01)

# Start from ground truth
with torch.no_grad():
    for (i, j), idx in model_gd.param_indices.items():
        true_rate = gt.Q_global[i, j]
        if true_rate > 0:
            model_gd.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

print(f"\n{'Epoch':<8} {'Loss':<15} {'Error':<15} {'Grad Norm':<15}")
print("-" * 60)

for epoch in range(20):
    optimizer.zero_grad()

    Q = model_gd.build_Q()

    log_prob_total = 0
    for b_val in unique_bs:
        mask = (bs_bucketed == b_val)
        if mask.sum() == 0:
            continue

        P = torch.matrix_exp(b_val * Q)
        batch_starts = starts_idx[mask]
        batch_ends = ends_idx[mask]
        probs = P[batch_starts, batch_ends]
        log_prob_total += torch.sum(torch.log(probs + 1e-9))

    loss = -log_prob_total / len(data)
    loss.backward()

    grad_norm = model_gd.log_rates.grad.norm().item()

    # Compute error before step
    loss_val, error_val = compute_loss_and_error(model_gd)

    optimizer.step()

    if epoch % 2 == 0:
        print(f"{epoch:<8} {loss_val:<15.6f} {error_val:<15.6f} {grad_norm:<15.6f}")

# Final values
loss_final, error_final = compute_loss_and_error(model_gd)
print(f"\nFinal:")
print(f"  Loss:  {loss_final:.6f} (started at {loss_truth:.6f})")
print(f"  Error: {error_final:.6f} (started at {error_truth:.6f})")
print(f"\nDid loss decrease?: {loss_final < loss_truth}")
print(f"Did error decrease?: {error_final < error_truth}")

# Test 4: Check if the issue is regularization/weight decay
print(f"\n" + "=" * 80)
print("Test: Is weight decay causing the drift?")
print("=" * 80)

model_wd = FullMLEModel().to(DEVICE)
optimizer_wd = optim.Adam(model_wd.parameters(), lr=0.01, weight_decay=1e-4)

# Start from ground truth
with torch.no_grad():
    for (i, j), idx in model_wd.param_indices.items():
        true_rate = gt.Q_global[i, j]
        if true_rate > 0:
            model_wd.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

initial_params = model_wd.log_rates.clone().detach()

# One step
optimizer_wd.zero_grad()
Q = model_wd.build_Q()

log_prob_total = 0
for b_val in unique_bs:
    mask = (bs_bucketed == b_val)
    if mask.sum() == 0:
        continue

    P = torch.matrix_exp(b_val * Q)
    batch_starts = starts_idx[mask]
    batch_ends = ends_idx[mask]
    probs = P[batch_starts, batch_ends]
    log_prob_total += torch.sum(torch.log(probs + 1e-9))

loss = -log_prob_total / len(data)
loss.backward()

# Check gradient before weight decay
grad_before_wd = model_wd.log_rates.grad.clone()

# Step (includes weight decay)
optimizer_wd.step()

# Check parameter change
param_change = model_wd.log_rates - initial_params
wd_effect = -1e-4 * 0.01 * initial_params  # weight_decay * lr * params

print(f"Gradient (before WD) norm: {grad_before_wd.norm().item():.6e}")
print(f"Weight decay effect norm: {wd_effect.norm().item():.6e}")
print(f"Actual param change norm: {param_change.norm().item():.6e}")
print(f"\nRatio (WD effect / gradient): {wd_effect.norm().item() / grad_before_wd.norm().item():.4f}")

print("\nDiagnostic complete!")
