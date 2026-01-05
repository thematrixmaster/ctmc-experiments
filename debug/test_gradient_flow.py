"""
Test if gradients are flowing correctly through matrix exponential.
"""

import sys
sys.path.append('/accounts/projects/yss/stephen.lu/ctmc')

from main import *

print("=" * 80)
print("Testing Gradient Flow Through Matrix Exponential")
print("=" * 80)

# Create a simple test case
gt = GroundTruthProcess(seed=42, epistasis=0.0)
data = gt.generate_data(100, lambda_param=1.0, min_mutations=0)

model = FullMLEModel().to(DEVICE)

# Initialize close to ground truth
Q_true = gt.Q_global
with torch.no_grad():
    for (i, j), idx in model.param_indices.items():
        true_rate = Q_true[i, j]
        if true_rate > 0:
            model.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

# Check gradients
model.log_rates.requires_grad = True

# Build Q
Q = model.build_Q()
print(f"\nQ matrix built. Shape: {Q.shape}")
print(f"Q[0,0] (diagonal): {Q[0,0].item():.6f}")
print(f"Q[0,1] (off-diag): {Q[0,1].item():.6f}")

# Simple test: compute P = exp(0.5 * Q) and a dummy loss
b_test = 0.5
P = torch.matrix_exp(b_test * Q)
print(f"\nP = exp({b_test} * Q) computed")
print(f"P[0,0]: {P[0,0].item():.6f}")
print(f"P[0,1]: {P[0,1].item():.6f}")

# Dummy loss: sum of P
loss = P.sum()
print(f"\nDummy loss: {loss.item():.6f}")

# Backprop
loss.backward()

# Check if gradients exist
print(f"\nGradients computed:")
print(f"model.log_rates.grad is None: {model.log_rates.grad is None}")
if model.log_rates.grad is not None:
    print(f"Gradient norm: {model.log_rates.grad.norm().item():.6f}")
    print(f"Gradient mean: {model.log_rates.grad.mean().item():.6e}")
    print(f"Gradient std: {model.log_rates.grad.std().item():.6e}")
    print(f"Gradient min: {model.log_rates.grad.min().item():.6e}")
    print(f"Gradient max: {model.log_rates.grad.max().item():.6e}")

    # Check for NaNs or Infs
    has_nan = torch.isnan(model.log_rates.grad).any()
    has_inf = torch.isinf(model.log_rates.grad).any()
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")

print("\n" + "=" * 80)
print("Testing Real Training Loss")
print("=" * 80)

# Now test the real training loss
model = FullMLEModel().to(DEVICE)

# Initialize close to ground truth again
with torch.no_grad():
    for (i, j), idx in model.param_indices.items():
        true_rate = Q_true[i, j]
        if true_rate > 0:
            model.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

optimizer = optim.Adam(model.parameters(), lr=0.01)

starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in data], dtype=torch.long).to(DEVICE)
ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in data], dtype=torch.long).to(DEVICE)
bs = torch.tensor([d[2] for d in data], dtype=torch.float32).to(DEVICE)

# Compute initial error
with torch.no_grad():
    Q_init = model.build_Q()
    Q_true_torch = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    init_error = torch.norm(Q_init - Q_true_torch) / torch.norm(Q_true_torch)
    print(f"Initial error: {init_error.item():.6f}")

# One training step
optimizer.zero_grad()
Q = model.build_Q()

bucket_factor = 100
bs_bucketed = torch.round(bs * bucket_factor) / bucket_factor
unique_bs = torch.unique(bs_bucketed)

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
print(f"Loss before backward: {loss.item():.6f}")

loss.backward()

print(f"\nGradients after backward:")
print(f"model.log_rates.grad norm: {model.log_rates.grad.norm().item():.6f}")
print(f"model.log_rates.grad mean: {model.log_rates.grad.mean().item():.6e}")

# Take optimizer step
optimizer.step()

# Check error after one step
with torch.no_grad():
    Q_after = model.build_Q()
    after_error = torch.norm(Q_after - Q_true_torch) / torch.norm(Q_true_torch)
    print(f"\nError after 1 step: {after_error.item():.6f}")
    print(f"Did error increase?: {after_error > init_error}")

    # Check what changed
    param_change = torch.norm(Q_after - Q_init) / torch.norm(Q_init)
    print(f"Relative change in Q: {param_change.item():.6f}")

print("\n" + "=" * 80)
print("Testing Loss Landscape")
print("=" * 80)

# Perturb Q slightly in a random direction and see if loss changes appropriately
model2 = FullMLEModel().to(DEVICE)
with torch.no_grad():
    for (i, j), idx in model2.param_indices.items():
        true_rate = Q_true[i, j]
        if true_rate > 0:
            model2.log_rates[idx] = np.log(np.exp(true_rate) - 1 + 1e-6)

def compute_loss(model, data):
    """Compute the training loss for a model."""
    with torch.no_grad():
        Q = model.build_Q()

        starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in data], dtype=torch.long).to(DEVICE)
        ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in data], dtype=torch.long).to(DEVICE)
        bs = torch.tensor([d[2] for d in data], dtype=torch.float32).to(DEVICE)

        bs_bucketed = torch.round(bs * 100) / 100
        unique_bs = torch.unique(bs_bucketed)

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

        return (-log_prob_total / len(data)).item()

loss_at_truth = compute_loss(model2, data)
print(f"Loss at ground truth Q: {loss_at_truth:.6f}")

# Perturb in a random direction
with torch.no_grad():
    perturbation = torch.randn_like(model2.log_rates) * 0.01
    model2.log_rates.add_(perturbation)

loss_perturbed = compute_loss(model2, data)
print(f"Loss after small random perturbation: {loss_perturbed:.6f}")
print(f"Did loss increase (as expected)?: {loss_perturbed > loss_at_truth}")

print("\nTest complete!")
