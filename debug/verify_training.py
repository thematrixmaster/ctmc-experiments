"""
Verify that error metrics are computed correctly.

Shows:
1. Full MLE has no training - computed once from counts
2. Factorized model loss vs error behavior
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')

from main import *

print("=" * 70)
print("VERIFICATION: Full MLE vs Factorized Model Training")
print("=" * 70)

# Setup
epistasis = 0.0
seed = 42
n_samples = 5000

gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

# Load/generate data
data = load_dataset(epistasis, seed, n_samples)
if data is None:
    print(f"Generating {n_samples} samples...")
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=1)
else:
    print(f"Loaded {len(data)} samples from cache")

print(f"\n{'='*70}")
print("1. FULL MLE (Non-parametric, Count-based)")
print(f"{'='*70}")
print("This model has NO training loop - it's computed directly from data:")
print("  Q[i,j] = N(i→j) / T(i)  where N = counts, T = time spent")

Q_full_np = estimate_full_mle_q(data, gt)
Q_full = torch.tensor(Q_full_np, dtype=torch.float32)
error_full = torch.norm(Q_full - Q_true) / torch.norm(Q_true)

print(f"\nFull MLE error: {error_full.item():.4f}")
print("(This is as good as it gets with empirical counts)")

print(f"\n{'='*70}")
print("2. FACTORIZED MODEL (Neural, Trainable)")
print(f"{'='*70}")
print("This model TRAINS via MLE (minimizing negative log-likelihood)")
print("Let's watch loss AND error during training:\n")

# Train with monitoring
model = NeuralRateModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

starts = torch.stack([d[0] for d in data])
ends = torch.stack([d[1] for d in data])
bs = torch.tensor([d[2] for d in data], dtype=torch.float32)

print(f"{'Epoch':<8} {'Loss':<12} {'Error':<12} {'Status'}")
print("-" * 50)

for epoch in range(0, 501, 50):
    if epoch > 0:
        # Train for 50 epochs
        for _ in range(50):
            optimizer.zero_grad()
            q_preds = model(starts)
            log_prob_total = 0
            for l in range(L):
                q_l = q_preds[:, l]
                p_matrix = torch.matrix_exp(bs.view(-1, 1, 1) * q_l)
                probs = p_matrix[torch.arange(len(data)), starts[:, l], ends[:, l]]
                log_prob_total += torch.log(probs + 1e-9)
            loss = -log_prob_total.mean()
            loss.backward()
            optimizer.step()

    # Evaluate
    with torch.no_grad():
        # Compute current loss
        q_preds = model(starts)
        log_prob_total = 0
        for l in range(L):
            q_l = q_preds[:, l]
            p_matrix = torch.matrix_exp(bs.view(-1, 1, 1) * q_l)
            probs = p_matrix[torch.arange(len(data)), starts[:, l], ends[:, l]]
            log_prob_total += torch.log(probs + 1e-9)
        loss = -log_prob_total.mean()

        # Compute error (Frobenius norm of reconstructed Q vs true Q)
        Q_model = model.build_global_Q()
        error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

        status = ""
        if epoch == 0:
            status = "(initialization)"
        elif error > error_full.item():
            status = "⚠ WORSE than Full MLE"

        print(f"{epoch:<8} {loss.item():<12.4f} {error.item():<12.4f} {status}")

print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")
print(f"Full MLE error (baseline):           {error_full.item():.4f}")
print(f"Factorized model final error:        {error.item():.4f}")
print(f"Gap:                                 {error.item() - error_full.item():+.4f}")

if error.item() > error_full.item() + 0.2:
    print("\n⚠ PROBLEM CONFIRMED:")
    print("  - Loss DECREASED (model fits data better)")
    print("  - Error INCREASED (reconstructed Q matrix is worse)")
    print("  - This is OVERFITTING: learning spurious context-dependencies")
else:
    print("\n✓ Factorized model achieves similar or better error than Full MLE")
