"""
Test the FIXED Full MLE model with proper likelihood maximization.

Should achieve much lower error than the broken count-based estimator.
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

print("=" * 70)
print("TESTING FIXED FULL MLE (Gradient Descent)")
print("=" * 70)

epistasis = 0.0
seed = 42
n_samples = 1000  # Reduce for faster testing

# Generate data
gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

print(f"\nGenerating {n_samples} samples...")
data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)

print(f"\nTraining Full MLE model (256×256, gradient descent)...")
print(f"Expected: Error should decrease significantly with training")
print()

model, history = train_full_mle_model(
    gt, data,
    epochs=100,  # Reduced for faster testing
    lr=0.01,
    verbose=True
)

# Final evaluation
with torch.no_grad():
    Q_model = model.build_Q()
    final_error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Initial error:  {history['errors'][0]:.4f}")
print(f"Final error:    {history['errors'][-1]:.4f}")
print(f"Improvement:    {history['errors'][0] - history['errors'][-1]:.4f}")

if history['errors'][-1] < 0.3:
    print(f"\n✓ SUCCESS: Full MLE achieves low error!")
    print(f"  The gradient descent approach works correctly.")
else:
    print(f"\n⚠ Error still high - may need more epochs or better LR")

# Plot training curve
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Error
epochs_plot = list(range(0, 100, 50)) + [99]
ax1.plot(epochs_plot, history['errors'], 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Relative Frobenius Error')
ax1.set_title('Full MLE Training (256×256 matrix)')
ax1.grid(True, alpha=0.3)

# Plot 2: Loss
ax2.plot(epochs_plot, history['losses'], 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Negative Log-Likelihood')
ax2.set_title('Training Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/full_mle_fixed.png', dpi=150)
print(f"\nSaved results/full_mle_fixed.png")
