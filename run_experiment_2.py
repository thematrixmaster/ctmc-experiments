"""
Experiment 2: SNR Weighting Comparison

Trains two factorized models on the same dataset:
1. Standard factorized model (no SNR weighting)
2. Factorized model with SNR weighting

Compares their prediction accuracy on the ground truth rate matrix.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle
import hashlib
from main import GroundTruthProcess, train_model, DEVICE, N_STATES, SEQS

print("=" * 80)
print("EXPERIMENT 2: SNR Weighting Comparison")
print("=" * 80)

# Configuration
EPISTASIS = 1.0
SEED = 42
LAMBDA_PARAM = 0.5
N_SAMPLES = 1000000  # Training dataset size

# Cache directory
USE_CACHE = False
CACHE_DIR = './exp2_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Use cache: {'YES' if USE_CACHE else 'NO'}")
print(f"  Epistasis: {EPISTASIS}")
print(f"  Training samples: {N_SAMPLES}")
print(f"  Lambda parameter: {LAMBDA_PARAM}")
print(f"  Random seed: {SEED}")

# Caching functions
def get_cache_key(epistasis, seed, n_samples, lambda_param):
    """Generate unique cache key based on parameters."""
    key_str = f"eps{epistasis}_seed{seed}_n{n_samples}_lam{lambda_param}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_dataset(epistasis, seed, n_samples, lambda_param):
    """Load cached dataset if available."""
    cache_key = get_cache_key(epistasis, seed, n_samples, lambda_param)
    cache_file = os.path.join(CACHE_DIR, f"dataset_{cache_key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_dataset(data, epistasis, seed, n_samples, lambda_param):
    """Save dataset to cache."""
    cache_key = get_cache_key(epistasis, seed, n_samples, lambda_param)
    cache_file = os.path.join(CACHE_DIR, f"dataset_{cache_key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def get_model_cache_key(epistasis, seed, n_samples, lambda_param, use_snr):
    """Generate unique cache key for model including SNR flag."""
    key_str = f"eps{epistasis}_seed{seed}_n{n_samples}_lam{lambda_param}_snr{use_snr}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_model(epistasis, seed, n_samples, lambda_param, use_snr):
    """Load cached model if available."""
    cache_key = get_model_cache_key(epistasis, seed, n_samples, lambda_param, use_snr)
    cache_file = os.path.join(CACHE_DIR, f"model_{cache_key}.pt")
    if os.path.exists(cache_file):
        return torch.load(cache_file, map_location=DEVICE)
    return None

def save_model(Q_matrix, epistasis, seed, n_samples, lambda_param, use_snr):
    """Save model Q matrix to cache."""
    cache_key = get_model_cache_key(epistasis, seed, n_samples, lambda_param, use_snr)
    cache_file = os.path.join(CACHE_DIR, f"model_{cache_key}.pt")
    torch.save(Q_matrix, cache_file)

# Step 1: Create ground truth with epistasis
print(f"\n" + "=" * 80)
print("Step 1: Creating Ground Truth Rate Matrix")
print("=" * 80)

gt = GroundTruthProcess(seed=SEED, epistasis=EPISTASIS)
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)

print(f"✓ Ground truth Q created with epistasis={EPISTASIS}")
print(f"  Frobenius norm: {torch.norm(Q_true).item():.2f}")

# Step 2: Generate or load training dataset
print(f"\n" + "=" * 80)
print("Step 2: Preparing Training Dataset")
print("=" * 80)

# Try to load cached dataset
train_data = load_dataset(EPISTASIS, SEED, N_SAMPLES, LAMBDA_PARAM) if USE_CACHE else None
if train_data is not None:
    print(f"✓ Loaded cached dataset ({len(train_data)} samples)")
else:
    print(f"Generating {N_SAMPLES} training samples...")
    train_data = gt.generate_data(N_SAMPLES, lambda_param=LAMBDA_PARAM, min_mutations=0)
    print(f"✓ Generated {len(train_data)} samples")
    # if USE_CACHE:
    #     save_dataset(train_data, EPISTASIS, SEED, N_SAMPLES, LAMBDA_PARAM)
    #     print(f"  Saved to cache")

# Step 3: Train factorized model WITHOUT SNR weighting
print(f"\n" + "=" * 80)
print("Step 3: Training Factorized Model (NO SNR Weighting)")
print("=" * 80)

Q_no_snr = load_model(EPISTASIS, SEED, N_SAMPLES, LAMBDA_PARAM, use_snr=False) if USE_CACHE else None
if Q_no_snr is not None:
    print(f"✓ Loaded cached model (no SNR)")
    error_no_snr = torch.norm(Q_no_snr - Q_true) / torch.norm(Q_true)
    print(f"  Training error: {error_no_snr.item():.4f}")
else:
    print(f"Training factorized model without SNR weighting...")
    model_no_snr, history_no_snr = train_model(
        gt, train_data,
        use_snr=False,
        epochs=1000,
        lr=0.01,
        early_stop_patience=100,
        early_stop_tolerance=0.0001,
        verbose=True,
    )

    Q_no_snr = model_no_snr.build_global_Q()
    error_no_snr = torch.norm(Q_no_snr - Q_true) / torch.norm(Q_true)
    print(f"✓ Model trained (no SNR)")
    print(f"  Training error: {error_no_snr.item():.4f}")

    if USE_CACHE:
        save_model(Q_no_snr, EPISTASIS, SEED, N_SAMPLES, LAMBDA_PARAM, use_snr=False)
        print(f"  Saved to cache")

# Step 4: Train factorized model WITH SNR weighting
print(f"\n" + "=" * 80)
print("Step 4: Training Factorized Model (WITH SNR Weighting)")
print("=" * 80)

Q_with_snr = load_model(EPISTASIS, SEED, N_SAMPLES, LAMBDA_PARAM, use_snr=True) if USE_CACHE else None
if Q_with_snr is not None:
    print(f"✓ Loaded cached model (with SNR)")
    error_with_snr = torch.norm(Q_with_snr - Q_true) / torch.norm(Q_true)
    print(f"  Training error: {error_with_snr.item():.4f}")
else:
    print(f"Training factorized model with SNR weighting...")
    model_with_snr, history_with_snr = train_model(
        gt, train_data,
        use_snr=True,
        epochs=1000,
        lr=0.01,
        early_stop_patience=100,
        early_stop_tolerance=0.0001,
        verbose=True,
    )

    Q_with_snr = model_with_snr.build_global_Q()
    error_with_snr = torch.norm(Q_with_snr - Q_true) / torch.norm(Q_true)
    print(f"✓ Model trained (with SNR)")
    print(f"  Training error: {error_with_snr.item():.4f}")

    if USE_CACHE:
        save_model(Q_with_snr, EPISTASIS, SEED, N_SAMPLES, LAMBDA_PARAM, use_snr=True)
        print(f"  Saved to cache")

# Step 5: Compare results
print(f"\n" + "=" * 80)
print("Step 5: Comparison Results")
print("=" * 80)

# Recompute errors for final comparison
error_no_snr = torch.norm(Q_no_snr - Q_true) / torch.norm(Q_true)
error_with_snr = torch.norm(Q_with_snr - Q_true) / torch.norm(Q_true)

print(f"\nRelative Frobenius Norm Errors:")
print(f"  No SNR weighting:   {error_no_snr.item():.6f}")
print(f"  With SNR weighting: {error_with_snr.item():.6f}")
print(f"  Difference:         {(error_with_snr - error_no_snr).item():.6f}")
print(f"  Improvement:        {((error_no_snr - error_with_snr) / error_no_snr * 100).item():.2f}%")

# Element-wise comparison
Q_true_np = Q_true.cpu().numpy()
Q_no_snr_np = Q_no_snr.cpu().numpy()
Q_with_snr_np = Q_with_snr.cpu().numpy()

abs_error_no_snr = np.abs(Q_true_np - Q_no_snr_np)
abs_error_with_snr = np.abs(Q_true_np - Q_with_snr_np)

print(f"\nElement-wise Absolute Errors:")
print(f"  No SNR - Mean:   {np.mean(abs_error_no_snr):.6f}, Max: {np.max(abs_error_no_snr):.6f}")
print(f"  With SNR - Mean: {np.mean(abs_error_with_snr):.6f}, Max: {np.max(abs_error_with_snr):.6f}")

# Step 6: Visualize comparison
print(f"\n" + "=" * 80)
print("Step 6: Creating Visualizations")
print("=" * 80)

os.makedirs('plots', exist_ok=True)

# Create comparison bar plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Error comparison
ax1 = axes[0]
methods = ['No SNR', 'With SNR']
errors = [error_no_snr.item(), error_with_snr.item()]
colors = ['#3498db', '#2ecc71']
bars = ax1.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Relative Frobenius Norm Error')
ax1.set_title(f'Model Comparison (Epistasis={EPISTASIS})')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, error in zip(bars, errors):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{error:.4f}',
             ha='center', va='bottom', fontweight='bold')

# Plot 2: Element-wise error distribution
ax2 = axes[1]
ax2.hist(abs_error_no_snr.flatten(), bins=50, alpha=0.6, label='No SNR', color='#3498db', density=True)
ax2.hist(abs_error_with_snr.flatten(), bins=50, alpha=0.6, label='With SNR', color='#2ecc71', density=True)
ax2.set_xlabel('Absolute Error per Element')
ax2.set_ylabel('Density')
ax2.set_title('Element-wise Error Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()

# Save plots
plot_path_png = 'plots/experiment_2_snr_comparison.png'
plot_path_pdf = 'plots/experiment_2_snr_comparison.pdf'
plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_path_pdf, bbox_inches='tight')
print(f"✓ Plot saved to {plot_path_png}")
print(f"✓ Plot saved to {plot_path_pdf}")

plt.close()

# Step 7: Save results summary
print(f"\n" + "=" * 80)
print("Step 7: Saving Results Summary")
print("=" * 80)

results = {
    'epistasis': EPISTASIS,
    'seed': SEED,
    'n_samples': N_SAMPLES,
    'lambda_param': LAMBDA_PARAM,
    'error_no_snr': error_no_snr.item(),
    'error_with_snr': error_with_snr.item(),
    'improvement_pct': ((error_no_snr - error_with_snr) / error_no_snr * 100).item(),
    'Q_true_norm': torch.norm(Q_true).item(),
    'Q_no_snr_norm': torch.norm(Q_no_snr).item(),
    'Q_with_snr_norm': torch.norm(Q_with_snr).item(),
}

results_path = 'plots/experiment_2_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"✓ Results saved to {results_path}")

print(f"\n" + "=" * 80)
print("EXPERIMENT 2 COMPLETED!")
print("=" * 80)
print(f"\nSummary:")
print(f"  Error without SNR: {error_no_snr.item():.6f}")
print(f"  Error with SNR:    {error_with_snr.item():.6f}")
if error_with_snr < error_no_snr:
    print(f"  ✓ SNR weighting IMPROVED accuracy by {((error_no_snr - error_with_snr) / error_no_snr * 100).item():.2f}%")
else:
    print(f"  ✗ SNR weighting DEGRADED accuracy by {((error_with_snr - error_no_snr) / error_no_snr * 100).item():.2f}%")
print(f"\nPlots saved to:")
print(f"  - {plot_path_png}")
print(f"  - {plot_path_pdf}")
print(f"Results saved to: {results_path}")
print("=" * 80)

