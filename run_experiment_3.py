"""
Experiment 3: Branch Length Error Scaling Analysis

Tests how the factorized model's transition probability prediction error
scales with branch length when epistasis is present.

Theoretical prediction: For a factorized model with epistasis in ground truth,
the error should scale as O(b²) for small branch lengths b.
"""

from pyexpat import errors
import sys
import numpy as np
import scipy.linalg
import torch
import matplotlib.pyplot as plt
import os
import pickle
import hashlib
from main import GroundTruthProcess, train_model, DEVICE, N_STATES, SEQS

print("=" * 80)
print("EXPERIMENT 3: Branch Length Error Scaling Analysis")
print("=" * 80)

# Configuration
EPISTASIS = 0.5
SEED = 42
LAMBDA_PARAM = 0.5
N_SAMPLES = 1000000  # For training factorized model

# Branch lengths to test (log-spaced from 0.001 to 10.0 for wider range)
BRANCH_LENGTHS = np.logspace(-3, 0, 100)  # 100 points from 0.001 to 1

# Cache directory
USE_CACHE = False
CACHE_DIR = './exp3_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Use cache: {'YES' if USE_CACHE else 'NO'}")
print(f"  Epistasis: {EPISTASIS}")
print(f"  Training samples: {N_SAMPLES}")
print(f"  Branch lengths: {len(BRANCH_LENGTHS)} values from {BRANCH_LENGTHS[0]:.4f} to {BRANCH_LENGTHS[-1]:.2f}")

# Caching functions
def get_cache_key(epistasis, seed, n_samples):
    """Generate unique cache key based on parameters."""
    key_str = f"eps{epistasis}_seed{seed}_n{n_samples}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_dataset(epistasis, seed, n_samples):
    """Load cached dataset if available."""
    cache_key = get_cache_key(epistasis, seed, n_samples)
    cache_file = os.path.join(CACHE_DIR, f"dataset_{cache_key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_dataset(data, epistasis, seed, n_samples):
    """Save dataset to cache."""
    cache_key = get_cache_key(epistasis, seed, n_samples)
    cache_file = os.path.join(CACHE_DIR, f"dataset_{cache_key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def load_model(epistasis, seed, n_samples):
    """Load cached model if available."""
    cache_key = get_cache_key(epistasis, seed, n_samples)
    cache_file = os.path.join(CACHE_DIR, f"model_{cache_key}.pt")
    if os.path.exists(cache_file):
        return torch.load(cache_file, map_location=DEVICE)
    return None

def save_model(Q_matrix, epistasis, seed, n_samples):
    """Save model Q matrix to cache."""
    cache_key = get_cache_key(epistasis, seed, n_samples)
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

# Step 2: Train factorized model (with caching)
print(f"\n" + "=" * 80)
print("Step 2: Training Factorized Model")
print("=" * 80)

# Try to load cached dataset
train_data = load_dataset(EPISTASIS, SEED, N_SAMPLES) if USE_CACHE else None
if train_data is not None:
    print(f"✓ Loaded cached dataset ({len(train_data)} samples)")
else:
    print(f"Generating {N_SAMPLES} training samples...")
    train_data = gt.generate_data(N_SAMPLES, lambda_param=LAMBDA_PARAM, min_mutations=0)
    print(f"✓ Generated {len(train_data)} samples")
    # save_dataset(train_data, EPISTASIS, SEED, N_SAMPLES)
    # print(f"  Saved to cache")

# Try to load cached model
Q_factorized = load_model(EPISTASIS, SEED, N_SAMPLES) if USE_CACHE else None
if Q_factorized is not None:
    print(f"✓ Loaded cached factorized model")
    training_error = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)
    print(f"  Training error: {training_error.item():.4f}")
else:
    print(f"\nTraining factorized model (no SNR weighting)...")
    model, history = train_model(
        gt, train_data,
        use_snr=False,
        epochs=1000,
        lr=0.001,
        early_stop_patience=50,
        early_stop_tolerance=0.001,
        verbose=True,
    )

    Q_factorized = model.build_global_Q()
    training_error = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)
    print(f"✓ Factorized model trained")
    print(f"  Training error: {training_error.item():.4f}")

    save_model(Q_factorized, EPISTASIS, SEED, N_SAMPLES)
    print(f"  Saved to cache")

# Step 3: Test error scaling with branch length (fixed transition)
print(f"\n" + "=" * 80)
print("Step 3: Testing Error Scaling vs Branch Length")
print("=" * 80)

# Convert Q matrices to numpy for scipy
Q_true_np = Q_true.cpu().numpy()
Q_factorized_np = Q_factorized.cpu().numpy()

# Sample random start and end states
np.random.seed(SEED)
n_pairs = 100
start_states = np.random.randint(0, N_STATES, size=n_pairs)
end_states = np.random.randint(0, N_STATES, size=n_pairs)

print(f"Sampling {n_pairs} random (start, end) state pairs...")
print(f"Computing transition probabilities for {len(BRANCH_LENGTHS)} branch lengths...")

slopes = []
valid_pairs = 0

for i, (start_state, end_state) in enumerate(zip(start_states, end_states)):
    start_seq = SEQS[start_state]
    end_seq = SEQS[end_state]
    
    datum = []
    for b in BRANCH_LENGTHS:
        # Get true transition probability and factorized transition probability
        P_true = scipy.linalg.expm(b * Q_true_np)[start_state, end_state]
        P_factorized = scipy.linalg.expm(b * Q_factorized_np)[start_state, end_state]
        P_diff = np.abs(P_true - P_factorized)
        datum.append((b, P_diff))
    
    # Calculate slope of log-log transform
    datum_array = np.array(datum)
    log_b = np.log(datum_array[:, 0])
    log_P_diff = np.log(datum_array[:, 1] + 1e-20)  # Add small epsilon to avoid log(0)
    
    # Filter out invalid values (inf, nan)
    valid_mask = np.isfinite(log_b) & np.isfinite(log_P_diff)
    if np.sum(valid_mask) > 10:  # Need at least 10 points for reliable fit
        slope, intercept = np.polyfit(log_b[valid_mask], log_P_diff[valid_mask], 1)
        slopes.append(slope)
        valid_pairs += 1
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1}/{n_pairs} pairs...")

slopes = np.array(slopes)
print(f"\n✓ Computed slopes for {valid_pairs}/{n_pairs} valid pairs")
print(f"\nSlope Statistics:")
print(f"  Mean: {np.mean(slopes):.4f}")
print(f"  Median: {np.median(slopes):.4f}")
print(f"  Std: {np.std(slopes):.4f}")
print(f"  Min: {np.min(slopes):.4f}")
print(f"  Max: {np.max(slopes):.4f}")
print(f"\nTheoretical prediction: slope ≈ 2.0 (O(b²) scaling for factorized model with epistasis)")

# Create and save boxplot
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(8, 6))
plt.boxplot(slopes)
plt.axhline(y=2.0, color='r', linestyle='--', label='Theoretical O(b²) scaling')
plt.ylabel('Slope (log error vs log branch length)')
plt.title(f'Error Scaling Slopes\n(Epistasis={EPISTASIS}, n={valid_pairs} pairs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = f'plots/experiment_3_error_scaling_lambda{LAMBDA_PARAM}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to {plot_path}")

# Also save as PDF
pdf_path = f'plots/experiment_3_error_scaling_lambda{LAMBDA_PARAM}.pdf'
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✓ Plot saved to {pdf_path}")

plt.close()
