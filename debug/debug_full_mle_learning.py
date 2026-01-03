"""
Debug: Check if Full MLE is actually learning the correct rates.
Compare learned rates vs true rates for observed transitions.
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *
import matplotlib.pyplot as plt

print("=" * 70)
print("DEBUGGING: What is Full MLE actually learning?")
print("=" * 70)

# Test with one epistasis level
eps = 1.0
print(f"\nEpistasis = {eps}")
print("-" * 70)

gt = GroundTruthProcess(seed=42, epistasis=eps)
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

# Generate substantial data
print("Generating 50000 samples...")
data = gt.generate_data(50000, lambda_param=1.0, min_mutations=0)

# Track which transitions are observed
observed = {}  # (i,j) -> count
for start_seq, end_seq, b in data:
    i = seq_to_idx(tuple(start_seq.tolist()))
    j = seq_to_idx(tuple(end_seq.tolist()))
    if i != j and hamming(SEQS[i], SEQS[j]) == 1:
        observed[(i, j)] = observed.get((i, j), 0) + 1

print(f"Observed {len(observed)}/3072 transitions")

# Train model
print("\nTraining Full MLE model (200 epochs, no early stopping)...")
model_full, history = train_full_mle_model(
    gt, data,
    epochs=200,
    lr=0.01,
    bucket_decimals=2,
    early_stop_patience=999,  # Disable early stopping
    early_stop_tolerance=0.0001,
    verbose=False
)

Q_model = model_full.build_Q()

print(f"\nFinal error: {torch.norm(Q_model - Q_true) / torch.norm(Q_true):.4f}")

# Analyze: Compare learned vs true rates for observed transitions
print("\n" + "=" * 70)
print("ANALYSIS: Learned vs True Rates")
print("=" * 70)

# Separate by observation count
bins = {
    'high': [],  # Observed >100 times
    'medium': [],  # Observed 10-100 times  
    'low': [],  # Observed 1-10 times
    'never': []  # Never observed
}

for i in range(N_STATES):
    for j in range(N_STATES):
        if i != j and hamming(SEQS[i], SEQS[j]) == 1:
            true_rate = Q_true[i, j].item()
            pred_rate = Q_model[i, j].item()
            count = observed.get((i, j), 0)
            
            if count > 100:
                bins['high'].append((true_rate, pred_rate, count))
            elif count >= 10:
                bins['medium'].append((true_rate, pred_rate, count))
            elif count >= 1:
                bins['low'].append((true_rate, pred_rate, count))
            else:
                bins['never'].append((true_rate, pred_rate, count))

# Print stats
for bin_name in ['high', 'medium', 'low', 'never']:
    if bins[bin_name]:
        true_rates = [x[0] for x in bins[bin_name]]
        pred_rates = [x[1] for x in bins[bin_name]]
        errors = [abs(t - p) for t, p in zip(true_rates, pred_rates)]
        
        print(f"\n{bin_name.upper()} frequency ({len(bins[bin_name])} transitions):")
        print(f"  Mean abs error: {np.mean(errors):.4f}")
        print(f"  Median abs error: {np.median(errors):.4f}")
        print(f"  Mean true rate: {np.mean(true_rates):.4f}")
        print(f"  Mean pred rate: {np.mean(pred_rates):.4f}")

# Plot: True vs Predicted for observed transitions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, bin_name in enumerate(['high', 'medium', 'low']):
    if bins[bin_name]:
        true_rates = [x[0] for x in bins[bin_name]]
        pred_rates = [x[1] for x in bins[bin_name]]
        
        axes[idx].scatter(true_rates, pred_rates, alpha=0.5, s=10)
        axes[idx].plot([0, max(true_rates)], [0, max(true_rates)], 'r--', label='Perfect fit')
        axes[idx].set_xlabel('True Rate')
        axes[idx].set_ylabel('Predicted Rate')
        axes[idx].set_title(f'{bin_name.capitalize()} frequency\n({len(bins[bin_name])} transitions)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_full_mle_scatter.png', dpi=150)
print("\nSaved debug_full_mle_scatter.png")

# Check if there's systematic bias
print("\n" + "=" * 70)
print("SYSTEMATIC BIAS CHECK")
print("=" * 70)

# Check: Does the model systematically under/over-estimate?
all_observed = bins['high'] + bins['medium'] + bins['low']
if all_observed:
    true_rates = np.array([x[0] for x in all_observed])
    pred_rates = np.array([x[1] for x in all_observed])
    
    bias = np.mean(pred_rates - true_rates)
    correlation = np.corrcoef(true_rates, pred_rates)[0, 1]
    
    print(f"Bias (mean(pred - true)): {bias:+.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    # Check if model is regressing toward mean
    mean_true = np.mean(true_rates)
    mean_pred = np.mean(pred_rates)
    print(f"Mean true rate: {mean_true:.4f}")
    print(f"Mean pred rate: {mean_pred:.4f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if bins['never']:
    print(f"✗ {len(bins['never'])} transitions NEVER observed")
    print(f"  These will have poorly estimated rates")

if bins['low']:
    print(f"⚠ {len(bins['low'])} transitions observed rarely (1-10 times)")
    print(f"  These may have high variance estimates")

print(f"\n✓ {len(bins['high']) + len(bins['medium'])} transitions well-observed")

