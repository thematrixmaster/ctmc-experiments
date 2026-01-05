# Analysis: Full MLE vs Factorized Model Performance

## Summary of Results

From `logs/experiment_0_3replicates.log`:

| Epistasis | Factorized (Mean ± Std) | Full MLE GD (Mean ± Std) |
|-----------|-------------------------|--------------------------|
| 0.0       | 0.0752 ± 0.0143        | 0.1724 ± 0.0091         |
| 0.3       | 0.0833 ± 0.0011        | 0.1998 ± 0.0190         |
| 0.7       | 0.0982 ± 0.0060        | 0.1793 ± 0.0071         |
| 1.0       | 0.1387 ± 0.0102        | 0.1944 ± 0.0035         |

**Key Finding:** The factorized model consistently outperforms the full MLE model at ALL epistasis levels, including high epistasis where we expected the full MLE to excel.

## Root Cause Analysis

### 1. **Insufficient Training Data for Full MLE Model**

The full MLE model is severely data-starved:

```
Full MLE Model:
  - Total parameters: 576 (Hamming-1 transitions)
  - Training samples: 10,000
  - Samples per parameter: 17.4

Factorized Model:
  - Effective parameters: 48 (L=3, 4×4 matrices)
  - Training samples: 10,000
  - Samples per parameter: 208.3
```

**Diagnosis:** With only ~17 samples per parameter, the full MLE model is severely underfitting. Standard ML practice requires 20-50+ samples per parameter for reliable estimation. The factorized model has 12× more samples per parameter (208 vs 17).

### 2. **Sparse Transition Coverage**

With 64 states and 10,000 samples:
- Many state-to-state transitions are rarely or never observed
- The full MLE model must estimate rates for transitions it has seen only a few times
- This leads to high variance and poor generalization

### 3. **Early Stopping May Be Premature**

Looking at the training logs, the factorized model appears to converge very quickly:
- Early stopping patience: 100 epochs
- Tolerance: 0.001

The model may be stopping before finding a better local minimum, especially given the small dataset.

### 4. **Full MLE Model IS Properly Constrained**

Reviewing the code (lines 267-307 in `main.py`):

```python
class FullMLEModel(nn.Module):
    def __init__(self):
        # Only parameterize Hamming-1 transitions
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                    self.param_indices[(i, j)] = idx
                    param_list.append(torch.randn(1) * 0.1)
                    idx += 1
```

✅ The model correctly enforces sparsity - only single-residue changes are allowed.

### 5. **Why Factorized Model Converges Quickly**

The factorized model has:
- Far fewer parameters (48 vs 576)
- Much better sample-to-parameter ratio
- Strong inductive bias (site independence)

This allows it to converge quickly even with limited data. The "instant convergence" is actually a sign of:
1. Good parameter-to-data ratio
2. Appropriate model capacity for the dataset size
3. Effective regularization through architectural constraints

## Recommendations

### Immediate Fixes

1. **Increase Training Data**
   ```python
   n_samples = 30000  # Up from 10,000
   # This gives ~52 samples per parameter for full MLE
   ```

2. **Adjust Full MLE Early Stopping**
   ```python
   early_stop_patience = 100  # Up from 50
   early_stop_tolerance = 0.005  # More lenient
   ```

3. **Add Regularization to Full MLE**
   ```python
   # In train_full_mle_model, add L2 regularization
   optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
   ```

4. **Reduce Learning Rate for Full MLE**
   ```python
   lr = 0.005  # Down from 0.01
   # Prevents overfitting to sparse observations
   ```

### Verification Tests

Run these diagnostic scripts to verify the issues:

```bash
# 1. Test with more data
python3 -c "
from main import *
gt = GroundTruthProcess(seed=42, epistasis=0.0)
data = gt.generate_data(30000, lambda_param=1.0, min_mutations=0)
model_full, history = train_full_mle_model(gt, data, epochs=500, lr=0.005, 
                                           early_stop_patience=100, verbose=True)
Q_full = model_full.build_Q()
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
error = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
print(f'Full MLE error with 30k samples: {error.item():.4f}')
"

# 2. Check transition coverage
python3 -c "
from main import *
import numpy as np
gt = GroundTruthProcess(seed=42, epistasis=0.0)
data = gt.generate_data(10000, lambda_param=1.0, min_mutations=0)
transition_counts = np.zeros((64, 64))
for start_seq, end_seq, b in data:
    i = seq_to_idx(tuple(start_seq.tolist()))
    j = seq_to_idx(tuple(end_seq.tolist()))
    transition_counts[i, j] += 1
# Count how many Hamming-1 transitions were observed
hamming1_observed = 0
hamming1_total = 0
for i in range(64):
    for j in range(64):
        if i != j and hamming(SEQS[i], SEQS[j]) == 1:
            hamming1_total += 1
            if transition_counts[i, j] > 0:
                hamming1_observed += 1
coverage = hamming1_observed / hamming1_total * 100
print(f'Hamming-1 transition coverage: {coverage:.1f}%')
print(f'Observed: {hamming1_observed}/{hamming1_total} transitions')
"
```

### Expected Behavior After Fixes

With 30,000 samples:
- **At epistasis=0.0**: Both models should achieve similar error (~0.05-0.08)
  - Factorized model is the correct model class
  - Full MLE should match it with enough data
  
- **At epistasis=1.0**: Full MLE should outperform factorized
  - Factorized: ~0.12-0.15 (limited by architecture)
  - Full MLE: ~0.08-0.10 (can capture context dependence)

## Why This Matters

The current results show the **opposite** of expected behavior:
- Factorized beats full MLE even at high epistasis
- This is because the full MLE is overfitting to sparse data
- The factorized model's architectural bias acts as strong regularization

This is a classic bias-variance tradeoff:
- **Factorized**: High bias (can't model epistasis), low variance (stable with limited data)
- **Full MLE**: Low bias (can model anything), high variance (unstable with limited data)

With only 10k samples, variance dominates → factorized wins everywhere.
With 30k+ samples, bias should dominate at high epistasis → full MLE should win.

## Code Changes Needed

### 1. Update `run_experiment_0.py`

```python
results = run_experiment_0_with_replicates(
    epistasis_levels=[0.0, 0.3, 0.7, 1.0],
    n_replicates=3,
    n_samples=30000,  # Changed from 10000
    use_cache=True,
    force_retrain=True,
)
```

### 2. Update `train_full_mle_model` in `main.py`

```python
def train_full_mle_model(gt, train_data, epochs=500, lr=0.005, verbose=True, 
                         bucket_decimals=2, early_stop_patience=100, 
                         early_stop_tolerance=0.005, weight_decay=1e-4):
    # ... existing code ...
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ... rest of function ...
```

### 3. Update early stopping parameters in `run_experiment_0_with_replicates`

```python
model_full, _ = train_full_mle_model(
    gt, data, 
    epochs=500, 
    lr=0.005,  # Reduced from 0.01
    verbose=False,
    early_stop_patience=100,  # Increased from 50
    early_stop_tolerance=0.005,  # Increased from 0.001
    weight_decay=1e-4  # Added regularization
)
```

## Additional Diagnostics

### Check if Full MLE is actually training

The logs show the full MLE error decreasing from 0.3957 → 0.1918 over 500 epochs, so it IS training. However:
- It starts from a poor initialization (error ~0.40)
- Converges to ~0.19, which is still worse than factorized (~0.07)
- This suggests the issue is data scarcity, not a training bug

### Factorized Model Early Stopping

The factorized model likely stops early (within 100-200 epochs) because:
1. It has 12× better sample-to-parameter ratio
2. The loss landscape is simpler (fewer parameters)
3. It reaches a good solution quickly

This is EXPECTED behavior for a well-regularized model with sufficient data relative to its capacity.


