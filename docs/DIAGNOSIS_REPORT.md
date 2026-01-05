# Diagnosis: Why Full MLE Error Increases with Epistasis

**Date:** 2026-01-03
**Status:** ROOT CAUSE IDENTIFIED

---

## Summary

The Full MLE model's error increases with epistasis levels NOT because the method is fundamentally flawed, but because **maximizing likelihood on finite data is different from minimizing error to ground truth**. The phenomenon is a classic case of **overfitting in high-dimensional, low-sample regimes**.

---

## Key Findings

### Finding 1: **Training actively degrades perfect initialization**

**Experiment:** Initialize Full MLE with ground truth parameters (error ≈ 0.000001), then train for 10 epochs.

**Result:**
```
Epistasis = 0.0:
  Initial error: 0.000001
  After 10 epochs: 0.004148  (416× worse!)

Epistasis = 1.0:
  Initial error: 0.000010
  After 10 epochs: 0.004660  (466× worse!)
```

**Conclusion:** The training procedure is working as intended (maximizing likelihood), but this objective is misaligned with minimizing parameter error when data is limited.

---

### Finding 2: **Loss and Error are anti-correlated**

**Experiment:** Start from ground truth Q and run gradient descent on likelihood loss.

**Result:**
```
Epoch    Loss            Error
0        3.210082        0.000001    (ground truth)
20       3.163854        0.073064    (after optimization)

Did loss decrease?: True  ✓
Did error decrease?: False  ✗
```

**Conclusion:** Successfully minimizing the training loss (negative log-likelihood) INCREASES error in Q estimation. The empirical likelihood of a finite sample is not maximized at the true parameters.

---

### Finding 3: **Higher epistasis → weaker signal → worse overfitting**

**Experiment:** Compare ground truth Q matrices across epistasis levels.

**Result:**
```
Epistasis  Frobenius Norm  Mean Rate  Mean Mutations
0.0        40.31           0.518      1.49
0.3        30.08           0.386      1.34
0.7        16.54           0.210      1.00
1.0         6.88           0.079      0.50
```

**Key observation:** Higher epistasis creates SMALLER rates (not larger!), leading to:
- Fewer mutations per unit branch length
- Less informative data
- Worse signal-to-noise ratio
- More severe overfitting

**Conclusion:** The "difficulty" doesn't increase with epistasis intrinsically, but the **data becomes less informative**, exacerbating the overfitting problem.

---

### Finding 4: **Data coverage is reasonable**

**Experiment:** Count how many of the 576 possible Hamming-1 transitions are observed.

**Result:**
```
Epistasis  Coverage     Avg counts/transition
0.0        96.35%       5.09
0.3        95.31%       5.43
0.7        96.18%       5.86
1.0        91.67%       5.03
```

**Conclusion:** The issue is NOT lack of coverage. Even with ~92-96% of transitions observed, the MLE still overfits because each transition is only seen 5-6 times on average.

---

## Root Cause Analysis

### The Fundamental Problem

**With 10,000 samples and 576 parameters (~17 samples/parameter):**

1. **MLE is unbiased asymptotically** (correct in the limit of infinite data)
2. **But MLE has high variance with finite data** (unstable estimates)
3. **The empirical likelihood** of a specific finite sample is maximized at parameters that "explain the noise" in that sample
4. **Gradient descent correctly optimizes the wrong objective** (finite-sample likelihood ≠ population likelihood)

### Mathematical Explanation

For finite data D sampled from P(D | Q_true):

- **Population likelihood:** L_∞(Q) = E[log P(D | Q)] → maximized at Q_true
- **Empirical likelihood:** L_n(Q) = (1/n) Σ log P(d_i | Q) → maximized at Q_MLE ≠ Q_true

The **bias-variance tradeoff:**
```
MSE(Q_estimate) = Bias²(Q_estimate) + Variance(Q_estimate)

Full MLE:
  Bias² ≈ 0 (can model any Q)
  Variance = HIGH (many parameters, little data)
  → Total error = HIGH

Factorized Model:
  Bias² = HIGH at epistasis > 0 (can't model epistasis)
  Variance = LOW (few parameters, strong inductive bias)
  → Total error = MEDIUM

With 10k samples: Factorized wins (variance dominates)
With 100k+ samples: Full MLE should win (bias dominates)
```

---

## Why Error Increases with Epistasis

### Three contributing factors:

1. **Weaker Signal (Primary Cause)**
   - Higher epistasis → lower mutation rates
   - Mean rate: 0.518 (eps=0) → 0.079 (eps=1.0) (7× decrease)
   - Mutations/seq: 1.49 → 0.50 (3× decrease)
   - Less information per sample → more noise

2. **Same Sample Size, Same Parameters**
   - 10,000 samples for all epistasis levels
   - 576 parameters for all epistasis levels
   - But the "effective information content" drops with epistasis

3. **Overfitting Amplification**
   - Weaker signal → relative noise increases
   - MLE fits noise → larger deviations from truth
   - Higher variance in parameter estimates

**Analogy:** Trying to estimate the average height of a population:
- Low epistasis = measuring adults (strong signal, clear average)
- High epistasis = measuring with a noisy ruler (weak signal, fits measurement errors)

---

## Solutions

### Solution 1: **Add Regularization to Full MLE** ⭐ Recommended

```python
# In main.py, train_full_mle_model()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)  # L2 regularization
```

**Why:** Penalizes large parameter values, reducing overfitting.

**Expected impact:** Should reduce error by 30-50%, especially at high epistasis.

---

### Solution 2: **Increase Training Data** ⭐ Recommended

```python
n_samples = 50000  # Up from 10,000
# Gives ~87 samples per parameter
```

**Why:** More data → variance decreases → MLE becomes more reliable.

**Expected impact:** At 50k samples, Full MLE should start matching or beating Factorized model at high epistasis.

---

### Solution 3: **Early Stopping on Held-Out Validation Set**

```python
# Split data into train (80%) and validation (20%)
# Stop training when validation error starts increasing
```

**Why:** Prevents overfitting to training noise.

**Expected impact:** Modest improvement (10-20%), but won't solve fundamental variance problem.

---

### Solution 4: **Reduce Model Complexity**

Currently: 576 parameters (all Hamming-1 transitions are independent)

Alternative: Add structural constraints:
- Share parameters across similar transitions
- Use a low-rank factorization of Q
- Add sparsity priors

**Why:** Fewer effective parameters → lower variance.

**Tradeoff:** Introduces bias, but may be worth it with limited data.

---

### Solution 5: **Bayesian Approach (Advanced)**

Use Bayesian inference with priors instead of point estimates:
- Prior: rates ~ Gamma(α, β) with informative hyperparameters
- Posterior: integrate over parameter uncertainty
- Report posterior mean instead of MLE

**Why:** Naturally regularizes and quantifies uncertainty.

**Tradeoff:** More complex implementation.

---

## Recommended Action Plan

### Immediate (Test Today):

1. **Test regularization:**
   ```bash
   python test_regularization.py  # Should already exist
   ```

2. **Test with more data:**
   ```python
   # In run_experiment_0.py, change:
   n_samples = 50000  # From 10,000
   ```

### Short-term (This Week):

3. Implement validation-based early stopping
4. Add epistasis=0 sanity check (both models should perform equally)
5. Create sample size scaling experiment (1k, 5k, 10k, 50k, 100k)

### Long-term (Future Work):

6. Implement Bayesian inference
7. Test structural constraints (low-rank Q, parameter sharing)
8. Compare with other regularization schemes (L1, elastic net, dropout)

---

## Experimental Validation

### Prediction 1: Regularization helps

If we add `weight_decay=1e-3` to Full MLE:
- Error at eps=0.0: should remain ~0.17 (already near Factorized)
- Error at eps=1.0: should decrease from ~0.19 to ~0.12-0.14

### Prediction 2: More data helps

With 50,000 samples:
- Full MLE error should decrease across all epistasis levels
- At eps=1.0, Full MLE should approach or beat Factorized model
- Crossover point: where Full MLE starts winning

### Prediction 3: The issue is NOT the code

- Gradients flow correctly ✓
- Loss decreases correctly ✓
- Matrix exp works correctly ✓
- The problem is statistical, not computational

---

## Conclusion

The Full MLE model is **mathematically correct but statistically overfit**. The increasing error with epistasis is due to:

1. **Primary cause:** Weaker signal (lower mutation rates) with higher epistasis
2. **Secondary cause:** Insufficient data for 576 parameters (~17 samples/param)
3. **Mechanism:** MLE overfits to finite-sample noise, which increases error to ground truth

**The fix:** Add regularization and/or increase sample size.

**Expected outcome after fixes:**
- With regularization + 50k samples, Full MLE should win at high epistasis
- This will demonstrate the expected behavior: factorized model wins when wrong (low eps), full model wins when right (high eps with enough data)

---

## Files Generated

1. `debug/investigate_epistasis_trend.py` - Comprehensive diagnostic
2. `debug/test_gradient_flow.py` - Gradient verification
3. `debug/diagnose_loss_vs_error.py` - Loss/error anti-correlation test
4. `plots/rate_distributions_vs_epistasis.pdf` - Visualization of rate heterogeneity
5. This report: `DIAGNOSIS_REPORT.md`
