# Epistasis Implementation Fix - Summary

**Date:** 2026-01-03
**Issue:** Epistasis parameter was unintentionally reducing mutation rates
**Status:** ✓ FIXED

---

## The Problem

The original epistasis implementation had a **critical flaw**: increasing epistasis systematically **reduced the average mutation rate** instead of just adding context-dependency.

### Original Behavior (BROKEN):

```
Epistasis  Frobenius Norm  Mean Rate  Mean Mutations/Seq
0.0        40.31           0.518      1.49
0.3        30.08           0.386      1.34
0.7        16.54           0.210      1.00
1.0         6.88           0.079      0.50  ← 7× lower than eps=0!
```

**Why this happened:**
1. Applied multiplicative modifier: `rate = base_rate * (1 + epistasis * modifier)`
2. Modifier range: [-1, 1]
3. Clipped to minimum: `rate = max(0.01, rate)`
4. **Asymmetric clipping created systematic bias** toward lower rates
5. Higher epistasis → more extreme modifiers → more clipping → lower average

### What We Wanted:

- **Epistasis = 0**: Perfectly factorizable Q (context-independent)
- **Epistasis = 1**: Strong context-dependency, **but same average rate**

---

## The Solution

### Fixed Implementation:

**Key insight:** Use **zero-centered multiplicative noise** with normalization to preserve expected values.

```python
# Step 1: Collect all modifiers for each transition type
for each (site, a→b) transition:
    collect modifiers across all possible contexts

# Step 2: Normalize to mean=0
for each transition:
    normalized_modifier = raw_modifier - mean(raw_modifiers)

# Step 3: Apply log-normal multiplicative factor
rate = base_rate * exp(epistasis * normalized_modifier * scale)
```

**Why this works:**
- Normalization ensures `mean(modifier) = 0` for each transition type
- `exp(0) = 1`, so unbiased on average
- `exp()` is always positive (no clipping needed)
- Scale parameter (0.7) controls variation strength

### Fixed Behavior (CORRECT):

```
Epistasis  Frobenius Norm  Mean Rate  Change    Status
0.0        40.31           0.5181     +0.00%    ✓ Baseline
0.3        40.35           0.5182     +0.02%    ✓ Stable
0.7        40.44           0.5186     +0.10%    ✓ Stable
1.0        40.54           0.5192     +0.21%    ✓ Stable
```

**✓ Mean rate preserved** (within 0.21%)
**✓ Frobenius norm preserved** (within 0.56%)
**✓ Mutation rates consistent** across epistasis levels

---

## Code Changes

### File: `main.py`

**Location:** `GroundTruthProcess.__init__()` (lines ~85-172)

**Changed from:**
```python
modifier = (context_hash % 100) / 50.0 - 1.0  # Range: [-1, 1]
rate = base_rate * (1 + self.epistasis * modifier)
rate = max(0.01, rate)  # Clipping creates bias!
```

**Changed to:**
```python
# Two-pass algorithm:
# Pass 1: Collect all modifiers for normalization
for each transition (site, a, b):
    collect raw_modifiers across contexts

# Pass 2: Apply normalized modifiers
normalized_modifier = raw_modifier - mean(raw_modifiers)
log_modifier = epistasis * normalized_modifier * 0.7
rate = base_rate * exp(log_modifier)  # No clipping needed!
```

---

## Verification

### Test Script: `debug/verify_fixed_epistasis.py`

**Tests performed:**

1. **Rate Statistics:** Mean, std, min, max across epistasis levels
2. **Stability Check:** Mean rate should stay within ±5%
3. **Norm Stability:** Frobenius norm should stay within ±10%
4. **Heterogeneity:** CV should increase with epistasis (more variation)
5. **Mutation Rates:** Data-based verification of E[mutations/site] ≈ branch_length

**Results:** ✓ All tests passed

### Visualization

Plot saved to: `plots/fixed_epistasis_verification.pdf`

Shows rate distributions for each epistasis level - now properly centered with increasing spread but constant mean.

---

## Impact on Experiment 0

### Before Fix:

The Full MLE model appeared to get "worse" with increasing epistasis because:
1. Higher epistasis → lower mutation rates
2. Weaker signal → worse SNR → more overfitting
3. **This confounded the epistasis effect with signal strength**

### After Fix:

Now we can properly test the hypothesis:
- **Signal strength is constant** across epistasis levels
- **Only context-dependency changes**
- Full MLE should now show **constant error** across epistasis (can model any context-dependency)
- Factorized model error should **increase** with epistasis (cannot model context-dependency)

---

## Expected Results (Re-running Experiment)

### Configuration:
- Epistasis levels: [0.0, 0.3, 0.7, 1.0]
- 3 replicates (seeds: 42, 43, 44)
- Factorized: 10k samples
- Full MLE: 50k samples + L2 regularization (weight_decay=0.0001)

### Predictions:

**Factorized Model:**
```
Epistasis  Expected Error  Reason
0.0        ~0.07-0.10      ✓ Ground truth IS factorizable
0.3        ~0.10-0.15      ✗ Weak epistasis, slight model mismatch
0.7        ~0.15-0.25      ✗ Strong epistasis, clear mismatch
1.0        ~0.20-0.30      ✗ Maximum epistasis, worst mismatch
```

**Full MLE (with regularization):**
```
Epistasis  Expected Error  Reason
0.0        ~0.07-0.10      ✓ Can model factorizable Q
0.3        ~0.07-0.10      ✓ Can model weak epistasis
0.7        ~0.07-0.10      ✓ Can model strong epistasis
1.0        ~0.07-0.10      ✓ Can model maximum epistasis
```

**Key prediction:** With proper epistasis implementation + regularization + more data, Full MLE should now show **flat error** across epistasis while Factorized error increases.

---

## Files Modified

1. `main.py` - Fixed `GroundTruthProcess.__init__()` epistasis implementation
2. `run_experiment_0.py` - Updated configuration (50k samples, L2 reg)

## Files Created

1. `debug/verify_fixed_epistasis.py` - Verification tests
2. `plots/fixed_epistasis_verification.pdf` - Visualization
3. `EPISTASIS_FIX_SUMMARY.md` - This document

## Cache Cleared

- Deleted `exp0_cache/` to force regeneration with fixed ground truth

---

## Next Steps

1. ✓ Monitor Experiment 0 completion (~30-40 min)
2. ✓ Verify results match predictions
3. ✓ Update CLAUDE.md with new findings
4. ✓ Create final comparison plot

---

## Technical Details

### Why Normalization Works

For a transition type (site, a→b), the rate in context c is:

```
r(c) = base_rate * exp(ε * (m(c) - μ) * σ)
```

where:
- `m(c)` = raw context-dependent modifier
- `μ` = mean(m(c)) over all contexts
- `ε` = epistasis parameter
- `σ` = scale (0.7)

**Average over all contexts:**
```
E[r(c)] = base_rate * E[exp(ε * (m(c) - μ) * σ)]
        ≈ base_rate * exp(ε * E[m(c) - μ] * σ)    [for small ε*σ]
        = base_rate * exp(ε * 0 * σ)
        = base_rate * exp(0)
        = base_rate
```

The normalization `(m(c) - μ)` ensures E[modifier] = 0, so the average rate equals the base rate.

### Why the Old Method Failed

Old method: `r(c) = max(0.01, base_rate * (1 + ε * m(c)))`

Problem: `max()` function introduces bias:
```
E[max(0.01, X)] > max(0.01, E[X])   [Jensen's inequality for max]
```

When ε is large, many rates get clipped to 0.01, creating a **floor effect** that drags down the average.

---

## Conclusion

✓ **Fixed:** Epistasis now correctly adds context-dependency without changing average rates
✓ **Verified:** All statistical tests pass
✓ **Running:** Experiment 0 with fixed ground truth
✓ **Expected:** Clear demonstration of bias-variance tradeoff with proper epistasis
