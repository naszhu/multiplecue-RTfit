# Analysis: Why Mixture Model Shows Unimodal Predictions

## Problem Identified

The fitted MIS-LBA mixture model shows unimodal predictions even though:
1. The model is designed as a mixture (express responses + LBA)
2. The observed data shows bimodal RT distributions for some cue conditions

## Root Causes

### 1. Express Component Collapsed to Near-Zero

**Finding**: For most cue conditions (2-9), the fitted `ProbExp` parameter is essentially zero (~1e-13), meaning the optimizer found that express responses are not needed to fit the data.

**Evidence from `model_fit_results.csv`**:
- Condition 1: `ProbExp = 0.0189` (1.9% express responses)
- Condition 10: `ProbExp = 0.0105` (1.1% express responses)  
- Conditions 2-9: `ProbExp ≈ 1e-13` (essentially zero)

**Why this happens**:
- The optimization algorithm minimizes negative log-likelihood
- If the LBA component alone can fit the data well, the optimizer will set `p_exp` to zero
- This is a valid solution if the data doesn't require a bimodal mixture

### 2. Plotting Function Averaging Issue

**Original Problem**: The plotting function was:
- Averaging PDFs across trials with different reward structures
- This averaging can smooth out bimodality even when it exists
- Not showing the express and LBA components separately

**Solution**: Updated plotting function now:
- Computes the unconditional PDF by properly weighting unique reward structures
- Shows express and LBA components separately
- Displays mixture parameters in the title
- Warns when express component has collapsed

## Model Implementation is Correct

The mixture model implementation in `model_utils.jl` is mathematically correct:
```julia
lik_tot = (p_exp * lik_exp) + ((1-p_exp) * lik_reg)
```

The issue is that the **optimization is finding that `p_exp ≈ 0`** for most conditions, which means:
- The model is correctly implementing a mixture
- But the data for most conditions doesn't require the express component
- The optimizer is correctly identifying that a single LBA component is sufficient

## When Bimodality Should Appear

Bimodality should be visible when:
1. `p_exp` is non-negligible (> 0.01)
2. `mu_exp` (express mean) is well-separated from the LBA mode (typically > 100ms separation)
3. The express component has sufficient weight in the mixture (express peak density > 10% of main peak)

## Why Condition 10 Appears Unimodal Despite Non-Zero Express Probability

**Condition 10 Parameters**:
- `ProbExp = 0.0105` (only 1.05% of responses)
- `MuExp = 0.200s` (express mean)
- `t0 = 0.265s` (non-decision time, LBA starts here)

**Problem**: Even though `p_exp > 0`, the express component is:
1. **Too small**: Only 1% of responses, so its contribution is minimal
2. **Too close to LBA mode**: Express mean (0.2s) is only 65ms before t0 (0.265s), not well-separated
3. **Visually lost**: When combined with the 99% LBA component, the 1% express component doesn't create a visible second mode

**Result**: The total mixture PDF appears unimodal because the express component is too small and too close to the main LBA mode to create visible bimodality.

**Solution**: The updated plotting function now:
- Shows component separation diagnostics
- Warns when express component is too small/close to create visible bimodality
- Displays peak locations with vertical lines
- Reports separation distance and peak densities

## Recommendations

1. **Check conditions 1 and 10**: These have non-zero express probabilities - the updated plots should show bimodality for these conditions.

2. **Re-examine optimization**: If you expect express responses but the optimizer sets `p_exp ≈ 0`, consider:
   - Different initial values for `p_exp` (try starting at 0.1-0.2)
   - Constraining `p_exp` to be > 0.01 (modify lower bound)
   - Checking if the data truly has a fast RT mode that needs explaining

3. **Model comparison**: Compare models with and without express component using:
   - AIC/BIC for model comparison
   - Visual inspection of residuals
   - Cross-validation

4. **Use the improved plots**: The updated plotting function now shows:
   - Separate express and LBA components
   - Warning when express component collapses (p_exp ≈ 0)
   - Warning when express component is too small/close to create visible bimodality
   - Vertical lines at component peak locations
   - Component separation diagnostics
   - Mixture parameters in the title

## Next Steps

1. Re-run the fitting with the updated plotting function
2. Examine plots for conditions 1 and 10 (which have non-zero express probabilities)
3. Consider whether the optimization constraints need adjustment if express responses are theoretically expected

