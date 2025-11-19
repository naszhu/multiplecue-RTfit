# Dual-LBA Mixture Model

## Overview

This model uses **TWO LBA components** with different parameters to capture bimodality, rather than LBA + express responses. This is more appropriate when the observed bimodality is within the normal decision-making range (both modes > 0.3s) rather than from express responses.

## Model Structure

### Components

1. **LBA Component 1 (Fast Mode)**
   - Parameters: `A1`, `k1`, `t0_1`
   - Typically has: Lower thresholds, faster non-decision time
   - Captures the faster RT mode

2. **LBA Component 2 (Slow Mode)**
   - Parameters: `A2`, `k2`, `t0_2`
   - Typically has: Higher thresholds, slower non-decision time
   - Captures the slower RT mode

3. **Mixing Probability**
   - `p_mix`: Probability of using Component 1
   - `1 - p_mix`: Probability of using Component 2

### Shared Parameters

Both components share:
- `C`: Capacity parameter (drift rate scaling)
- `w_slope`: Reward weight slope (MIS theory)
- Same drift rates computed from rewards (MIS theory)

## Parameters (9 total)

1. `C` - Capacity (drift rate scaling)
2. `w_slope` - Reward weight slope
3. `A1` - Start point variability for Component 1 (fast)
4. `k1` - Threshold gap for Component 1 (fast)
5. `t0_1` - Non-decision time for Component 1 (fast)
6. `A2` - Start point variability for Component 2 (slow)
7. `k2` - Threshold gap for Component 2 (slow)
8. `t0_2` - Non-decision time for Component 2 (slow)
9. `p_mix` - Probability of using Component 1

## When to Use This Model

Use the **dual-LBA model** when:
- ✅ Observed bimodality has both modes > 0.3s (within normal LBA range)
- ✅ Bimodality appears to come from different decision strategies/processes
- ✅ The express model (`p_exp ≈ 0`) suggests express responses aren't needed
- ✅ You want to model two different decision-making modes within LBA framework

Use the **express model** when:
- ✅ Fast mode is < 0.3s (true express responses)
- ✅ You want to model automatic/reflexive responses separately

## Model Comparison

### Express Model (Original)
- **Components**: LBA + Normal distribution (express)
- **Parameters**: 8 total
- **Assumption**: Fast mode is express (< 0.3s)
- **When it works**: When there are true express responses

### Dual-LBA Model (New)
- **Components**: LBA1 (fast) + LBA2 (slow)
- **Parameters**: 9 total
- **Assumption**: Both modes are within normal LBA range
- **When it works**: When bimodality comes from different LBA processes

## Usage

Run the dual-LBA model:
```bash
julia fit-mis-model-dual.jl
```

This will:
1. Fit the dual-LBA model for each cue condition
2. Save results to `model_fit_results_dual_condition_X.csv`
3. Generate plots showing both LBA components: `model_fit_plot_dual_condition_X.png`
4. Create combined results file: `model_fit_results_dual.csv`

## Output Files

- `model_fit_results_dual.csv` - Combined results for all conditions
- `model_fit_results_dual_condition_X.csv` - Individual results for condition X
- `model_fit_plot_dual_condition_X.png` - Plot showing both LBA components for condition X

## Plot Interpretation

The plots show:
- **Orange dashed line**: LBA Component 1 (fast mode)
- **Green dashed line**: LBA Component 2 (slow mode)
- **Red solid line**: Total mixture (weighted combination)
- **Vertical lines**: Peak locations for each component

## Expected Results

For conditions with clear bimodality (like condition 10):
- `p_mix` should be between 0.2-0.8 (both components used)
- `t0_1` < `t0_2` (fast component has faster non-decision time)
- Component separation should be > 100ms
- Both components should be visible in the plot

## Advantages

1. **More appropriate for observed data**: Captures bimodality within LBA framework
2. **No express assumption**: Doesn't assume fast mode is express (< 0.3s)
3. **Flexible**: Allows different thresholds and non-decision times for each mode
4. **Interpretable**: Two LBA processes can represent different strategies or conditions

## Model Selection

Compare both models using:
- **AIC/BIC**: Lower is better
- **Visual fit**: Which captures the bimodality better?
- **Parameter interpretability**: Which makes more theoretical sense?

The dual-LBA model is likely better when:
- Observed fast mode is > 0.3s
- Express model sets `p_exp ≈ 0`
- Bimodality appears to come from different decision processes

