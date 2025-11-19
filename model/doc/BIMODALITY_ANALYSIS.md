# Understanding Bimodality: Express Responses vs. LBA Parameter Variation

## Your Important Observation

You've noticed that the **observed data** shows clear bimodality for condition 10:
- **First mode**: Lower RT (faster responses)
- **Second mode**: Slightly bigger RT (slower responses), and appears larger

But the **model fit** shows the express component is too small/close to create visible bimodality. This raises a critical question:

## Is the Fast Mode Really "Express" Responses?

**Probably NOT!** Here's why:

### What "Express" Responses Should Look Like

True express responses are:
- **Very fast**: Typically < 0.25-0.3 seconds
- **Automatic/reflexive**: Bypass normal decision-making
- **Small proportion**: Usually 5-15% of responses
- **Well-separated**: At least 100-150ms before the main LBA mode

### What You're Observing

Looking at the observed data for condition 10:
- If the first mode is around **0.3-0.4 seconds** and the second is around **0.5-0.6 seconds**
- Neither mode is truly "express" (< 0.3s)
- Both modes are likely within the **normal LBA decision-making range**

## Alternative Explanations for Bimodality

The bimodality in your observed data is likely **NOT from express responses**, but from **within-LBA variation**:

### 1. **Different Reward Structures Within Condition**

Condition 10 might contain multiple cue configurations (different reward patterns):
- Some trials have high-reward cues → faster drift rates → faster RTs (first mode)
- Some trials have low-reward cues → slower drift rates → slower RTs (second mode)
- When averaged, this creates bimodality **within the LBA framework**

**Evidence**: Your model computes different drift rates for each trial based on rewards:
```julia
weights = 1.0 .+ (w_slope .* rewards)
drift_rates = C .* (weights ./ sum(weights))
```

### 2. **LBA Parameter Variation**

The LBA model itself can create bimodality through:
- **Start point variability** (`A` parameter): Different starting points → different RT distributions
- **Threshold variation**: Different thresholds across trials
- **Drift rate variation**: Different drift rates for different choices/rewards

### 3. **Choice-Specific RTs**

Different choices might have systematically different RTs:
- Choosing option 1 might be faster (first mode)
- Choosing option 2 might be slower (second mode)
- This is still within normal LBA decision-making

## Why the Model Isn't Finding Express Responses

The optimizer is correctly identifying that:
1. **The fast mode isn't fast enough** to be "express" (< 0.3s)
2. **The bimodality can be explained by LBA parameter variation** (different drift rates, start points, etc.)
3. **No separate express component is needed** - the LBA model alone can explain the bimodality

## What This Means

### The Model is Working Correctly

The fact that `p_exp ≈ 0` for most conditions suggests:
- The optimizer is correctly identifying that express responses aren't needed
- The bimodality is being explained by **variation within the LBA model itself**
- This is a more parsimonious explanation (fewer parameters)

### The Bimodality is Real, But Not "Express"

The observed bimodality is likely due to:
- **Different reward structures** creating different LBA distributions
- **Parameter variation** within the LBA framework
- **Choice-specific effects** or other systematic variation

### Implications for Your Analysis

1. **The mixture model approach may be unnecessary**: If bimodality is explained by LBA parameter variation, you don't need a separate express component.

2. **Focus on LBA parameters**: The bimodality might be better explained by:
   - Allowing drift rates to vary more across trials
   - Modeling start point variability
   - Accounting for choice-specific effects

3. **Model comparison**: Compare:
   - **Model 1**: LBA-only (current model with p_exp ≈ 0)
   - **Model 2**: LBA with express component (force p_exp > 0.05)
   - Use AIC/BIC to see which fits better

## Recommendations

1. **Examine the actual RT values**: Check what RT ranges the two modes are in. If both are > 0.3s, they're not express responses.

2. **Check reward structure variation**: Analyze whether condition 10 has multiple reward patterns that could create bimodality.

3. **Consider alternative models**: 
   - LBA with trial-to-trial parameter variation
   - LBA with choice-specific parameters
   - Hierarchical LBA models

4. **Visual inspection**: The fact that the second mode is larger suggests it's the "main" mode, and the first mode might be a subset of trials with specific characteristics (e.g., high-reward cues, easier choices).

## Conclusion

Your observation is correct: **the fast mode in the observed data is probably NOT express responses**. Instead, the bimodality is likely due to **systematic variation within the LBA model** (different reward structures, drift rates, or choice effects). The optimizer is correctly identifying this by setting `p_exp ≈ 0`.

