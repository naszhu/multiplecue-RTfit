# Investigation Summary: Trial Exclusion Issue

## Problem

The data loading process was excluding **1780 out of 5250 trials** (33.9%), leaving only **3469 valid trials**. This seemed excessive and warranted investigation.

## Investigation Results

### Exclusion Breakdown

1. **RT Filtering**: Only **1 trial** removed (RT <= 0.05 seconds)
   - This is expected and appropriate

2. **Choice Filtering**: **1780 trials** removed
   - **1770 trials**: `Choice > length(ParsedRewards)` (choice index out of bounds)
   - **10 trials**: `Choice <= 0` (no valid choice found)

### Root Cause

The issue was in how `PointTargetResponse` is encoded:

- **PointTargetResponse** uses a **fixed position encoding** (values 1-4) that always refers to positions, regardless of how many options are actually present in a trial
- When there are fewer than 4 options, `PointTargetResponse` can still be 3 or 4, leading to invalid choices
- Example: `PointTargetResponse=4` with only 1 option → invalid (position 4 doesn't exist)

### Key Finding

Analysis revealed that **99.3% of invalid trials** would be valid if we used `CueResponseValue` to infer the choice instead:

- `CueResponseValue` contains the **actual reward value** that was selected
- We can find its index in `ParsedRewards` to get the correct choice
- This approach works correctly regardless of how many options are present

## Solution

Modified the choice determination logic in `data_utils.jl`:

1. **Strategy A**: Try `PointTargetResponse` first, but **validate it's within bounds**
   - Only use it if `1 <= PointTargetResponse <= length(ParsedRewards)`

2. **Strategy B**: If `PointTargetResponse` is invalid, fall back to `CueResponseValue`
   - Find the index of `CueResponseValue` in `ParsedRewards`
   - This correctly handles cases with fewer than 4 options

3. **Strategy C**: If both fail, try `RespLoc` (with validation)

## Results After Fix

- **Before**: 3469 valid trials (1780 excluded)
- **After**: 5237 valid trials (only 12 excluded)
- **Recovered**: 1768 additional trials (51% increase)

### Final Exclusion Breakdown

- **RT filtering**: 1 trial (RT <= 0.05s)
- **Choice filtering**: 12 trials (likely edge cases where all strategies fail)
- **Total excluded**: 13 / 5250 (0.25%)

## Files Modified

- `model/data_utils.jl`: Updated choice determination logic (lines 128-172)

## Diagnostic Scripts Created

- `diagnose_excluded_trials.jl`: Comprehensive diagnostic of excluded trials
- `investigate_choice_encoding.jl`: Analysis of choice encoding methods

These scripts can be run to investigate data quality issues in the future.


