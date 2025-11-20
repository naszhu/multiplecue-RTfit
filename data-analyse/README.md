# Data Analysis: Hypothesis Testing

This directory contains R scripts for testing different hypotheses about the bimodal reaction time (RT) distributions observed in the multiple-cue selection task.

## Overview

The analyses investigate why RT distributions show two peaks (fast and slow) by testing various hypotheses about the underlying cognitive and motor processes.

## Hypothesis Files

### Hypothesis A: Pop-out vs. Search
**File:** `hypothesis_A_popout_vs_search.r`

**Question:** Does the fast peak correspond to single high-value cue trials (pop-out), while the slow peak corresponds to competing high-value cues (search)?

**Method:** 
- Classifies trials based on the reward gap between the maximum and second-maximum cue values
- Large gap = "Pop-out" (single dominant cue)
- Small gap = "Search" (competing cues)
- Compares RT distributions between these trial types

**Key Outputs:**
- RT distributions by trial type (median split and strict classification)
- RT vs. reward gap scatter plots
- Summary statistics by competition level

---

### Hypothesis B: Pre-planning vs. Reactive Saccades
**File:** `hypothesis_B_preplanning_vs_reactive.r`

**Question:** Does the fast peak correspond to pre-planned/guessed movements, while the slow peak corresponds to value-based decisions?

**Method:**
- Classifies trials by RT speed and choice optimality
- Very fast RTs (< 0.25s) that are non-optimal might be guesses
- Analyzes the relationship between RT and choice optimality
- Tests if pre-planned responses show different RT patterns

**Key Outputs:**
- RT distributions by choice optimality
- Optimality rate vs. RT curves
- Condition-specific analyses (conditions 3, 4, 5, 10)

---

### Hypothesis C: Eye-Tracking Artifacts (Re-fixations)
**File:** `hypothesis_C_eyetracking_artifacts.r`

**Question:** Does the slow peak come from trials where the eye started to move, missed the target, and then corrected to land on the correct target?

**Method:**
- Analyzes eye-tracking data for evidence of re-fixations
- Uses number of eye samples and eye sample density as indicators
- High sample density with long RT might indicate corrections
- Classifies trials by RT and eye activity level

**Key Outputs:**
- RT distributions by eye activity type
- RT vs. number of eye samples
- Eye sample density vs. RT plots

---

### Hypothesis C2: Redefined RT (Circle Entry)
**File:** `hypothesis_C2_eyetracking_redefined_RT.r`

**Question:** Does bimodality persist when RT is redefined based on when the eye enters a larger circle around each cue location?

**Method:**
- Redefines RT as the time from cue onset to when the eye first enters a larger circle around the target location
- This measures when the eye reaches the target area, not when the response is made
- Tests if the original RT measurement was affected by eye-tracking artifacts

**Key Outputs:**
- Comparison of original vs. redefined RT distributions
- Redefined RT distributions by cue condition
- Summary statistics for redefined RT

---

### Hypothesis C3: Saccade Initiation RT
**File:** `hypothesis_C3_eyetracking_saccade_initiation.r`

**Question:** Does bimodality persist when RT is redefined as the time from cue onset to saccade initiation (when the eye leaves the initial fixation)?

**Method:**
- Redefines RT as the time from cue onset to when the eye first moves away from the initial fixation point
- This measures when the saccade is initiated, not when it reaches the target
- Uses a fixation circle around the initial gaze position to detect when the eye leaves fixation

**Key Outputs:**
- Comparison of original vs. saccade initiation RT
- Saccade initiation RT distributions by cue condition
- Summary statistics for saccade initiation RT

**Note:** This script includes an adjustable parameter `FIXATION_RADIUS_FACTOR` (line 184) that controls the sensitivity of saccade detection (0.3-1.0, default 1.0).

---

### Hypothesis D: Dual-Cue Detailed Analysis
**File:** `hypothesis_D_dual_cue_detailed.r`

**Question:** How do RT distributions differ for dual-cue conditions (5-10) when classified by choice type (Optimal, Second Best, or Mismatch)?

**Method:**
- Focuses on dual-cue conditions (5-10) only
- Classifies trials by:
  - **Optimal:** Chose the location with maximum reward
  - **Second Best:** Chose the location with second-highest reward
  - **Mismatch:** Chose a location with no cue (0 reward)
- Analyzes RT distributions for each choice type across speed categories (Fast, Medium, Slow)

**Key Outputs:**
- Overall RT density by trial type (weighted densities)
- Condition-specific plots for conditions 5-10
- Summary statistics by trial type

---

### Hypothesis E: Single-Cue Correctness
**File:** `hypothesis_E_single_cue_correctness.r`

**Question:** How do RT distributions differ for single-cue conditions (1-4) when classified by correctness?

**Method:**
- Focuses on single-cue conditions (1-4) only
- Classifies trials by:
  - **Correct:** Chose the cued location
  - **Incorrect:** Chose wrong/empty location
- Analyzes RT distributions for correct vs. incorrect responses across speed categories

**Key Outputs:**
- Overall RT density by trial type (weighted densities)
- Condition-specific plots for conditions 1-4
- Summary statistics by trial type

---

## Other Analysis Files

### `original_RT_by_condition.r`
Initial analysis of RT distributions by cue condition, showing the original bimodal patterns.

### `skewed_distribution_groups.r`
Analysis of skewed distribution groups in the data.

### `initial-analyze.r`
Initial exploratory data analysis.

---

## Output Directory

All scripts save their plots and summary statistics to the `figures/` subdirectory.

## Data Source

All scripts read data from:
```
../data/ParticipantCPP002-003/ParticipantCPP002-003/
```

The scripts expect `.dat` files with eye-tracking and behavioral data.

---

## Running the Analyses

Each hypothesis script can be run independently. Make sure you have the required R packages installed:
- `tidyverse`
- `data.table`
- `gridExtra` (for some scripts)

Example:
```r
source("hypothesis_A_popout_vs_search.r")
```

---

## Notes

- All RT values are in seconds
- RT filtering: 0 < RT <= 10 seconds
- Fast RT threshold: < 0.25s
- Medium RT threshold: 0.25-0.4s
- Slow RT threshold: > 0.4s

