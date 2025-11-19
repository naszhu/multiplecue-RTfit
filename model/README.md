# MIS-LBA Mixture Model Fitting

This directory contains code for fitting a mixture model that combines:
- **MIS (Multiple-cue Intention Selection)** theory for decision-making
- **LBA (Linear Ballistic Accumulator)** for response time modeling
- **Express response component** for fast guesses

## Files

### Main Scripts

- **`fit-mis-model-modular.jl`** - Modular version (recommended)
  - Clean, organized code with separate modules
  - Easy to maintain and extend
  - Better for understanding the code structure

- **`fit-mis-model.jl`** - Original monolithic version
  - All code in one file
  - Useful if you need everything in one place
  - Contains all the same bug fixes

### Module Files

- **`data_utils.jl`** - Data reading and preprocessing
  - `parse_clean_float()` - Safely parse numeric values
  - `parse_array_string()` - Parse cue value arrays
  - `read_psychopy_dat()` - Read PsychoPy data files
  - `load_and_process_data()` - Complete data loading pipeline

- **`model_utils.jl`** - Model likelihood calculations
  - `mis_lba_mixture_loglike()` - Compute negative log-likelihood for the MIS-LBA mixture model

- **`fitting_utils.jl`** - Optimization and visualization
  - `fit_model()` - Run parameter optimization
  - `save_results()` - Save fitted parameters to CSV
  - `generate_plot()` - Create visualization of model fit

### Test/Helper Files

- **`test_data_reading.jl`** - Test data reading functions
- **`test_simple.jl`** - Simple tests for header detection
- **`test_fix.jl`** - Verify bug fixes

## Usage

### Running the Analysis

```bash
julia fit-mis-model-modular.jl
```

Or use the original version:

```bash
julia fit-mis-model.jl
```

### Configuration

Edit the constants at the top of the main script:

```julia
const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"
const OUTPUT_CSV = "model_fit_results.csv"
const OUTPUT_PLOT = "model_fit_plot.png"
```

### Parameter Bounds

The model has 8 parameters:

1. **C** - Capacity (drift rate scaling)
2. **w_slope** - Reward weight slope
3. **A** - Maximum start point variability
4. **k** - Threshold gap (b - A)
5. **t0** - Non-decision time
6. **p_exp** - Probability of express responses
7. **mu_exp** - Mean of express response distribution
8. **sig_exp** - Standard deviation of express response distribution

Default bounds and initial values are set in the main script.

## Output Files

- **`model_fit_results.csv`** - Fitted parameter values
- **`model_fit_plot.png`** - Visualization comparing observed vs predicted RT distributions

## Data Format

The code expects PsychoPy `.dat` files with:
- Tab-delimited format
- Header line containing `ExperimentName` and `RT` columns
- Required columns: `RT`, `CueValues`, `PointTargetResponse` (or `RespLoc`), `CueResponseValue`
- Practice files identified by `-Prac-` in filename (automatically excluded)

## Bug Fixes Applied

This version includes fixes for:

1. **Variable scoping bug** - Header line detection now works correctly
2. **Practice file filtering** - Uses filename pattern instead of non-existent column
3. **CueValues parsing** - Correctly handles digit strings like "0410" → [0,4,1,0]
4. **Response column** - Uses `PointTargetResponse` instead of empty `RespLoc`

**Result**: Valid trials increased from 592 to 3,469 (586% increase)

## Requirements

```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Glob", "Distributions",
         "SequentialSamplingModels", "Optim", "Statistics",
         "Random", "Plots"])
```

## Module Structure

```
fit-mis-model-modular.jl (main)
├── data_utils.jl
│   └── Load and preprocess data
├── model_utils.jl
│   └── Compute model likelihood
└── fitting_utils.jl
    ├── Optimize parameters
    ├── Save results
    └── Generate plots
```

## Extending the Code

### Adding a New Data Processing Step

Edit `data_utils.jl` and add your processing in `load_and_process_data()`:

```julia
# Add after line 127 in data_utils.jl
full_df.YourNewColumn = your_processing_function.(full_df.SomeColumn)
```

### Modifying the Model

Edit `model_utils.jl` and modify `mis_lba_mixture_loglike()`:

```julia
# Change the drift rate calculation or add new parameters
```

### Changing Optimization Settings

Edit `fitting_utils.jl` and modify `fit_model()`:

```julia
# Change optimizer, options, or constraints
```

## Troubleshooting

### Issue: "No data files found"
- Check that `DATA_PATH` points to the correct directory
- Verify `.dat` files exist in that location

### Issue: "All trials were filtered out"
- Check RT filtering bounds (default: 0.05 < RT < 3.0)
- Verify column names match your data

### Issue: Optimization fails or gives Inf
- Check parameter bounds are reasonable
- Ensure data quality (no NaN or missing values in key columns)
- Try different initial values

## Contact

For questions about the code structure or bugs, refer to the git commit history or contact the maintainer.
