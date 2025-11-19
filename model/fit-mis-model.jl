# ==========================================================================
# MIS-LBA Mixture Model Fitting Script (Modular Version)
# ==========================================================================
#
# This script fits a mixture model combining:
# - MIS (Multiple-cue Intention Selection) theory for decision-making
# - LBA (Linear Ballistic Accumulator) for response time modeling
# - Express response component for fast guesses
#
# The code is organized into modules:
# - data_utils.jl: Data reading and preprocessing
# - model_utils.jl: Model likelihood calculations
# - fitting_utils.jl: Optimization and visualization
#
# ==========================================================================

using Pkg

# Ensure required packages are installed
# Uncomment if needed:
# Pkg.add(["CSV", "DataFrames", "Glob", "Distributions", "SequentialSamplingModels", "Optim", "Statistics", "Random", "Plots"])

# Load utility modules
include("data_utils.jl")
include("model_utils.jl")
include("fitting_utils.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils

# ==========================================================================
# CONFIGURATION
# ==========================================================================

const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"
const OUTPUT_CSV = "model_fit_results.csv"
const OUTPUT_PLOT = "model_fit_plot.png"

# ==========================================================================
# MAIN ANALYSIS FUNCTION
# ==========================================================================

function run_analysis()
    # Step 1: Load and process data
    println("=" ^ 70)
    println("LOADING DATA")
    println("=" ^ 70)
    data = load_and_process_data(DATA_PATH, FILE_PATTERN)

    # Step 2: Set up optimization
    println("\n" * "=" ^ 70)
    println("FITTING MODEL")
    println("=" ^ 70)

    # Parameter bounds and initial values
    # [C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp]
    lower = [1.0,  0.0,   0.01, 0.05, 0.05, 0.0,  0.05,  0.001]
    upper = [30.0, 10.0,  1.0,  1.0,  0.6,  0.8,  0.20,  0.1]
    x0    = [10.0, 1.0,   0.3,  0.3,  0.2,  0.2,  0.10,  0.02]

    # Fit the model
    result = fit_model(data, mis_lba_mixture_loglike;
                       lower=lower, upper=upper, x0=x0, time_limit=600.0)

    # Step 3: Save results
    println("\n" * "=" ^ 70)
    println("SAVING RESULTS")
    println("=" ^ 70)

    results_df = save_results(result, OUTPUT_CSV)
    println("\nFitted Parameters:")
    println(results_df)

    # Step 4: Generate plot
    println("\n" * "=" ^ 70)
    println("GENERATING VISUALIZATION")
    println("=" ^ 70)

    best_params = Optim.minimizer(result)
    generate_plot(data, best_params, OUTPUT_PLOT)

    println("\n" * "=" ^ 70)
    println("ANALYSIS COMPLETE")
    println("=" ^ 70)
    println("Results saved to: $OUTPUT_CSV")
    println("Plot saved to: $OUTPUT_PLOT")
end

# ==========================================================================
# RUN ANALYSIS
# ==========================================================================

using Optim  # Needed for Optim.minimizer

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
