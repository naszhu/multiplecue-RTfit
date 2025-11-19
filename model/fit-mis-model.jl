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

    # Check if CueCondition column exists
    if !("CueCondition" in names(data))
        error("CueCondition column not found in data. Please ensure data files contain this column.")
    end

    # Get unique cue conditions
    cue_conditions = unique(data.CueCondition)
    filter!(x -> !ismissing(x), cue_conditions)
    sort!(cue_conditions)
    
    println("\nFound $(length(cue_conditions)) unique cue conditions:")
    for (i, cc) in enumerate(cue_conditions)
        n_trials = sum(data.CueCondition .== cc)
        println("  $i. CueCondition $cc: $n_trials trials")
    end

    # Step 2: Set up optimization parameters
    println("\n" * "=" ^ 70)
    println("FITTING MODEL FOR EACH CUE CONDITION")
    println("=" ^ 70)

    # Parameter bounds and initial values
    # [C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp]
    lower = [1.0,  0.0,   0.01, 0.05, 0.05, 0.0,  0.05,  0.001]
    upper = [30.0, 10.0,  1.0,  1.0,  0.6,  0.8,  0.20,  0.1]
    x0    = [10.0, 1.0,   0.3,  0.3,  0.2,  0.2,  0.10,  0.02]

    # Store all results
    all_results = DataFrame[]
    
    # Step 3: Fit model for each cue condition
    for (idx, cue_cond) in enumerate(cue_conditions)
        println("\n" * "-" ^ 70)
        println("FITTING CUE CONDITION: $cue_cond ($idx/$(length(cue_conditions)))")
        println("-" ^ 70)
        
        # Filter data for this cue condition
        condition_data = filter(row -> row.CueCondition == cue_cond, data)
        n_trials = nrow(condition_data)
        println("Number of trials: $n_trials")
        
        if n_trials < 10
            println("WARNING: Too few trials ($n_trials) for cue condition $cue_cond. Skipping...")
            continue
        end

        # Fit the model for this condition
        result = fit_model(condition_data, mis_lba_mixture_loglike;
                           lower=lower, upper=upper, x0=x0, time_limit=600.0)

        # Save results for this condition
        results_df = save_results(result, 
                                   "model_fit_results_condition_$(cue_cond).csv";
                                   cue_condition=cue_cond)
        push!(all_results, results_df)
        
        # Generate plot for this condition
        best_params = Optim.minimizer(result)
        generate_plot(condition_data, best_params, 
                      "model_fit_plot_condition_$(cue_cond).png";
                      cue_condition=cue_cond)
    end

    # Step 4: Combine and save all results
    println("\n" * "=" ^ 70)
    println("SAVING COMBINED RESULTS")
    println("=" ^ 70)

    if !isempty(all_results)
        combined_results = vcat(all_results...)
        CSV.write(OUTPUT_CSV, combined_results)
        println("\nCombined fitted parameters:")
        println(combined_results)
        println("\nResults saved to: $OUTPUT_CSV")
    else
        println("WARNING: No results to save!")
    end

    println("\n" * "=" ^ 70)
    println("ANALYSIS COMPLETE")
    println("=" ^ 70)
    println("Combined results saved to: $OUTPUT_CSV")
    println("Individual condition results and plots saved with condition-specific filenames")
end

# ==========================================================================
# RUN ANALYSIS
# ==========================================================================

using Optim  # Needed for Optim.minimizer
using CSV    # Needed for CSV.write
using DataFrames  # Needed for DataFrame operations

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
