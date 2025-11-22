# ==========================================================================
# MIS-LBA Single Component Model Fitting Script
# ==========================================================================
#
# This script fits a single LBA model (NO mixture) to capture decision
# processes using the MIS (Multiple-Item Selection) theory.
#
# The model assumes a single decision process with:
# - Reward-based attentional weights determining drift rates
# - Standard LBA decision threshold and non-decision time
# - No mixture components (no bimodality modeling)
#
# ==========================================================================

using Pkg

# Load utility modules
include("data_utils.jl")
include("model_utils.jl")
include("fitting_utils.jl")
include("config.jl")
include("run_flags.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils
using .Config
using .RunFlags: get_plot_config, SAVE_INDIVIDUAL_CONDITION_PLOTS

# ==========================================================================
# CONFIGURATION
# ==========================================================================

# ========== CHANGE THIS TO SELECT PARTICIPANT ==========
const PARTICIPANT_ID = 3  # Options: 1, 2, or 3
# ========================================================

const OUTPUT_CSV = joinpath(@__DIR__, "outputdata", "model_fit_results_single_P$(PARTICIPANT_ID).csv")
const OUTPUT_PLOT = "model_fit_plot_single.png"

# ==========================================================================
# MAIN ANALYSIS FUNCTION
# ==========================================================================

function run_analysis()
    # Get data configuration for selected participant
    data_config = get_data_config(PARTICIPANT_ID)
    println("=" ^ 70)
    println("PARTICIPANT SELECTION")
    println("=" ^ 70)
    println("Selected Participant ID: $(data_config.participant_id)")
    println("Data path: $(data_config.data_base_path)")

    # Create configuration with plot display flags
    plot_config = get_plot_config()

    # Create images subfolder if it doesn't exist
    images_dir = joinpath(@__DIR__, "images")
    if !isdir(images_dir)
        mkdir(images_dir)
        println("Created images directory: $images_dir")
    end

    # Step 1: Load and process data
    println("\n" * "=" ^ 70)
    println("LOADING DATA")
    println("=" ^ 70)
    data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)

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

    # Step 2: Compute r_max from entire experiment (for consistent normalization)
    println("\n" * "=" ^ 70)
    println("COMPUTING EXPERIMENT-WIDE r_max")
    println("=" ^ 70)
    r_max = 0.0
    for rewards in data.ParsedRewards
        if !isempty(rewards)
            r_max = max(r_max, maximum(rewards))
        end
    end
    @assert r_max == 4 "rmax calculated incorrectly current number is: $(r_max)"

    println("r_max (maximum reward across entire experiment): $r_max")
    println("This value will be used consistently across all conditions for weight normalization.")

    # Step 3: Set up optimization parameters
    println("\n" * "=" ^ 70)
    println("FITTING SINGLE LBA MODEL FOR EACH CUE CONDITION")
    println("=" ^ 70)

    # Get parameter bounds and initial values from configuration
    params_config = get_default_single_params()
    lower = params_config.lower
    upper = params_config.upper
    x0 = params_config.x0

    # Store all results
    all_results = DataFrame[]
    # Store fitted parameters and data for overall accuracy plot
    condition_fits = Dict()
    # Store individual plot objects for combined plot
    individual_plots = []

    # Step 4: Fit model for each cue condition
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

        # Fit the single LBA model for this condition
        # Pass r_max to ensure consistent normalization across all conditions
        result = fit_model(condition_data, mis_lba_single_loglike;
                           lower=lower, upper=upper, x0=x0, time_limit=600.0, r_max=r_max)

        # Print all parameters (optimized and fixed)
        println("\n" * "=" ^ 70)
        println("FITTED PARAMETERS FOR CUE CONDITION: $cue_cond")
        println("=" ^ 70)
        
        best_params = Optim.minimizer(result)
        param_names = ["C", "w_slope", "A", "k", "t0"]
        
        println("\n--- OPTIMIZED PARAMETERS (in search) ---")
        for (i, name) in enumerate(param_names)
            println("  $name = $(round(best_params[i], digits=6))  [bounds: $(lower[i]) - $(upper[i])]")
        end
        
        println("\n--- FIXED PARAMETERS (out of search) ---")
        println("  r_max = $r_max  (experiment-wide maximum reward, used for MIS weight normalization)")
        
        println("\n--- PARAMETER DESCRIPTIONS ---")
        println("  C: Capacity parameter (drift rate scaling)")
        println("  w_slope: Reward weight slope (θ in MIS theory: exp(θ * r / r_max))")
        println("  A: Maximum start point variability in LBA")
        println("  k: Threshold gap in LBA (b - A, where b is decision threshold)")
        println("  t0: Non-decision time in LBA")
        println("  r_max: Maximum reward value across entire experiment (fixed, not optimized)")
        
        println("\n--- MIS THEORY PARAMETERS ---")
        println("  Weight calculation: w_i = exp(w_slope * r_i / r_max)")
        println("  Relative weights: rel_w_i = w_i / Σw_j")
        println("  Drift rates: ν_i = C * rel_w_i")
        
        println("\n--- LBA PARAMETERS ---")
        println("  LBA model: LBA(ν=drift_rates, A=A, k=k, τ=t0)")
        println("  where: ν = drift rates (vector), A = max start point, k = threshold gap, τ = non-decision time")
        println("=" ^ 70)

        # Save results for this condition
        results_df = save_results_single(result,
                                       "model_fit_results_single_P$(data_config.participant_id)_condition_$(cue_cond).csv";
                                       cue_condition=cue_cond)
        push!(all_results, results_df)

        # Store for overall accuracy plot
        best_params = Optim.minimizer(result)
        condition_fits[cue_cond] = (data=condition_data, params=best_params)

        # Generate plots for this condition
        plot_path = joinpath(images_dir, "model_fit_plot_single_P$(data_config.participant_id)_condition_$(cue_cond).png")
        p = generate_plot_single(condition_data, best_params,
                              plot_path;
                              cue_condition=cue_cond, r_max=r_max, config=plot_config,
                              save_plot=SAVE_INDIVIDUAL_CONDITION_PLOTS)
        push!(individual_plots, p)
    end

    # Step 4.5: Generate overall accuracy plot showing all conditions
    if !isempty(condition_fits)
        println("\n" * "=" ^ 70)
        println("GENERATING OVERALL ACCURACY PLOT")
        println("=" ^ 70)

        overall_accuracy_plot = joinpath(images_dir, "accuracy_plot_single_P$(data_config.participant_id)_all_conditions.png")
        generate_overall_accuracy_plot_single(condition_fits, overall_accuracy_plot; r_max=r_max)
    end

    # Step 4.6: Generate combined RT fit plot for all conditions
    if !isempty(individual_plots)
        println("\n" * "=" ^ 70)
        println("GENERATING COMBINED RT FIT PLOT FOR ALL CONDITIONS")
        println("=" ^ 70)

        # Calculate grid dimensions (aim for roughly square layout)
        n_plots = length(individual_plots)
        n_cols = ceil(Int, sqrt(n_plots))
        n_rows = ceil(Int, n_plots / n_cols)

        # Create combined plot with larger fonts
        combined_plot = plot(individual_plots...,
                            layout=(n_rows, n_cols),
                            size=(n_cols * 600, n_rows * 500),
                            plot_title="Single LBA Fit - Participant $(data_config.participant_id) - All Conditions",
                            plot_titlefontsize=18,
                            titlefontsize=14,
                            legendfontsize=12,
                            guidefontsize=14,
                            tickfontsize=12,
                            fontsize=12)

        # Save combined plot
        combined_plot_path = joinpath(images_dir, "model_fit_plot_single_P$(data_config.participant_id)_all_conditions.png")
        savefig(combined_plot, combined_plot_path)
        println("Saved combined RT fit plot to $combined_plot_path")
    end

    # Step 5: Combine and save all results
    println("\n" * "=" ^ 70)
    println("SAVING COMBINED RESULTS")
    println("=" ^ 70)

    if !isempty(all_results)
        combined_results = vcat(all_results...)

        # Create outputdata subfolder if it doesn't exist
        outputdata_dir = dirname(OUTPUT_CSV)
        if !isdir(outputdata_dir)
            mkdir(outputdata_dir)
            println("Created outputdata directory: $outputdata_dir")
        end

        # Save to outputdata subfolder with participant ID
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
    println("Participant: $(data_config.participant_id)")
    println("Combined results saved to $OUTPUT_CSV")
    if SAVE_INDIVIDUAL_CONDITION_PLOTS
        println("Individual condition results and plots saved with condition-specific filenames")
    else
        println("Individual condition plots skipped (SAVE_INDIVIDUAL_CONDITION_PLOTS=false)")
    end
    println("\nNote: This model uses a single LBA component (no mixture).")
end

# ==========================================================================
# RUN ANALYSIS
# ==========================================================================

using Optim  # Needed for Optim.minimizer
using CSV    # Needed for CSV.write
using DataFrames  # Needed for DataFrame operations
using Plots  # Needed for combining plots

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
