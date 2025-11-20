# ==========================================================================
# MIS-LBA Dual Mixture Model Fitting Script
# ==========================================================================
#
# This script fits a dual-LBA mixture model that uses TWO LBA components
# with different parameters to capture bimodality, rather than LBA + express.
#
# The model assumes bimodality comes from two different decision processes:
# - LBA Component 1: Fast mode (lower thresholds, faster non-decision time)
# - LBA Component 2: Slow mode (higher thresholds, slower non-decision time)
#
# This is more appropriate when bimodality is within the normal LBA range
# rather than from express responses (< 0.3s).
#
# ==========================================================================

using Pkg

# Load utility modules
include("data_utils.jl")
include("model_utils.jl")
include("fitting_utils.jl")
include("config.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils
using .Config

# ==========================================================================
# CONFIGURATION
# ==========================================================================

const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"
const OUTPUT_CSV = "model_fit_results_dual.csv"
const OUTPUT_PLOT = "model_fit_plot_dual.png"

# ==========================================================================
# MAIN ANALYSIS FUNCTION
# ==========================================================================

function run_analysis()
    # Create configuration with plot display flags
    # Set to false to disable target/distractor choice lines in plots
    plot_config = ModelConfig(false, false)  # show_target_choice, show_distractor_choice

    # Create images subfolder if it doesn't exist
    images_dir = joinpath(@__DIR__, "images")
    if !isdir(images_dir)
        mkdir(images_dir)
        println("Created images directory: $images_dir")
    end

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
    if r_max <= 0.0
        r_max = 1.0
    end
    println("r_max (maximum reward across entire experiment): $r_max")
    println("This value will be used consistently across all conditions for weight normalization.")

    # Step 3: Set up optimization parameters
    println("\n" * "=" ^ 70)
    println("FITTING DUAL-LBA MODEL FOR EACH CUE CONDITION")
    println("=" ^ 70)

    # Parameter bounds and initial values for dual-LBA model
    # [C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix]
    # Component 1 (fast): lower thresholds, faster t0
    # Component 2 (slow): higher thresholds, slower t0
    lower = [1.0,  0.0,   0.01, 0.05, 0.05,  0.01, 0.05, 0.15,  0.0]
    upper = [30.0, 10.0,  1.0,  1.0,  0.4,   1.0,  1.0,  0.6,   0.99]
    x0    = [10.0, 1.0,   0.2,  0.2,  0.2,   0.3,  0.3,  0.35,  0.4]

    # Store all results
    all_results = DataFrame[]
    # Store fitted parameters and data for overall accuracy plot
    condition_fits = Dict()
    
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

        # Fit the dual-LBA model for this condition
        # Pass r_max to ensure consistent normalization across all conditions
        result = fit_model(condition_data, mis_lba_dual_mixture_loglike;
                           lower=lower, upper=upper, x0=x0, time_limit=600.0, r_max=r_max)

        # Save results for this condition
        results_df = save_results_dual(result, 
                                       "model_fit_results_dual_condition_$(cue_cond).csv";
                                       cue_condition=cue_cond)
        push!(all_results, results_df)
        
        # Store for overall accuracy plot
        best_params = Optim.minimizer(result)
        condition_fits[cue_cond] = (data=condition_data, params=best_params)
        
        # Generate plots for this condition
        plot_path = joinpath(images_dir, "model_fit_plot_dual_condition_$(cue_cond).png")
        generate_plot_dual(condition_data, best_params,
                          plot_path;
                          cue_condition=cue_cond, r_max=r_max, config=plot_config)
    end
    
    # Step 4.5: Generate overall accuracy plot showing all conditions
    if !isempty(condition_fits)
        println("\n" * "=" ^ 70)
        println("GENERATING OVERALL ACCURACY PLOT")
        println("=" ^ 70)
        
        overall_accuracy_plot = joinpath(images_dir, "accuracy_plot_dual_all_conditions.png")
        generate_overall_accuracy_plot(condition_fits, overall_accuracy_plot; r_max=r_max)
    end

    # Step 5: Combine and save all results
    println("\n" * "=" ^ 70)
    println("SAVING COMBINED RESULTS")
    println("=" ^ 70)

    if !isempty(all_results)
        combined_results = vcat(all_results...)

        # Create outputdata subfolder if it doesn't exist
        outputdata_dir = joinpath(@__DIR__, "outputdata")
        if !isdir(outputdata_dir)
            mkdir(outputdata_dir)
            println("Created outputdata directory: $outputdata_dir")
        end

        # Save to outputdata subfolder
        output_path = joinpath(outputdata_dir, OUTPUT_CSV)
        CSV.write(output_path, combined_results)
        println("\nCombined fitted parameters:")
        println(combined_results)
        println("\nResults saved to: $output_path")
    else
        println("WARNING: No results to save!")
    end

    println("\n" * "=" ^ 70)
    println("ANALYSIS COMPLETE")
    println("=" ^ 70)
    println("Combined results saved to outputdata/$OUTPUT_CSV")
    println("Individual condition results and plots saved with condition-specific filenames")
    println("\nNote: This model uses TWO LBA components to capture bimodality,")
    println("      rather than LBA + express responses.")
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

