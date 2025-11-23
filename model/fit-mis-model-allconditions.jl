# ==========================================================================
# MIS-LBA All-Conditions Model Fitting Script
# ==========================================================================
#
# This script fits a single LBA model with SHARED parameters across ALL cue
# conditions. Unlike the condition-specific models, this uses:
# - Single C (capacity) parameter for all conditions
# - Single θ (theta/w_slope) parameter for all conditions
# - Single set of LBA parameters (A, k, t0) for all conditions
#
# The model still generates separate prediction outputs for each cue condition
# to allow visualization of how well the shared parameters fit each condition.
#
# ==========================================================================

using Pkg

# Load utility modules
include("data_utils.jl")
include("model_utils.jl")
include("config.jl")
include("run_flags.jl")
include("fitting_utils.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils
using .Config
using .RunFlags: get_plot_config, SAVE_INDIVIDUAL_CONDITION_PLOTS

# ==========================================================================
# CONFIGURATION
# ==========================================================================

# ==========================================================================
# MAIN ANALYSIS FUNCTION
# ==========================================================================

function run_analysis()
    # Get data configuration for selected participant
    data_config = get_data_config(Config.PARTICIPANT_ID_ALLCONDITIONS)
    println("=" ^ 70)
    println("PARTICIPANT SELECTION")
    println("=" ^ 70)
    println("Selected Participant ID: $(data_config.participant_id)")
    println("Data path: $(data_config.data_base_path)")
    weighting_mode = isnothing(Config.WEIGHTING_MODE_OVERRIDE_ALLCONDITIONS) ? get_weighting_mode() : Config.WEIGHTING_MODE_OVERRIDE_ALLCONDITIONS
    vary_C_by_cue_type = Config.VARY_C_BY_CUECOUNT_ALLCONDITIONS
    vary_t0_by_cue_type = Config.VARY_T0_BY_CUECOUNT_ALLCONDITIONS
    vary_k_by_cue_type = Config.VARY_K_BY_CUECOUNT_ALLCONDITIONS
    use_contam = Config.USE_CONTAMINANT_FLOOR_ALLCONDITIONS
    estimate_contam = Config.ESTIMATE_CONTAMINANT_ALLCONDITIONS
    println("Reward weighting mode: $weighting_mode")
    println("Vary C by cue-count (single vs double): $vary_C_by_cue_type")
    println("Vary t0 by cue-count (single vs double): $vary_t0_by_cue_type")
    println("Vary k by cue-count (single vs double): $vary_k_by_cue_type")
    println("Vary k by cue-count (single vs double): $vary_k_by_cue_type")

    # Create configuration with plot display flags
    plot_config = get_plot_config()  # from RunFlags

    # Create images subfolder if it doesn't exist
    images_dir = Config.IMAGES_DIR
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
    known_conditions = union(Config.SINGLE_CUE_CONDITIONS, Config.DOUBLE_CUE_CONDITIONS)
    @assert all(cc -> cc in known_conditions, cue_conditions) "Unexpected CueCondition values detected: $(setdiff(cue_conditions, collect(known_conditions)))"
    if any(ismissing, data.CueCondition)
        error("CueCondition column contains missing values; required for cue-type specific parameters.")
    end
    cue_condition_types = Config.cue_condition_type.(data.CueCondition)
    @assert all(ct -> ct in (:single, :double), cue_condition_types) "CueCondition types must be :single or :double"
    println("\nTotal trials across all conditions: $(nrow(data))")

    # Step 2: Compute r_max from entire experiment (for consistent normalization)
    println("\n" * "=" ^ 70)
    println("COMPUTING EXPERIMENT-WIDE r_max")
    println("=" ^ 70)

    # Find maximum reward across all trials in the entire experiment
    r_max = 0.0
    for rewards in data.ParsedRewards
        if !isempty(rewards)
            r_max = max(r_max, maximum(rewards))
        end
    end

    # Check assertion after processing all rewards
    if r_max <= 0.0
        r_max = 1.0
    end

    println("r_max (maximum reward across entire experiment): $r_max")
    @assert r_max == 4 "rmax calculated incorrectly: expected 4, got $r_max"
    println("This value will be used consistently across all conditions for weight normalization.")

    # Step 3: Preprocess data for ultra-fast fitting
    println("\n" * "=" ^ 70)
    println("PREPROCESSING DATA FOR OPTIMIZATION")
    println("=" ^ 70)
    group_by_condition = vary_C_by_cue_type || vary_t0_by_cue_type || vary_k_by_cue_type || Config.VARY_CONTAM_BY_CUE_ALLCONDITIONS
    preprocessed_data = preprocess_data_for_fitting(data; cue_condition_types=cue_condition_types, group_by_condition=group_by_condition)

    # Step 4: Fit single model to ALL data at once with shared parameters
    params_config, layout, param_names = build_allconditions_params(weighting_mode;
        vary_C_by_cue=vary_C_by_cue_type,
        vary_t0_by_cue=vary_t0_by_cue_type,
        vary_k_by_cue=vary_k_by_cue_type,
        use_contaminant=use_contam,
        estimate_contaminant=estimate_contam,
        vary_contam_by_cue=Config.VARY_CONTAM_BY_CUE_ALLCONDITIONS,
        c_start_override=Config.C_START_OVERRIDE_ALLCONDITIONS)
    lower = params_config.lower
    upper = params_config.upper
    x0 = params_config.x0

    flag_tokens = String[]
    push!(flag_tokens, weighting_mode == :free ? "wfree" : "wslope")
    if vary_C_by_cue_type
        push!(flag_tokens, "Ccue")
    end
    if vary_t0_by_cue_type
        push!(flag_tokens, "t0cue")
    end
    if vary_k_by_cue_type
        push!(flag_tokens, "kcue")
    end
    if use_contam
        if estimate_contam
            push!(flag_tokens, layout.vary_contam_by_cue ? "contamEstCue" : "contamEst")
        else
            push!(flag_tokens, "contam$(Int(round(Config.CONTAMINANT_ALPHA_ALLCONDITIONS*100)))")
        end
    end
    flag_suffix = isempty(flag_tokens) ? "" : "_" * join(flag_tokens, "-")

    println("\n" * "=" ^ 70)
    println("FITTING SINGLE LBA MODEL TO ALL CONDITIONS")
    println("=" ^ 70)
    println("Model type: Single LBA with SHARED parameters across all conditions (C/t0 can vary by cue-count)")
    println("Parameters to be fitted: $(join(param_names, ", "))")
    println("Number of conditions: $(length(cue_conditions))")
    println("Total trials: $(nrow(data))")
    println("Using ULTRA-FAST preprocessed data structure")

    println("\n" * "-" ^ 70)
    println("OPTIMIZATION SETUP")
    println("-" ^ 70)
    println("Parameter bounds:")
    for (i, name) in enumerate(param_names)
        println("  $name: [$(lower[i]), $(upper[i])], initial: $(x0[i])")
    end

    # Fit the model using preprocessed data (3-5x faster than DataFrame version)
    println("\n" * "-" ^ 70)
    println("RUNNING OPTIMIZATION")
    println("-" ^ 70)
    objective_func = (x, d) -> mis_lba_allconditions_loglike(x, d; layout=layout, r_max=r_max, weighting_mode=weighting_mode, vary_C_by_cue_type=vary_C_by_cue_type, vary_t0_by_cue_type=vary_t0_by_cue_type, vary_k_by_cue_type=vary_k_by_cue_type, use_contaminant=Config.USE_CONTAMINANT_FLOOR_ALLCONDITIONS, estimate_contaminant=Config.ESTIMATE_CONTAMINANT_ALLCONDITIONS, contaminant_alpha=Config.CONTAMINANT_ALPHA_ALLCONDITIONS, contaminant_rt_max=Config.CONTAMINANT_RT_MAX_ALLCONDITIONS)
    result = fit_model(preprocessed_data, objective_func;
                       lower=lower, upper=upper, x0=x0, time_limit=600.0)

    # Get the fitted parameters
    best_params = Optim.minimizer(result)
    contam_alpha_use = Config.CONTAMINANT_ALPHA_ALLCONDITIONS
    contam_rtmax_use = Config.CONTAMINANT_RT_MAX_ALLCONDITIONS
    if layout.use_contaminant && layout.estimate_contaminant
        if !isempty(layout.idx_contam_alpha)
            key = haskey(layout.idx_contam_alpha, :all) ? :all : :single
            contam_alpha_use = best_params[layout.idx_contam_alpha[key]]
        end
        if !isempty(layout.idx_contam_rt)
            key = haskey(layout.idx_contam_rt, :all) ? :all : :single
            contam_rtmax_use = best_params[layout.idx_contam_rt[key]]
        end
    end

    # Print all parameters (optimized and fixed)
    println("\n" * "=" ^ 70)
    println("FITTED PARAMETERS (SHARED / cue-type-specific as configured)")
    println("=" ^ 70)

    println("\n--- OPTIMIZED PARAMETERS (in search) ---")
    for (i, name) in enumerate(param_names)
        println("  $name = $(round(best_params[i], digits=6))  [bounds: $(lower[i]) - $(upper[i])]")
    end

    println("\n--- FIXED PARAMETERS (out of search) ---")
    if weighting_mode == :exponential
        println("  r_max = $r_max  (experiment-wide maximum reward, used for MIS weight normalization)")
    else
        println("  weighting_mode = :free  (w1 fixed at 1.0; r_max not used for weights)")
    end

    println("\n--- PARAMETER DESCRIPTIONS ---")
    println("  C: Capacity parameter (drift rate scaling)")
    if weighting_mode == :exponential
        println("  w_slope: Reward weight slope (θ in MIS theory: exp(θ * r / r_max))")
    else
        println("  w2/w3/w4: Free reward weights (w1 fixed at 1.0 baseline)")
    end
    println("  A: Maximum start point variability in LBA")
    println("  k: Threshold gap in LBA (b - A, where b is decision threshold)")
    println("  t0: Non-decision time in LBA")
    if weighting_mode == :exponential
        println("  r_max: Maximum reward value across entire experiment (fixed, not optimized)")
    end

    println("\n--- MIS THEORY PARAMETERS ---")
    if weighting_mode == :exponential
        println("  Weight calculation: w_i = exp(w_slope * r_i / r_max)")
    else
        println("  Weight calculation: w_i pulled from fitted [1.0, w2, w3, w4] lookup")
    end
    println("  Relative weights: rel_w_i = w_i / Σw_j")
    println("  Drift rates: ν_i = C * rel_w_i")

    println("\n--- LBA PARAMETERS ---")
    println("  LBA model: LBA(ν=drift_rates, A=A, k=k, τ=t0)")
    println("  where: ν = drift rates (vector), A = max start point, k = threshold gap, τ = non-decision time")

    println("\n--- OPTIMIZATION INFO ---")
    println("  Negative log-likelihood: $(round(Optim.minimum(result), digits=2))")
    println("  Converged: $(Optim.converged(result))")
    println("=" ^ 70)

    # Step 4: Save results
    println("\n" * "=" ^ 70)
    println("SAVING RESULTS")
    println("=" ^ 70)

    # Create outputdata subfolder if it doesn't exist
    outputdata_dir = joinpath(@__DIR__, "outputdata")
    if !isdir(outputdata_dir)
        mkdir(outputdata_dir)
        println("Created outputdata directory: $outputdata_dir")
    end

    # Save overall results
    output_filename = "model_fit_results_allconditions_P$(data_config.participant_id).csv"
    results_df = save_results_allconditions(result, output_filename; param_names=param_names)
    println("\nFitted parameters:")
    println(results_df)

    # Step 5: Generate plots for each condition using the SHARED parameters
    println("\n" * "=" ^ 70)
    println("GENERATING CONDITION-SPECIFIC PLOTS (USING SHARED PARAMETERS)")
    println("=" ^ 70)

    # Store individual plot objects and condition data
    individual_plots = []
    condition_data_dict = Dict{Any,DataFrame}()

    for (idx, cue_cond) in enumerate(cue_conditions)
        println("\n" * "-" ^ 70)
        println("GENERATING PLOT FOR CUE CONDITION: $cue_cond ($idx/$(length(cue_conditions)))")
        println("-" ^ 70)

        # Filter data for this cue condition
        condition_data = filter(row -> row.CueCondition == cue_cond, data)
        n_trials = nrow(condition_data)
        println("Number of trials: $n_trials")

        if n_trials < 10
            println("WARNING: Too few trials ($n_trials) for cue condition $cue_cond. Skipping plot...")
            continue
        end

        # Store condition data for accuracy plot
        condition_data_dict[cue_cond] = condition_data

        # Generate plot for this condition using the SHARED parameters
        plot_path = joinpath(images_dir, "model_fit_plot_allconditions_P$(data_config.participant_id)_condition_$(cue_cond)$(flag_suffix).png")
        p = generate_plot_allconditions(condition_data, best_params,
                                       plot_path;
                                       cue_condition=cue_cond, r_max=r_max, config=plot_config, weighting_mode=weighting_mode, save_plot=SAVE_INDIVIDUAL_CONDITION_PLOTS, vary_C_by_cue_type=vary_C_by_cue_type, vary_t0_by_cue_type=vary_t0_by_cue_type, vary_k_by_cue_type=vary_k_by_cue_type, cue_condition_type=Config.cue_condition_type(cue_cond), use_contaminant=Config.USE_CONTAMINANT_FLOOR_ALLCONDITIONS, contaminant_alpha=contam_alpha_use, contaminant_rt_max=contam_rtmax_use, estimate_contaminant=Config.ESTIMATE_CONTAMINANT_ALLCONDITIONS, layout=layout)
        push!(individual_plots, p)
    end

    # Step 6: Generate combined RT fit plot for all conditions
    if !isempty(individual_plots)
        println("\n" * "=" ^ 70)
        println("GENERATING COMBINED RT FIT PLOT FOR ALL CONDITIONS")
        println("=" ^ 70)

        # Calculate grid dimensions (aim for roughly square layout)
        n_plots = length(individual_plots)
        n_cols = ceil(Int, sqrt(n_plots))
        n_rows = ceil(Int, n_plots / n_cols)

        # Create combined plot
        combined_plot = plot(individual_plots...,
                            layout=(n_rows, n_cols),
                            size=(n_cols * 600, n_rows * 500),
                            plot_title="All-Conditions Model (Shared Parameters) - Participant $(data_config.participant_id)",
                            plot_titlefontsize=18,
                            titlefontsize=14,
                            legendfontsize=FittingUtils.PlottingUtils.AXIS_FONT_SIZE,
                            guidefontsize=FittingUtils.PlottingUtils.AXIS_FONT_SIZE,
                            tickfontsize=FittingUtils.PlottingUtils.AXIS_FONT_SIZE,
                            fontsize=FittingUtils.PlottingUtils.AXIS_FONT_SIZE,
                            ylims=FittingUtils.PlottingUtils.RT_ALLCONDITIONS_YLIM)

        # Save combined plot
        combined_plot_path = joinpath(images_dir, "model_fit_plot_allconditions_P$(data_config.participant_id)_all_conditions$(flag_suffix).png")
        savefig(combined_plot, combined_plot_path)
        println("Saved combined RT fit plot to $combined_plot_path")
    end

    # Step 7: Generate overall accuracy plot
    if !isempty(condition_data_dict)
        println("\n" * "=" ^ 70)
        println("GENERATING OVERALL ACCURACY PLOT")
        println("=" ^ 70)

        overall_accuracy_plot = joinpath(images_dir, "accuracy_plot_allconditions_P$(data_config.participant_id)_all_conditions$(flag_suffix).png")
        generate_overall_accuracy_plot_allconditions(condition_data_dict, best_params, overall_accuracy_plot; r_max=r_max, weighting_mode=weighting_mode, vary_C_by_cue_type=vary_C_by_cue_type, vary_t0_by_cue_type=vary_t0_by_cue_type, vary_k_by_cue_type=vary_k_by_cue_type, cue_condition_type_fn=Config.cue_condition_type, use_contaminant=Config.USE_CONTAMINANT_FLOOR_ALLCONDITIONS, contaminant_alpha=contam_alpha_use, estimate_contaminant=Config.ESTIMATE_CONTAMINANT_ALLCONDITIONS, layout=layout)
    end

    println("\n" * "=" ^ 70)
    println("ANALYSIS COMPLETE")
    println("=" ^ 70)
    println("Participant: $(data_config.participant_id)")
    println("Model: Single LBA with shared parameters across ALL conditions (C/t0 cue-specific flags applied as configured)")
    println("\nResults saved to:")
    println("  - Parameters: outputdata/model_fit_results_allconditions_P$(data_config.participant_id).csv")
    if SAVE_INDIVIDUAL_CONDITION_PLOTS
        println("  - Individual condition plots: images/model_fit_plot_allconditions_P$(data_config.participant_id)_condition_*$(flag_suffix).png")
    else
        println("  - Individual condition plots skipped (SAVE_INDIVIDUAL_CONDITION_PLOTS=false)")
    end
    println("  - Combined plot: images/model_fit_plot_allconditions_P$(data_config.participant_id)_all_conditions$(flag_suffix).png")
    println("  - Accuracy plot: images/accuracy_plot_allconditions_P$(data_config.participant_id)_all_conditions$(flag_suffix).png")
    println("\nNote: This model uses shared parameters across conditions, with optional cue-count-specific C/t0 parameters.")
    println("      Separate predictions are still generated for each condition.")
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
