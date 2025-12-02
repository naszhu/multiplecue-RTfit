# ==========================================================================
# MIS Poisson Random Walk All-Conditions Fitting Script
# ==========================================================================
#
# Fits MIS with the Poisson random walk decision stage across all cue
# conditions. Shares parameterization with the LBA scripts for data loading
# and preprocessing but swaps the likelihood for the PRW implementation.
# ==========================================================================

using Pkg
using Optim

include("data_utils.jl")
include("model_utils.jl")
include("config_prw.jl")
include("optimization_utils_prw.jl")
include("results_utils.jl")
include("prw_model_utils.jl")
include("plotting_utils_prw.jl")

using .DataUtils
using .ModelUtils
using .ConfigPRW
using .OptimizationUtilsPRW
using .ResultsUtils
using .PRWModel
using .PlottingUtilsPRW
using DataFrames
using Random
using Plots

const PRW_K_BOUNDS = (1.0, 12.0, 4.0)  # integer threshold; mixed via Blurton et al. Appendix C
const PRW_MAX_STEPS = 20

"""
    build_prw_allconditions_params(weighting_mode; vary_C_by_cue, vary_t0_by_cue, vary_k_by_cue, c_start_override)

Construct bounds, starts, and index layout for PRW all-conditions fitting.
"""
function build_prw_allconditions_params(weighting_mode::Symbol=ConfigPRW.DEFAULT_WEIGHTING_MODE_PRW;
    vary_C_by_cue::Bool=ConfigPRW.VARY_C_BY_CUECOUNT_PRW,
    vary_t0_by_cue::Bool=ConfigPRW.VARY_T0_BY_CUECOUNT_PRW,
    vary_k_by_cue::Bool=ConfigPRW.VARY_K_BY_CUECOUNT_PRW,
    c_start_override::Union{Nothing,Real,Tuple}=nothing)

    names = String[]
    lower = Float64[]
    upper = Float64[]
    x0 = Float64[]

    idx_C = Dict{Symbol,Int}()
    idx_k = Dict{Symbol,Int}()
    idx_t0 = Dict{Symbol,Int}()
    idx_w = Dict{Symbol,Int}()

    pushp!(n, lo, hi, start) = (push!(names, n); push!(lower, lo); push!(upper, hi); push!(x0, start); length(names))

    c_lo, c_hi, c_start_default = ConfigPRW.C_BOUNDS_PRW
    c_single_start = isnothing(c_start_override) ? c_start_default : (isa(c_start_override, Tuple) ? c_start_override[1] : c_start_override)
    c_double_start = isnothing(c_start_override) ? c_start_default : (isa(c_start_override, Tuple) ? c_start_override[end] : c_start_override)
    if vary_C_by_cue
        idx_C[:single] = pushp!("C_single", c_lo, c_hi, c_single_start)
        idx_C[:double] = pushp!("C_double", c_lo, c_hi, c_double_start)
    else
        idx_C[:all] = pushp!("C_all", c_lo, c_hi, c_single_start)
    end

    if weighting_mode == :exponential
        ws_lo, ws_hi, ws_start = ConfigPRW.W_SLOPE_BOUNDS_PRW
        idx_w[:w_slope] = pushp!("w_slope", ws_lo, ws_hi, ws_start)
    elseif weighting_mode == :free
        for sym in (:w2, :w3, :w4)
            bnds = ConfigPRW.W_FREE_BOUNDS_PRW[sym]
            idx_w[sym] = pushp!(String(sym), bnds...)
        end
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free.")
    end

    k_lo, k_hi, k_start = PRW_K_BOUNDS
    if vary_k_by_cue
        idx_k[:single] = pushp!("k_single", k_lo, k_hi, k_start)
        idx_k[:double] = pushp!("k_double", k_lo, k_hi, k_start)
    else
        idx_k[:all] = pushp!("k_all", k_lo, k_hi, k_start)
    end

    t0_lo, t0_hi, t0_start = ConfigPRW.T0_BOUNDS_PRW
    if vary_t0_by_cue
        idx_t0[:single] = pushp!("t0_single", t0_lo, t0_hi, t0_start)
        idx_t0[:double] = pushp!("t0_double", t0_lo, t0_hi, t0_start)
    else
        idx_t0[:all] = pushp!("t0_all", t0_lo, t0_hi, t0_start)
    end

    layout = PRWLayout(weighting_mode, vary_C_by_cue, vary_t0_by_cue, vary_k_by_cue, idx_C, idx_t0, idx_k, idx_w)
    return ConfigPRW.AllConditionsParams(lower, upper, x0), layout, names
end

# ==========================================================================
# MAIN ANALYSIS FUNCTION
# ==========================================================================

function run_analysis()
    data_config = ConfigPRW.get_data_config_prw(ConfigPRW.PARTICIPANT_ID_PRW)
    println("=" ^ 70)
    println("PARTICIPANT SELECTION")
    println("=" ^ 70)
    println("Selected Participant ID: $(data_config.participant_id)")
    println("Data path: $(data_config.data_base_path)")

    weighting_mode = isnothing(ConfigPRW.WEIGHTING_MODE_OVERRIDE_PRW) ? ConfigPRW.DEFAULT_WEIGHTING_MODE_PRW : ConfigPRW.WEIGHTING_MODE_OVERRIDE_PRW
    if !(weighting_mode in (:exponential, :free))
        error("PRW supports weighting_mode :exponential or :free; got $(weighting_mode)")
    end
    vary_C_by_cue_type = ConfigPRW.VARY_C_BY_CUECOUNT_PRW
    vary_t0_by_cue_type = ConfigPRW.VARY_T0_BY_CUECOUNT_PRW
    vary_k_by_cue_type = ConfigPRW.VARY_K_BY_CUECOUNT_PRW

    println("Reward weighting mode: $weighting_mode")
    println("Vary C by cue-count (single vs double): $vary_C_by_cue_type")
    println("Vary t0 by cue-count (single vs double): $vary_t0_by_cue_type")
    println("Vary k by cue-count (single vs double): $vary_k_by_cue_type")

    # Create images subfolder if it doesn't exist (reused by plotting utilities)
    images_dir = ConfigPRW.IMAGES_DIR_PRW
    if !isdir(images_dir)
        mkdir(images_dir)
        println("Created images directory: $images_dir")
    end

    println("\n" * "=" ^ 70)
    println("LOADING DATA")
    println("=" ^ 70)
    data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)
    debug_sample = parse(Int, get(ENV, "PRW_DEBUG_SAMPLE", "0"))
    if debug_sample > 0 && nrow(data) > debug_sample
        println("DEBUG: Downsampling data to $debug_sample trials via PRW_DEBUG_SAMPLE")
        idx = randperm(nrow(data))[1:debug_sample]
        data = data[idx, :]
        println("DEBUG: After downsample nrow = $(nrow(data))")
    end

    if !("CueCondition" in names(data))
        error("CueCondition column not found in data. Please ensure data files contain this column.")
    end
    cue_conditions = unique(data.CueCondition)
    filter!(x -> !ismissing(x), cue_conditions)
    sort!(cue_conditions)
    println("\nFound $(length(cue_conditions)) unique cue conditions.")

    cue_condition_types = ConfigPRW.cue_condition_type_prw.(data.CueCondition)
    println("Total trials across all conditions: $(nrow(data))")

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

    println("\n" * "=" ^ 70)
    println("PREPROCESSING DATA FOR OPTIMIZATION")
    println("=" ^ 70)
    group_by_condition = vary_C_by_cue_type || vary_t0_by_cue_type || vary_k_by_cue_type
    preprocessed_data = preprocess_data_for_fitting(data; cue_condition_types=cue_condition_types, group_by_condition=group_by_condition)

    params_config, layout, param_names = build_prw_allconditions_params(weighting_mode;
        vary_C_by_cue=vary_C_by_cue_type,
        vary_t0_by_cue=vary_t0_by_cue_type,
        vary_k_by_cue=vary_k_by_cue_type,
        c_start_override=nothing)

    lower = params_config.lower
    upper = params_config.upper
    x0 = params_config.x0

    println("\n" * "=" ^ 70)
    println("FITTING MIS PRW MODEL TO ALL CONDITIONS")
    println("=" ^ 70)
    println("Parameters to be fitted: $(join(param_names, ", "))")
    println("Number of conditions: $(length(cue_conditions))")
    println("Total trials: $(nrow(data))")
    max_steps_use = debug_sample > 0 ? min(PRW_MAX_STEPS, 10) : PRW_MAX_STEPS
    println("Max PRW steps per trial: $max_steps_use")

    println("\n" * "-" ^ 70)
    println("OPTIMIZATION SETUP")
    println("-" ^ 70)
    println("Parameter bounds:")
    for (i, name) in enumerate(param_names)
        println("  $name: [$(lower[i]), $(upper[i])], initial: $(x0[i])")
    end

    objective_func = (x, d) -> mis_prw_allconditions_loglike(x, d; layout=layout, r_max=r_max, max_steps=max_steps_use)
    opt_cfg = ConfigPRW.get_optimization_config_prw()
    time_limit_use = debug_sample > 0 ? min(20.0, opt_cfg.time_limit) : opt_cfg.time_limit
    result = fit_model_prw(preprocessed_data, objective_func;
                       lower=lower, upper=upper, x0=x0, time_limit=time_limit_use)

    best_params = Optim.minimizer(result)
    println("\nBest-fit parameters:")
    for (name, val) in zip(param_names, best_params)
        println("  $name = $(round(val, digits=4))")
    end

    if !isdir(ConfigPRW.OUTPUTDATA_DIR_PRW)
        mkdir(ConfigPRW.OUTPUTDATA_DIR_PRW)
    end

    # Save parameters
    results_path = joinpath(ConfigPRW.OUTPUTDATA_DIR_PRW, "model_fit_results_prw_allconditions_P$(data_config.participant_id).csv")
    save_results_allconditions(result, results_path; param_names=param_names)

    # Organize data by cue condition for plotting
    condition_data = Dict{Any,DataFrame}()
    for cc in unique(data.CueCondition)
        condition_data[cc] = data[data.CueCondition .== cc, :]
    end

    # Combined RT grid across all cue conditions (no individual saves)
    plots_for_grid = Plots.Plot[]
    sorted_cc = sort(collect(keys(condition_data)))
    for cc in sorted_cc
        df_cond = condition_data[cc]
        p = generate_prw_condition_plot(df_cond, best_params, layout; r_max=r_max, max_steps=PRW_MAX_STEPS, output_path=nothing)
        push!(plots_for_grid, p)
    end
    if !isempty(plots_for_grid)
        n = length(plots_for_grid)
        cols = ceil(Int, sqrt(n))
        rows = ceil(Int, n / cols)
        combined = plot(plots_for_grid..., layout=(rows, cols), size=(cols*500, rows*400))
        combined_path = joinpath(ConfigPRW.IMAGES_DIR_PRW, "prw_fit_all_conditions_grid_P$(data_config.participant_id).png")
        savefig(combined, combined_path)
        println("Saved combined PRW RT grid to $combined_path")
    end

    # Accuracy plot across conditions
    acc_plot_path = joinpath(ConfigPRW.IMAGES_DIR_PRW, "accuracy_prw_allconditions_P$(data_config.participant_id).png")
    generate_overall_accuracy_plot_prw_allconditions(condition_data, best_params, layout; r_max=r_max, max_steps=PRW_MAX_STEPS, output_plot=acc_plot_path)

    println("\nAnalysis complete. Outputs:")
    println("  Params: $(results_path)")
    println("  Accuracy plot: $(acc_plot_path)")
    rt_grid_path = joinpath(ConfigPRW.IMAGES_DIR_PRW, "prw_fit_all_conditions_grid_P$(data_config.participant_id).png")
    println("  RT grid: $(rt_grid_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
