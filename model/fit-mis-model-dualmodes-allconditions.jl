# ==========================================================================
# Dual-Mode MIS-LBA (Two LBAs with shared weights) across all cue conditions
# ==========================================================================

using Pkg

include("data_utils.jl")
include("model_utils.jl")
include("config.jl")
include("config_dualmodes.jl")
include("plotting_utils_dualmodes.jl")
include("optimization_utils.jl")
include("results_utils.jl")

using .DataUtils
using .ModelUtils
using .Config
using .ConfigDualModes
using .PlottingUtilsDualModes
using .OptimizationUtils
using .ResultsUtils
using DataFrames
using Optim
using Plots

function run_analysis()
    data_config = ConfigDualModes.get_data_config(ConfigDualModes.PARTICIPANT_ID_DUALMODES)
    weighting_mode = ConfigDualModes.get_weighting_mode()
    println("Participant: $(data_config.participant_id)")
    println("Weighting mode: $weighting_mode")

    data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)
    cue_conditions = unique(data.CueCondition)
    filter!(x -> !ismissing(x), cue_conditions)
    sort!(cue_conditions)
    cue_condition_types = Config.cue_condition_type.(data.CueCondition)

    params_cfg = get_default_dualmodes_params(weighting_mode)
    lower = params_cfg.lower
    upper = params_cfg.upper
    x0 = params_cfg.x0
    param_names = weighting_mode == :free ?
        ["C_fast","C_slow","w2","w3","w4","A","k_fast","k_slow","t0","pi_single","pi_double"] :
        ["C_fast","C_slow","w_slope","A","k_fast","k_slow","t0","pi_single","pi_double"]

    println("Parameter bounds/starts:")
    for (i,n) in enumerate(param_names)
        println("$n: [$(lower[i]), $(upper[i])] x0=$(x0[i])")
    end

    preprocessed = preprocess_data_for_fitting(data; cue_condition_types=cue_condition_types, group_by_condition=true)
    objective = (x,d)->mis_lba_dualmodes_loglike(x,d; weighting_mode=weighting_mode, cue_condition_types=cue_condition_types)
    result = fit_model(preprocessed, objective; lower=lower, upper=upper, x0=x0, time_limit=600.0)
    best = Optim.minimizer(result)

    println("Fitted params:")
    for (i,n) in enumerate(param_names)
        println("$n = $(best[i])")
    end

    # Save results
    save_results_allconditions(result, ConfigDualModes.OUTPUT_CSV_DUALMODES; param_names=param_names)

    # Plots
    img_dir = joinpath(@__DIR__,"images")
    if !isdir(img_dir)
        mkdir(img_dir)
    end
    plots = []
    for cc in cue_conditions
        cond_df = filter(r->r.CueCondition==cc, data)
        cond_type = Config.cue_condition_type(cc)
        p = mixture_rt_plot(cond_df, best; weighting_mode=weighting_mode, cue_condition=cc, cue_condition_type=cond_type)
        savefig(p, joinpath(img_dir, "model_fit_dualmodes_P$(data_config.participant_id)_cond$(cc).png"))
        push!(plots, p)
    end
    println("Analysis complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
