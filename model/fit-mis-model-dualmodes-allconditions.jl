# ==========================================================================
# Dual-Mode MIS-LBA (Two LBAs with shared weights) across all cue conditions
# ==========================================================================

using Pkg

include("data_utils.jl")
include("model_utils.jl")
include("config_dualmodes.jl")
include("plotting_utils_dualmodes.jl")
include("optimization_utils.jl")
include("results_utils.jl")

using .DataUtils
using .ModelUtils
using .ConfigDualModes
using .PlottingUtilsDualModes
using .OptimizationUtils
using .ResultsUtils
using DataFrames
using Optim
using Plots

function run_analysis()
    data_config = get_data_config(PARTICIPANT_ID_DUALMODES)
    weighting_mode = get_weighting_mode()
    println("Participant: $(data_config.participant_id)")
    println("Weighting mode: $weighting_mode")

    data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)
    cue_conditions = unique(data.CueCondition)
    filter!(x -> !ismissing(x), cue_conditions)
    sort!(cue_conditions)
    cue_condition_types = cue_condition_type.(data.CueCondition)

    params_cfg, layout, param_names = build_dualmodes_params(weighting_mode)
    lower = params_cfg.lower
    upper = params_cfg.upper
    x0 = params_cfg.x0

    # Build suffix describing variation
    tokens = String[]
    push!(tokens, weighting_mode == :free ? "wfree" : "wslope")
    if layout.vary_C_by_mode push!(tokens, "Cmode") end
    if layout.vary_C_by_cue push!(tokens, "Ccue") end
    if layout.vary_k_by_mode push!(tokens, "kmode") end
    if layout.vary_k_by_cue push!(tokens, "kcue") end
    if layout.vary_t0_by_mode push!(tokens, "t0mode") end
    if layout.vary_t0_by_cue push!(tokens, "t0cue") end
    if layout.vary_A_by_mode push!(tokens, "Amode") end
    if layout.vary_A_by_cue push!(tokens, "Acue") end
    if layout.vary_pi_by_cue push!(tokens, "picue") end
    if layout.use_contaminant
        if layout.estimate_contaminant
            push!(tokens, "contamEst")
            if layout.vary_contam_alpha_by_mode push!(tokens, "alphaMode") end
            if layout.vary_contam_alpha_by_cue push!(tokens, "alphaCue") end
            if layout.vary_contam_rt_by_mode push!(tokens, "rtMode") end
            if layout.vary_contam_rt_by_cue push!(tokens, "rtCue") end
        else
            push!(tokens, "contamFx")
        end
    end
    flag_suffix = isempty(tokens) ? "" : "_" * join(tokens, "-")

    println("Parameter bounds/starts:")
    for (i,n) in enumerate(param_names)
        println("$n: [$(lower[i]), $(upper[i])] x0=$(x0[i])")
    end

    need_group_by = layout.vary_C_by_cue || layout.vary_k_by_cue || layout.vary_t0_by_cue || layout.vary_A_by_cue || layout.vary_pi_by_cue
    preprocessed = preprocess_data_for_fitting(data; cue_condition_types=cue_condition_types, group_by_condition=need_group_by)
    objective = (x,d)->mis_lba_dualmodes_loglike(x,d, layout; cue_condition_types=cue_condition_types)
    result = fit_model(preprocessed, objective; lower=lower, upper=upper, x0=x0, time_limit=600.0)
    best = Optim.minimizer(result)

    println("Fitted params:")
    for (i,n) in enumerate(param_names)
        println("$n = $(best[i])")
    end

    # Save results
    csv_dir = joinpath(@__DIR__, "outputdata")
    if !isdir(csv_dir); mkdir(csv_dir) end
    csv_path = joinpath(csv_dir, "model_fit_results_dualmodes_P$(data_config.participant_id)$(flag_suffix).csv")
    save_results_allconditions(result, csv_path; param_names=param_names)

    # Plots
    img_dir = joinpath(@__DIR__,"images")
    if !isdir(img_dir)
        mkdir(img_dir)
    end
    plots = []
    for cc in cue_conditions
        cond_df = filter(r->r.CueCondition==cc, data)
        cond_type = cue_condition_type(cc)
        p = mixture_rt_plot(cond_df, best, layout; cue_condition=cc, cue_condition_type=cond_type)
        push!(plots, p)
    end
    if !isempty(plots)
        n_plots = length(plots)
        n_cols = ceil(Int, sqrt(n_plots))
        n_rows = ceil(Int, n_plots / n_cols)
        combined = plot(plots..., layout=(n_rows,n_cols), size=(n_cols*500, n_rows*400))
        savefig(combined, joinpath(img_dir, "model_fit_dualmodes_P$(data_config.participant_id)_allconditions$(flag_suffix).png"))
    end

    # Accuracy plot
    cond_dict = Dict{Any,DataFrame}()
    for cc in cue_conditions
        cond_dict[cc] = filter(r->r.CueCondition==cc, data)
    end
    acc_plot = accuracy_plot_dualmodes(cond_dict, best, layout)
    savefig(acc_plot, joinpath(img_dir, "accuracy_dualmodes_P$(data_config.participant_id)_allconditions$(flag_suffix).png"))
    println("Analysis complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
