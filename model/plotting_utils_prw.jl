# ==========================================================================
# Plotting Utilities for MIS Poisson Random Walk
# Simplified overlays of observed RT densities and predicted PRW PDFs,
# plus overall accuracy plots per cue condition.
# ==========================================================================

module PlottingUtilsPRW

ENV["GKSwstype"] = "100"

using DataFrames
using Statistics
using Plots
using KernelDensity

include("config_prw.jl")
using .ConfigPRW
using Main.PRWModel

const RT_ALLCONDITIONS_YLIM = ConfigPRW.RT_ALLCONDITIONS_YLIM_PRW
const AXIS_FONT_SIZE = ConfigPRW.AXIS_FONT_SIZE_PRW

export generate_prw_condition_plot, generate_overall_accuracy_plot_prw_allconditions

"""
    _compute_abs_cache(weights, k_use, max_steps)

Helper to compute floor/ceil absorption matrices and mixing weights for fractional k.
"""
function _compute_abs_cache(step_probs, k_use::Float64, max_steps::Int)
    k_floor = floor(Int, k_use)
    k_ceil = ceil(Int, k_use)
    p_ceil = k_use - k_floor
    p_floor = 1.0 - p_ceil

    T_floor = PRWModel.build_transition_matrix(step_probs, k_floor)
    abs_floor = PRWModel.compute_absorption_probs(T_floor, step_probs, k_floor, max_steps)
    abs_ceil = if k_ceil > k_floor
        T_ceil = PRWModel.build_transition_matrix(step_probs, k_ceil)
        PRWModel.compute_absorption_probs(T_ceil, step_probs, k_ceil, max_steps)
    else
        abs_floor
    end
    return abs_floor, abs_ceil, p_floor, p_ceil
end

"""
    _weights_for_rewards(weighting_mode, params, layout, rewards, r_max)
"""
function _weights_for_rewards(weighting_mode::Symbol, params, layout, rewards, r_max)
    if weighting_mode == :exponential
        w_slope = params[layout.idx_w[:w_slope]]
        return exp.(w_slope .* rewards ./ r_max)
    else
        w2 = params[layout.idx_w[:w2]]
        w3 = params[layout.idx_w[:w3]]
        w4 = params[layout.idx_w[:w4]]
        lookup = Dict(1.0=>1.0, 2.0=>w2, 3.0=>w3, 4.0=>w4, 0.0=>1e-10)
        return [get(lookup, r, lookup[0.0]) for r in rewards]
    end
end

"""
    generate_prw_condition_plot(df, params, layout; r_max=4.0, max_steps=60, output_path=\"plot.png\")

Plot observed RT KDE for a condition and overlay PRW predicted PDF (all choices).
Returns the Plots.Plot object.
"""
function generate_prw_condition_plot(df::DataFrame, params, layout; r_max::Float64=4.0, max_steps::Int=60, output_path::Union{Nothing,String}=nothing, use_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)
    cond_type = first(ConfigPRW.cue_condition_type_prw.(df.CueCondition))
    C_use = cond_type == :double && haskey(layout.idx_C, :double) ? params[layout.idx_C[:double]] : params[layout.idx_C[haskey(layout.idx_C, :all) ? :all : :single]]
    k_use = cond_type == :double && haskey(layout.idx_k, :double) ? params[layout.idx_k[:double]] : params[layout.idx_k[haskey(layout.idx_k, :all) ? :all : :single]]
    t0_use = cond_type == :double && haskey(layout.idx_t0, :double) ? params[layout.idx_t0[:double]] : params[layout.idx_t0[haskey(layout.idx_t0, :all) ? :all : :single]]

    # KDE of observed RTs using KernelDensity
    rt_vals = df.CleanRT
    rt_min, rt_max = minimum(rt_vals), maximum(rt_vals)
    grid = range(max(0.05, rt_min - 0.2), stop=rt_max + 0.2, length=300)
    kde_res = kde(rt_vals)
    kde_dens = pdf(kde_res, collect(grid))
    dx = grid[2] - grid[1]
    kde_dens ./= max(sum(kde_dens) * dx, 1e-9)

    # Predicted unconditional RT PDF: average over unique rewards
    uniq = unique(df.ParsedRewards)
    counts = Dict{Vector{Float64},Int}()
    for r in uniq counts[r] = 0 end
    for r in df.ParsedRewards
        counts[r] += 1
    end

    pred_pdf = zeros(length(grid))
    for rewards in uniq
        weight = counts[rewards]
        ws = _weights_for_rewards(layout.weighting_mode, params, layout, rewards, r_max)
        step_probs = ws ./ sum(ws)
        abs_floor, abs_ceil, p_floor, p_ceil = _compute_abs_cache(step_probs, k_use, max_steps)
        for (j, t) in enumerate(grid)
            # unconditional over choices
            dens_c = 0.0
            for c in 1:length(rewards)
                dens_c += p_floor * PRWModel.prw_pdf_point(t, t0_use, C_use, abs_floor, c)
                dens_c += p_ceil * PRWModel.prw_pdf_point(t, t0_use, C_use, abs_ceil, c)
            end
            if use_contaminant
                dens_c = (1 - contaminant_alpha) * dens_c + contaminant_alpha * (1 / contaminant_rt_max)
            end
            pred_pdf[j] += weight * dens_c
        end
    end
    total = sum(values(counts))
    pred_pdf ./= total

    title_str = "PRW Fit - CueCondition $(first(df.CueCondition)) (type=$(cond_type))"
    p = plot(grid, kde_dens, label="Observed KDE", linewidth=2.5, color=:darkblue,
        xlabel="RT (s)", ylabel="Density", title=title_str, legend=:topright,
        ylims=RT_ALLCONDITIONS_YLIM, guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE)
    plot!(p, grid, pred_pdf, label="PRW Predicted", linewidth=2.5, color=:red, alpha=0.8)

    if output_path !== nothing
        savefig(p, output_path)
        println("Saved PRW RT plot to $output_path")
    end
    return p
end

"""
    generate_overall_accuracy_plot_prw_allconditions(condition_data, params, layout; r_max, max_steps, output_plot)

Observed vs predicted accuracy per cue condition for PRW.
"""
function generate_overall_accuracy_plot_prw_allconditions(condition_data::Dict{Any,DataFrame}, params, layout; r_max::Float64=4.0, max_steps::Int=60, output_plot::String="accuracy_plot_prw.png", use_contaminant::Bool=false, contaminant_alpha::Float64=0.0)
    observed_acc = Float64[]
    predicted_acc = Float64[]
    labels = String[]
    trials = Int[]

    sorted_conditions = sort(collect(keys(condition_data)))
    for cc in sorted_conditions
        df = condition_data[cc]
        cond_type = ConfigPRW.cue_condition_type_prw(cc)
        C_use = cond_type == :double && haskey(layout.idx_C, :double) ? params[layout.idx_C[:double]] : params[layout.idx_C[haskey(layout.idx_C, :all) ? :all : :single]]
        k_use = cond_type == :double && haskey(layout.idx_k, :double) ? params[layout.idx_k[:double]] : params[layout.idx_k[haskey(layout.idx_k, :all) ? :all : :single]]
        t0_use = cond_type == :double && haskey(layout.idx_t0, :double) ? params[layout.idx_t0[:double]] : params[layout.idx_t0[haskey(layout.idx_t0, :all) ? :all : :single]]

        # Observed/predicted accuracy aggregated over unique reward sets
        reward_stats = Dict{String,Dict{String,Any}}()
        for row in eachrow(df)
            key = string(row.ParsedRewards)
            if !haskey(reward_stats, key)
                reward_stats[key] = Dict(
                    "rewards" => row.ParsedRewards,
                    "n" => 0,
                    "correct" => 0,
                )
            end
            reward_stats[key]["n"] += 1
            target_choice = argmax(reward_stats[key]["rewards"])
            if row.Choice == target_choice
                reward_stats[key]["correct"] += 1
            end
        end

        acc_obs_num = sum(v["correct"] for v in values(reward_stats))
        acc_obs_den = sum(v["n"] for v in values(reward_stats))

        acc_pred_weighted = 0.0
        total_weight = 0
        for v in values(reward_stats)
            rewards = v["rewards"]
            count = v["n"]
            target_choice = argmax(rewards)
            ws = _weights_for_rewards(layout.weighting_mode, params, layout, rewards, r_max)
            step_probs = ws ./ sum(ws)
            abs_floor, abs_ceil, p_floor, p_ceil = _compute_abs_cache(step_probs, k_use, max_steps)
            p_choice = p_floor * sum(abs_floor[:, target_choice]) + p_ceil * sum(abs_ceil[:, target_choice])
            if use_contaminant
                p_choice = (1 - contaminant_alpha) * p_choice + contaminant_alpha * (1 / length(rewards))
            end
            acc_pred_weighted += count * p_choice
            total_weight += count
        end

        acc_obs = acc_obs_den == 0 ? 0.0 : acc_obs_num / acc_obs_den
        acc_pred = total_weight == 0 ? 0.0 : acc_pred_weighted / total_weight

        push!(observed_acc, acc_obs)
        push!(predicted_acc, acc_pred)
        push!(labels, "CC $cc ($(cond_type))")
        push!(trials, nrow(df))
    end

    x = 1:length(labels)
    p = bar(x .- 0.15, observed_acc, bar_width=0.3, label="Observed", color=:gray70;
        legend=:topright, guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE, ylims=ConfigPRW.ACCURACY_YLIM_PRW)
    bar!(p, x .+ 0.15, predicted_acc, bar_width=0.3, label="Predicted PRW", color=:steelblue;
        legend=:topright, guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE, ylims=ConfigPRW.ACCURACY_YLIM_PRW)
    xticks!(p, x, labels)
    plot!(p; xtickfont=font(8, rotation=20))
    ylabel!(p, "Accuracy")
    title!(p, "Observed vs Predicted Accuracy (PRW)")

    if output_plot != ""
        savefig(p, output_plot)
        println("Saved PRW accuracy plot to $output_plot")
    end

    return p
end

end # module
