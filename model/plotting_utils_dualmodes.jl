# ==========================================================================
# Plotting for Dual-Mode LBA (mixture of two LBAs with shared weights)
# ==========================================================================

ENV["GKSwstype"] = "100"

using DataFrames
using Distributions
using SequentialSamplingModels
using Statistics
using Plots

# Note: config_dualmodes.jl must be included before this file

const AXIS_FONT_SIZE = 12

cue_key(flag, cond_type) = flag ? cond_type : :all
mode_key(flag, mode) = flag ? mode : :shared

function mixture_rt_plot(data::DataFrame, params::Vector{<:Real}, layout; cue_condition::Any=nothing, cue_condition_type::Symbol=:single)
    weighting_mode = layout.weighting_mode
    fetch(dict, vary_mode, vary_cue, mode, cue) = params[dict[mode_key(vary_mode, mode)][cue_key(vary_cue, cue)]]
    pi_use = params[layout.idx_pi[layout.vary_pi_by_cue ? cue_condition_type : :all]]
    C_fast = fetch(layout.idx_C, layout.vary_C_by_mode, layout.vary_C_by_cue, :fast, cue_condition_type)
    C_slow = fetch(layout.idx_C, layout.vary_C_by_mode, layout.vary_C_by_cue, :slow, cue_condition_type)
    k_fast = fetch(layout.idx_k, layout.vary_k_by_mode, layout.vary_k_by_cue, :fast, cue_condition_type)
    k_slow = fetch(layout.idx_k, layout.vary_k_by_mode, layout.vary_k_by_cue, :slow, cue_condition_type)
    A = fetch(layout.idx_A, layout.vary_A_by_mode, layout.vary_A_by_cue, :shared, cue_condition_type)
    t0 = fetch(layout.idx_t0, layout.vary_t0_by_mode, layout.vary_t0_by_cue, :shared, cue_condition_type)

    # KDE observed
    rt_min, rt_max = minimum(data.CleanRT), maximum(data.CleanRT)
    rt_range = rt_max - rt_min
    kde_grid = range(max(0.05, rt_min-0.1*rt_range), min(1.5, rt_max+0.1*rt_range), length=200)
    n = length(data.CleanRT)
    bandwidth = max(0.01, 0.9 * min(std(data.CleanRT), (quantile(data.CleanRT,0.75)-quantile(data.CleanRT,0.25))/1.34) * n^(-1/5))
    kde_dens = [mean(pdf(Normal(t, bandwidth), rt) for rt in data.CleanRT) for t in kde_grid]
    dx = kde_grid[2]-kde_grid[1]
    kde_dens ./= sum(kde_dens)*dx

    r_max = 4.0

    reward_counts = Dict{String,Int}()
    reward_arrays = Dict{String,Vector{Float64}}()
    for r in data.ParsedRewards
        key=string(r)
        reward_counts[key]=get(reward_counts,key,0)+1
        reward_arrays[key]=r
    end

    t_grid = range(0.05,1.5,length=300)
    y_mix = zeros(length(t_grid))
    y_fast = zeros(length(t_grid))
    y_slow = zeros(length(t_grid))
    for (key,rewards) in reward_arrays
        weight = reward_counts[key]
        ws = if weighting_mode==:exponential
            w_slope = params[layout.idx_w[:w_slope]]
            exp.(w_slope .* rewards ./ r_max)
        else
            w2 = params[layout.idx_w[:w2]]; w3 = params[layout.idx_w[:w3]]; w4 = params[layout.idx_w[:w4]]
            wlu = Dict(1.0=>1.0,2.0=>w2,3.0=>w3,4.0=>w4,0.0=>1e-10)
            [get(wlu,r,1e-10) for r in rewards]
        end
        rel = ws ./ sum(ws)
        drift_fast = C_fast .* rel
        drift_slow = C_slow .* rel
        lba_fast = LBA(ν=drift_fast, A=A, k=k_fast, τ=t0)
        lba_slow = LBA(ν=drift_slow, A=A, k=k_slow, τ=t0)
        for (j,t) in enumerate(t_grid)
            if t > t0
                try
                    val_fast = pdf(lba_fast,(choice=argmax(rewards),rt=t))
                    val_slow = pdf(lba_slow,(choice=argmax(rewards),rt=t))
                    if isnan(val_fast)||isinf(val_fast) val_fast=0.0 end
                    if isnan(val_slow)||isinf(val_slow) val_slow=0.0 end
                    y_fast[j] += weight * val_fast
                    y_slow[j] += weight * val_slow
                    y_mix[j] += weight * (pi_use*val_fast + (1-pi_use)*val_slow)
                catch
                end
            end
        end
    end
    total_w = sum(values(reward_counts))
    if total_w > 0
        y_mix ./= total_w
        y_fast ./= total_w
        y_slow ./= total_w
    end

    # Scale individual mode curves by their mixture weights so they reflect their contribution
    y_fast_weighted = pi_use .* y_fast
    y_slow_weighted = (1 - pi_use) .* y_slow

    p = plot(kde_grid, kde_dens, label="Observed", color=:blue, lw=2, xlabel="RT (s)", ylabel="Density", title="Cue $cue_condition", legend=:topright, guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE, ylim=(0.0,10.0), xlim=(0.0,0.8))
    plot!(p, t_grid, y_mix, label="Mixture RT", color=:red, lw=3)
    plot!(p, t_grid, y_fast_weighted, label="Fast LBA (weighted)", color=:green, lw=2, linestyle=:dash)
    plot!(p, t_grid, y_slow_weighted, label="Slow LBA (weighted)", color=:orange, lw=2, linestyle=:dash)
    return p
end

function accuracy_plot_dualmodes(condition_data::Dict{Any,DataFrame}, params::Vector{<:Real}, layout; weighting_mode::Symbol=layout.weighting_mode)
    fetch(dict, vary_mode, vary_cue, mode, cue) = params[dict[mode_key(vary_mode, mode)][cue_key(vary_cue, cue)]]
    r_max = 4.0
    t_grid = range(0.05,3.0,length=800)
    dt = t_grid[2]-t_grid[1]

    observed_acc = Float64[]
    predicted_acc = Float64[]
    labels = String[]
    ns = Int[]

    for cc in sort(collect(keys(condition_data)))
        df = condition_data[cc]
        cond_type = cue_condition_type(cc)
        pi_use = params[layout.idx_pi[layout.vary_pi_by_cue ? cond_type : :all]]
        groups = Dict{String,Any}()
        for row in eachrow(df)
            key=string(row.ParsedRewards)
            if !haskey(groups,key)
                groups[key]=(rewards=row.ParsedRewards, choices=Int[], n=0)
            end
            push!(groups[key].choices, row.Choice)
            groups[key] = (rewards=groups[key].rewards, choices=groups[key].choices, n=groups[key].n+1)
        end
        total_obs=0; total_pred=0.0; total_w=0
        for (key,g) in groups
            rewards=g.rewards
            target=argmax(rewards)
            choices=g.choices
            ntr=g.n
            obs = sum(choices .== target)
            total_obs += obs
            total_w += ntr
            ws = if weighting_mode==:exponential
                w_slope = params[layout.idx_w[:w_slope]]
                exp.(w_slope .* rewards ./ r_max)
            else
                w2 = params[layout.idx_w[:w2]]; w3=params[layout.idx_w[:w3]]; w4=params[layout.idx_w[:w4]]
                wlu = Dict(1.0=>1.0,2.0=>w2,3.0=>w3,4.0=>w4,0.0=>1e-10)
                [get(wlu,r,1e-10) for r in rewards]
            end
            rel = ws ./ sum(ws)
            C_fast = fetch(layout.idx_C, layout.vary_C_by_mode, layout.vary_C_by_cue, :fast, cond_type)
            C_slow = fetch(layout.idx_C, layout.vary_C_by_mode, layout.vary_C_by_cue, :slow, cond_type)
            drift_fast = C_fast .* rel
            drift_slow = C_slow .* rel
            A_use = fetch(layout.idx_A, layout.vary_A_by_mode, layout.vary_A_by_cue, :shared, cond_type)
            k_fast = fetch(layout.idx_k, layout.vary_k_by_mode, layout.vary_k_by_cue, :fast, cond_type)
            k_slow = fetch(layout.idx_k, layout.vary_k_by_mode, layout.vary_k_by_cue, :slow, cond_type)
            t0_use = fetch(layout.idx_t0, layout.vary_t0_by_mode, layout.vary_t0_by_cue, :shared, cond_type)
            lba_fast = LBA(ν=drift_fast, A=A_use, k=k_fast, τ=t0_use)
            lba_slow = LBA(ν=drift_slow, A=A_use, k=k_slow, τ=t0_use)
            pred_prob=0.0
            for t in t_grid
                if t>t0_use
                    try
                        pred_prob += (pi_use*pdf(lba_fast,(choice=target,rt=t)) + (1-pi_use)*pdf(lba_slow,(choice=target,rt=t))) * dt
                    catch
                    end
                end
            end
            total_pred += pred_prob * ntr
        end
        push!(observed_acc, total_w>0 ? total_obs/total_w : 0.0)
        push!(predicted_acc, total_w>0 ? total_pred/total_w : 0.0)
        push!(labels, string(cc))
        push!(ns, total_w)
    end

    x = 1:length(labels)
    p = plot(x, observed_acc, seriestype=:scatter, label="Observed", color=:blue, markersize=8, xlabel="Cue Condition", ylabel="Accuracy", ylim=(0.0,1.05), legend=:bottomright, xticks=(x, labels), title="Dual-Mode LBA Accuracy", guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE)
    scatter!(p, x, predicted_acc, label="Predicted", color=:red, markershape=:x, markersize=8)
    for (i,n) in enumerate(ns)
        annotate!(p, i, 0.05, text("n=$n", :gray, 8, :center))
    end
    return p
end
