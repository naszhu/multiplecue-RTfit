# ==========================================================================
# Plotting for Dual-Mode LBA (mixture of two LBAs with shared weights)
# ==========================================================================

module PlottingUtilsDualModes

export mixture_rt_plot, accuracy_plot_dualmodes

ENV["GKSwstype"] = "100"

using DataFrames
using Distributions
using SequentialSamplingModels
using Statistics
using Plots
using ..ConfigDualModes

const AXIS_FONT_SIZE = 12

function mixture_rt_plot(data::DataFrame, params::Vector{<:Real}; weighting_mode::Symbol=:free, cue_condition::Any=nothing, cue_condition_type::Symbol=:single)
    # Unpack
    p_idx = 1
    C_fast = params[p_idx]; p_idx+=1
    C_slow = params[p_idx]; p_idx+=1
    if weighting_mode == :free
        w2 = params[p_idx]; p_idx+=1
        w3 = params[p_idx]; p_idx+=1
        w4 = params[p_idx]; p_idx+=1
    else
        w2=w3=w4=0.0
    end
    A = params[p_idx]; p_idx+=1
    k_fast = params[p_idx]; p_idx+=1
    k_slow = params[p_idx]; p_idx+=1
    t0 = params[p_idx]; p_idx+=1
    pi_single = params[p_idx]; p_idx+=1
    pi_double = params[p_idx]; p_idx+=1
    pi_use = cue_condition_type==:double ? pi_double : pi_single

    # KDE observed
    rt_min, rt_max = minimum(data.CleanRT), maximum(data.CleanRT)
    rt_range = rt_max - rt_min
    kde_grid = range(max(0.05, rt_min-0.1*rt_range), min(1.5, rt_max+0.1*rt_range), length=200)
    n = length(data.CleanRT)
    bandwidth = max(0.01, 0.9 * min(std(data.CleanRT), (quantile(data.CleanRT,0.75)-quantile(data.CleanRT,0.25))/1.34) * n^(-1/5))
    kde_dens = [mean(pdf(Normal(t, bandwidth), rt) for rt in data.CleanRT) for t in kde_grid]
    dx = kde_grid[2]-kde_grid[1]
    kde_dens ./= sum(kde_dens)*dx

    weight_lookup = weighting_mode==:free ? Dict(1.0=>1.0,2.0=>w2,3.0=>w3,4.0=>w4,0.0=>1e-10) : nothing
    default_weight = weighting_mode==:free ? weight_lookup[0.0] : 1e-10
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
        ws = weighting_mode==:exponential ? exp.(params[3] .* rewards ./ r_max) : [get(weight_lookup,r,default_weight) for r in rewards]
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

    p = plot(kde_grid, kde_dens, label="Observed", color=:blue, lw=2, xlabel="RT (s)", ylabel="Density", title="Cue $cue_condition", legend=:topright, guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE)
    plot!(p, t_grid, y_mix, label="Mixture RT", color=:red, lw=3)
    plot!(p, t_grid, y_fast, label="Fast LBA", color=:green, lw=2, linestyle=:dash)
    plot!(p, t_grid, y_slow, label="Slow LBA", color=:orange, lw=2, linestyle=:dash)
    return p
end

function accuracy_plot_dualmodes(condition_data::Dict{Any,DataFrame}, params::Vector{<:Real}; weighting_mode::Symbol=:free)
    p_idx=1
    C_fast=params[p_idx]; p_idx+=1
    C_slow=params[p_idx]; p_idx+=1
    if weighting_mode==:free
        w2=params[p_idx]; p_idx+=1
        w3=params[p_idx]; p_idx+=1
        w4=params[p_idx]; p_idx+=1
    else
        w2=w3=w4=0.0
    end
    A=params[p_idx]; p_idx+=1
    k_fast=params[p_idx]; p_idx+=1
    k_slow=params[p_idx]; p_idx+=1
    t0=params[p_idx]; p_idx+=1
    pi_single=params[p_idx]; p_idx+=1
    pi_double=params[p_idx]; p_idx+=1

    weight_lookup = weighting_mode==:free ? Dict(1.0=>1.0,2.0=>w2,3.0=>w3,4.0=>w4,0.0=>1e-10) : nothing
    default_weight = weighting_mode==:free ? weight_lookup[0.0] : 1e-10
    r_max = 4.0
    t_grid = range(0.05,3.0,length=800)
    dt = t_grid[2]-t_grid[1]

    observed_acc = Float64[]
    predicted_acc = Float64[]
    labels = String[]
    ns = Int[]

    for cc in sort(collect(keys(condition_data)))
        df = condition_data[cc]
        cond_type = ConfigDualModes.cue_condition_type(cc)
        pi_use = cond_type==:double ? pi_double : pi_single
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
            ws = weighting_mode==:exponential ? exp.(params[3] .* rewards ./ r_max) : [get(weight_lookup,r,default_weight) for r in rewards]
            rel = ws ./ sum(ws)
            drift_fast = C_fast .* rel
            drift_slow = C_slow .* rel
            lba_fast = LBA(ν=drift_fast, A=A, k=k_fast, τ=t0)
            lba_slow = LBA(ν=drift_slow, A=A, k=k_slow, τ=t0)
            pred_prob=0.0
            for t in t_grid
                if t>t0
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

end # module
