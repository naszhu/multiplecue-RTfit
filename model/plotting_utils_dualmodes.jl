# ==========================================================================
# Plotting for Dual-Mode LBA (mixture of two LBAs with shared weights)
# ==========================================================================

module PlottingUtilsDualModes

export mixture_rt_plot

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
                    val = pi_use*pdf(lba_fast,(choice=argmax(rewards),rt=t)) + (1-pi_use)*pdf(lba_slow,(choice=argmax(rewards),rt=t))
                    if isnan(val)||isinf(val) val=0.0 end
                    y_mix[j] += weight * val
                catch
                end
            end
        end
    end
    total_w = sum(values(reward_counts))
    if total_w > 0
        y_mix ./= total_w
    end

    p = plot(kde_grid, kde_dens, label="Observed", color=:blue, lw=2, xlabel="RT (s)", ylabel="Density", title="Cue $cue_condition", legend=:topright, guidefontsize=AXIS_FONT_SIZE, tickfontsize=AXIS_FONT_SIZE)
    plot!(p, t_grid, y_mix, label="Mixture RT", color=:red, lw=3)
    return p
end

end # module
