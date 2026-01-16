# ==========================================================================
# Model Utilities
# MIS-LBA mixture model logic and likelihood calculations
# ==========================================================================

using DataFrames
using Distributions
using SequentialSamplingModels

"""
    PreprocessedData

Pre-indexed data structure for ultra-fast likelihood computation.
Groups trials by unique reward configurations to eliminate redundant computations.
"""
struct PreprocessedData
    unique_rewards::Vector{Vector{Float64}}
    trial_groups::Vector{Vector{Int}}  # Indices of trials for each unique reward/condition config
    group_condition_types::Vector{Symbol}  # :single, :double, :all, or :mixed (if grouping not requested)
    rts::Vector{Float64}
    choices::Vector{Int}
    n_trials::Int
end

# ==========================================================================
# Dual-mode (two LBA) mixture with shared weights, flexible variation
# ==========================================================================
function _build_cond_types(df::DataFrame, provided)::Vector{Symbol}
    if !isnothing(provided)
        return provided
    end
    if !("CueCondition" in names(df))
        error("CueCondition column required for dual-mode mixture.")
    end
    return [if cc in (1,2,3,4) :single elseif cc in (5,6,7,8,9,10) :double else error("Unexpected CueCondition $cc") end for cc in df.CueCondition]
end

cue_key(flag, cond_type) = flag ? cond_type : :all
mode_key(flag, mode) = flag ? mode : :shared

function _fetch_param(params, dict, vary_mode::Bool, vary_cue::Bool, mode::Symbol, cue::Symbol)
    return params[dict[mode_key(vary_mode, mode)][cue_key(vary_cue, cue)]]
end

function _weight_components(params, layout, r_max, rewards)
    if layout.weighting_mode == :exponential
        w_slope = params[layout.idx_w[:w_slope]]
        r_use = isnothing(r_max) ? 4.0 : r_max
        w_slope_normalized = w_slope / r_use
        weights = exp.(w_slope_normalized .* rewards)
        return weights, nothing, 0.0
    else
        w2 = params[layout.idx_w[:w2]]
        w3 = params[layout.idx_w[:w3]]
        w4 = params[layout.idx_w[:w4]]
        if w2<=0 || w3<=0 || w4<=0
            return nothing, nothing, 0.0
        end
        weight_lookup = Dict{Float64, Float64}(1.0=>1.0, 2.0=>w2, 3.0=>w3, 4.0=>w4, 0.0=>1e-10)
        default_weight = weight_lookup[0.0]
        return nothing, weight_lookup, default_weight
    end
end

function _get_weights_for_rewards(weighting_mode, w_slope_normalized, weight_lookup, default_weight, rewards)
    if weighting_mode == :exponential
        return exp.(w_slope_normalized .* rewards)
    else
        return [get(weight_lookup, r, default_weight) for r in rewards]
    end
end

function mis_lba_dualmodes_loglike(params::Vector{<:Real}, df::DataFrame, layout; cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, r_max::Union{Nothing,Float64}=nothing)::Float64
    cond_types = _build_cond_types(df, cue_condition_types)

    weights_expo, weight_lookup, default_weight = _weight_components(params, layout, r_max, df.ParsedRewards[1])
    weighting_mode = layout.weighting_mode
    w_slope_normalized = weighting_mode == :exponential ? (params[layout.idx_w[:w_slope]] / (isnothing(r_max) ? 4.0 : r_max)) : 0.0

    drift_cache_fast = Dict{Tuple{Vararg{Float64}}, Any}()
    drift_cache_slow = Dict{Tuple{Vararg{Float64}}, Any}()

    total_neg_ll = 0.0
    for i in 1:nrow(df)
        cond = cond_types[i]
        C_fast = _fetch_param(params, layout.idx_C, layout.vary_C_by_mode, layout.vary_C_by_cue, :fast, cond)
        C_slow = _fetch_param(params, layout.idx_C, layout.vary_C_by_mode, layout.vary_C_by_cue, :slow, cond)
        k_fast = _fetch_param(params, layout.idx_k, layout.vary_k_by_mode, layout.vary_k_by_cue, :fast, cond)
        k_slow = _fetch_param(params, layout.idx_k, layout.vary_k_by_mode, layout.vary_k_by_cue, :slow, cond)
        A_use  = _fetch_param(params, layout.idx_A, layout.vary_A_by_mode, layout.vary_A_by_cue, :shared, cond)
        t0_use = _fetch_param(params, layout.idx_t0, layout.vary_t0_by_mode, layout.vary_t0_by_cue, :shared, cond)
        if C_fast<=0 || C_slow<=0 || k_fast<=0 || k_slow<=0 || A_use<=0 || t0_use<=0 || t0_use<0.01
            return Inf
        end

        rewards = df.ParsedRewards[i]; key = Tuple(rewards)
        if haskey(drift_cache_fast, key)
            drift_fast = drift_cache_fast[key]; drift_slow = drift_cache_slow[key]
        else
            weights = weighting_mode==:exponential ? exp.(w_slope_normalized .* rewards) : [get(weight_lookup, r, default_weight) for r in rewards]
            rel_weights = weights ./ sum(weights)
            drift_fast = C_fast .* rel_weights
            drift_slow = C_slow .* rel_weights
            drift_cache_fast[key] = drift_fast; drift_cache_slow[key] = drift_slow
        end

        lba_fast = LBA(ν=drift_fast, A=A_use, k=k_fast, τ=t0_use)
        lba_slow = LBA(ν=drift_slow, A=A_use, k=k_slow, τ=t0_use)
        rt = df.CleanRT[i]; choice = df.Choice[i]
        lik_fast = (rt>t0_use) ? try
                val = pdf(lba_fast,(choice=choice, rt=rt)); (isnan(val)||isinf(val)) ? 1e-10 : val
            catch; 1e-10 end : 1e-10
        lik_slow = (rt>t0_use) ? try
                val = pdf(lba_slow,(choice=choice, rt=rt)); (isnan(val)||isinf(val)) ? 1e-10 : val
            catch; 1e-10 end : 1e-10

        pi_key = layout.vary_pi_by_cue ? cond : :all
        pi_fast = params[layout.idx_pi[pi_key]]
        if pi_fast<0 || pi_fast>1 return Inf end

        lik = pi_fast*lik_fast + (1-pi_fast)*lik_slow

        if layout.use_contaminant && layout.estimate_contaminant
            mkeyα = layout.vary_contam_alpha_by_mode ? :fast : :shared
            ckeyα = layout.vary_contam_alpha_by_cue ? cond : :all
            alpha = params[layout.idx_contam_alpha[mkeyα][ckeyα]]
            mkeyr = layout.vary_contam_rt_by_mode ? :fast : :shared
            ckeyr = layout.vary_contam_rt_by_cue ? cond : :all
            rtmax = params[layout.idx_contam_rt[mkeyr][ckeyr]]
            if alpha<0 || alpha>0.5 || rtmax<=0 return Inf end
            lik = (1-alpha)*lik + alpha*(1/rtmax)
        elseif layout.use_contaminant
            lik = (1-layout.contam_alpha_fixed)*lik + layout.contam_alpha_fixed*(1/layout.contam_rt_fixed)
        end

        if lik <= 1e-20 lik = 1e-20 end
        total_neg_ll -= log(lik)
    end
    return total_neg_ll
end

function mis_lba_dualmodes_loglike(params::Vector{<:Real}, preprocessed::PreprocessedData, layout; cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, r_max::Union{Nothing,Float64}=nothing)::Float64
    cond_types = isnothing(cue_condition_types) ? preprocessed.group_condition_types : cue_condition_types
    weighting_mode = layout.weighting_mode

    w_slope_normalized = 0.0
    weight_lookup = nothing
    default_weight = 1e-10
    if weighting_mode == :exponential
        r_use = isnothing(r_max) ? 4.0 : r_max
        w_slope_normalized = params[layout.idx_w[:w_slope]] / r_use
    else
        w2 = params[layout.idx_w[:w2]]
        w3 = params[layout.idx_w[:w3]]
        w4 = params[layout.idx_w[:w4]]
        if w2<=0 || w3<=0 || w4<=0
            return Inf
        end
        weight_lookup = Dict{Float64,Float64}(1.0=>1.0, 2.0=>w2, 3.0=>w3, 4.0=>w4, 0.0=>1e-10)
        default_weight = weight_lookup[0.0]
    end

    total_neg_ll = 0.0
    for (idx, rewards) in enumerate(preprocessed.unique_rewards)
        trial_indices = preprocessed.trial_groups[idx]
        cond = cond_types[idx]

        C_fast = params[layout.idx_C[mode_key(layout.vary_C_by_mode,:fast)][cue_key(layout.vary_C_by_cue,cond)]]
        C_slow = params[layout.idx_C[mode_key(layout.vary_C_by_mode,:slow)][cue_key(layout.vary_C_by_cue,cond)]]
        k_fast = params[layout.idx_k[mode_key(layout.vary_k_by_mode,:fast)][cue_key(layout.vary_k_by_cue,cond)]]
        k_slow = params[layout.idx_k[mode_key(layout.vary_k_by_mode,:slow)][cue_key(layout.vary_k_by_cue,cond)]]
        A_use  = params[layout.idx_A[mode_key(layout.vary_A_by_mode,:shared)][cue_key(layout.vary_A_by_cue,cond)]]
        t0_use = params[layout.idx_t0[mode_key(layout.vary_t0_by_mode,:shared)][cue_key(layout.vary_t0_by_cue,cond)]]
        if C_fast<=0 || C_slow<=0 || k_fast<=0 || k_slow<=0 || A_use<=0 || t0_use<=0 || t0_use<0.01
            return Inf
        end

        weights = weighting_mode==:exponential ? exp.(w_slope_normalized .* rewards) : [get(weight_lookup, r, default_weight) for r in rewards]
        rel_weights = weights ./ sum(weights)
        drift_fast = C_fast .* rel_weights
        drift_slow = C_slow .* rel_weights

        lba_fast = LBA(ν=drift_fast, A=A_use, k=k_fast, τ=t0_use)
        lba_slow = LBA(ν=drift_slow, A=A_use, k=k_slow, τ=t0_use)

        pi_key = layout.vary_pi_by_cue ? cond : :all
        pi_fast = params[layout.idx_pi[pi_key]]
        if pi_fast<0 || pi_fast>1
            return Inf
        end

        contam_alpha = layout.contam_alpha_fixed
        contam_rt = layout.contam_rt_fixed
        if layout.use_contaminant && layout.estimate_contaminant
            contam_alpha = params[layout.idx_contam_alpha[mode_key(layout.vary_contam_alpha_by_mode,:shared)][cue_key(layout.vary_contam_alpha_by_cue,cond)]]
            contam_rt = params[layout.idx_contam_rt[mode_key(layout.vary_contam_rt_by_mode,:shared)][cue_key(layout.vary_contam_rt_by_cue,cond)]]
            if contam_alpha<0 || contam_alpha>0.5 || contam_rt<=0
                return Inf
            end
        end

        for trial_idx in trial_indices
            rt = preprocessed.rts[trial_idx]
            choice = preprocessed.choices[trial_idx]
            lik_fast = (rt>t0_use) ? try
                    val = pdf(lba_fast,(choice=choice, rt=rt)); (isnan(val)||isinf(val)) ? 1e-10 : val
                catch; 1e-10 end : 1e-10
            lik_slow = (rt>t0_use) ? try
                    val = pdf(lba_slow,(choice=choice, rt=rt)); (isnan(val)||isinf(val)) ? 1e-10 : val
                catch; 1e-10 end : 1e-10
            lik = pi_fast*lik_fast + (1-pi_fast)*lik_slow
            if layout.use_contaminant
                lik = (1-contam_alpha)*lik + contam_alpha*(1/contam_rt)
            end
            if lik <= 1e-20 lik = 1e-20 end
            total_neg_ll -= log(lik)
        end
    end
    return total_neg_ll
end
"""
    preprocess_data_for_fitting(df::DataFrame)::PreprocessedData

Preprocess data to group trials by unique reward configurations.
This dramatically speeds up likelihood computation by computing drift rates only once per unique configuration.
"""
function preprocess_data_for_fitting(df::DataFrame; cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, group_by_condition::Bool=false)::PreprocessedData
    # Use flexible keys so identical reward sets hash by value (and cue-type when requested)
    reward_to_idx = Dict{Any, Int}()
    unique_rewards = Vector{Vector{Float64}}()
    trial_groups = Vector{Vector{Int}}()
    group_condition_types = Symbol[]

    cue_condition_types_use = cue_condition_types
    if group_by_condition && isnothing(cue_condition_types_use)
        error("cue_condition_types must be provided when group_by_condition=true")
    end
    if isnothing(cue_condition_types_use)
        cue_condition_types_use = fill(:all, nrow(df))
    end
    @assert length(cue_condition_types_use) == nrow(df) "cue_condition_types length must match number of rows"

    for (trial_idx, rewards) in enumerate(df.ParsedRewards)
        cond_type = cue_condition_types_use[trial_idx]
        key = group_by_condition ? (cond_type, Tuple(rewards)) : Tuple(rewards)
        group_idx = get(reward_to_idx, key, 0)
        if group_idx == 0
            push!(unique_rewards, rewards)
            push!(trial_groups, Int[])
            push!(group_condition_types, cond_type)
            group_idx = length(unique_rewards)
            reward_to_idx[key] = group_idx
        elseif group_condition_types[group_idx] != cond_type
            # Mark mixed groups so downstream code can assert if grouping was forgotten
            group_condition_types[group_idx] = :mixed
        end
        push!(trial_groups[group_idx], trial_idx)
    end

    n_unique = length(unique_rewards)

    # Extract RT and choice arrays for fast access
    rts = Vector{Float64}(df.CleanRT)
    choices = Vector{Int}(df.Choice)

    println("Preprocessed data: $(n_unique) unique reward configurations across $(nrow(df)) trials")
    println("Cache efficiency: $(round((1 - n_unique/nrow(df)) * 100, digits=1))%")

    return PreprocessedData(unique_rewards, trial_groups, group_condition_types, rts, choices, nrow(df))
end

"""
    mis_lba_mixture_loglike(params::Vector{Float64}, df::DataFrame)::Float64

    Computes the negative log-likelihood for the MIS-LBA mixture model.

    Parameters:
    - C: Capacity parameter (drift rate scaling)
    - w_slope: Reward weight slope
    - A: Maximum start point variability in LBA
    - k: Threshold gap (b - A)
    - t0: Non-decision time
    - p_exp: Probability of express responses
    - mu_exp: Mean of express response distribution
    - sig_exp: Standard deviation of express response distribution

    Returns negative log-likelihood (to be minimized).
"""
function mis_lba_mixture_loglike(params::Vector{<:Real}, df::DataFrame)::Float64
    # Unpack parameters
    C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp = params

    # Constraints (strict check to prevent integrator errors)
    if C<=0 || w_slope<0 || A<=0 || k<=0 || t0<=0 || t0 < 0.01 ||
       p_exp<0 || p_exp>0.99 || sig_exp<=0
        return Inf
    end

    total_neg_ll = 0.0
    dist_express = Normal(mu_exp, sig_exp)

    # Accumulate log-likelihood across all trials
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]

        # --- MIS THEORY ---
        # Weight = 1 + w * Reward
        # We add 1.0 to base weight to ensure drift > 0 even for 0 reward
        weights = 1.0 .+ (w_slope .* rewards)
        rel_weights = weights ./ sum(weights)
        drift_rates = C .* rel_weights

        # --- LBA ---
        # Parameters: ν=drift, A=max start, k=b-A, τ=non-decision
        # Note: A must be strictly positive for LBA
        lba = LBA(ν=drift_rates, A=A, k=k, τ=t0)

        lik_reg = 0.0
        # LBA density is 0 if RT < t0. We handle this gracefully.
        if rt > t0
            try
                # pdf(d, (choice, rt))
                lik_reg = pdf(lba, (choice=choice, rt=rt))
                if isnan(lik_reg) || isinf(lik_reg) lik_reg = 1e-10 end
            catch
                lik_reg = 1e-10
            end
        else
            lik_reg = 1e-10
        end

        # --- MIXTURE ---
        lik_exp = pdf(dist_express, rt)
        lik_tot = (p_exp * lik_exp) + ((1-p_exp) * lik_reg)

        if lik_tot <= 1e-20 lik_tot = 1e-20 end
        total_neg_ll -= log(lik_tot)
    end

    return total_neg_ll
end

"""
    mis_lba_dual_mixture_loglike(params::Vector{Float64}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing)::Float64

    Computes the negative log-likelihood for a dual-LBA mixture model.
    This model uses TWO LBA components with different parameters to capture
    bimodality, rather than LBA + express responses.

    Parameters:
    - C: Capacity parameter (drift rate scaling)
    - w_slope: Reward weight slope
    - A1: Maximum start point variability for LBA component 1 (fast mode)
    - k1: Threshold gap for LBA component 1 (b - A)
    - t0_1: Non-decision time for LBA component 1
    - A2: Maximum start point variability for LBA component 2 (slow mode)
    - k2: Threshold gap for LBA component 2 (b - A)
    - t0_2: Non-decision time for LBA component 2
    - p_mix: Probability of using LBA component 1 (vs component 2)
    - r_max: Optional maximum reward value across entire experiment.
             If not provided, computed from df (for backward compatibility).

    Returns negative log-likelihood (to be minimized).
"""
function mis_lba_dual_mixture_loglike(params::Vector{<:Real}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing)::Float64
    # Unpack parameters
    C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix = params

    # Constraints (strict check to prevent integrator errors)
    if C<=0 || w_slope<0 || A1<=0 || k1<=0 || t0_1<=0 || t0_1 < 0.01 ||
       A2<=0 || k2<=0 || t0_2<=0 || t0_2 < 0.01 || p_mix<0 || p_mix>0.99
        return Inf
    end

    total_neg_ll = 0.0

    # Compute r_max: use provided value, or compute from dataset if not provided
    if isnothing(r_max)
        # Fallback: compute from current dataset (for backward compatibility)
        r_max = 0.0
        for rewards in df.ParsedRewards
            if !isempty(rewards)
                r_max = max(r_max, maximum(rewards))
                @assert r_max == 4 "rmax calculated incorrectly"
            end
        end
    end
    # Avoid division by zero if all rewards are 0
    if r_max <= 0.0
        error("r_max should not smaller than 0")
        r_max = 1.0
    end

    # Precompute constant factor to avoid repeated divisions
    w_slope_normalized = w_slope / r_max

    # Cache for drift rates based on reward configurations (and cue type if needed)
    drift_cache = Dict{Any, Any}()

    # Accumulate log-likelihood across all trials
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]
        key = Tuple(rewards)

        # --- MIS THEORY ---
        # Check if we've already computed drift rates for this reward configuration
        if haskey(drift_cache, key)
            drift_rates = drift_cache[key]
        else
            # Weight = exp(θ * r / r_max) as per paper
            # Exponential weighting allows winner-take-all behavior
            weights = exp.(w_slope_normalized .* rewards)
            rel_weights = weights ./ sum(weights)
            drift_rates = C .* rel_weights
            # Cache the result
            drift_cache[key] = drift_rates
        end

        # --- LBA COMPONENT 1 (Fast Mode) ---
        lba1 = LBA(ν=drift_rates, A=A1, k=k1, τ=t0_1)
        lik1 = 0.0
        if rt > t0_1
            try
                lik1 = pdf(lba1, (choice=choice, rt=rt))
                if isnan(lik1) || isinf(lik1) lik1 = 1e-10 end
            catch
                lik1 = 1e-10
            end
        else
            lik1 = 1e-10
        end

        # --- LBA COMPONENT 2 (Slow Mode) ---
        lba2 = LBA(ν=drift_rates, A=A2, k=k2, τ=t0_2)
        lik2 = 0.0
        if rt > t0_2
            try
                lik2 = pdf(lba2, (choice=choice, rt=rt))
                if isnan(lik2) || isinf(lik2) lik2 = 1e-10 end
            catch
                lik2 = 1e-10
            end
        else
            lik2 = 1e-10
        end

        # --- DUAL MIXTURE ---
        lik_tot = (p_mix * lik1) + ((1-p_mix) * lik2)

        if lik_tot <= 1e-20 lik_tot = 1e-20 end
        total_neg_ll -= log(lik_tot)
    end

    return total_neg_ll
end

"""
    mis_lba_single_loglike(params::Vector{Float64}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing)::Float64

    Computes the negative log-likelihood for a single LBA model (no mixture).
    This model uses ONE LBA component to fit the entire RT distribution.

    Parameters:
    - C: Capacity parameter (drift rate scaling)
    - w_slope: Reward weight slope
    - A: Maximum start point variability
    - k: Threshold gap (b - A)
    - t0: Non-decision time
    - r_max: Optional maximum reward value across entire experiment.
             If not provided, computed from df (for backward compatibility).

    Returns negative log-likelihood (to be minimized).
"""
function mis_lba_single_loglike(params::Vector{<:Real}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing)::Float64
    # Unpack parameters
    C, w_slope, A, k, t0 = params

    # Constraints (strict check to prevent integrator errors)
    if C<=0 || w_slope<0 || A<=0 || k<=0 || t0<=0 || t0 < 0.01
        return Inf
    end

    total_neg_ll = 0.0

    # Compute r_max: use provided value, or compute from dataset if not provided
    if isnothing(r_max)
        # Fallback: compute from current dataset (for backward compatibility)
        r_max = 0.0
        for rewards in df.ParsedRewards
            if !isempty(rewards)
                r_max = max(r_max, maximum(rewards))
                @assert r_max == 4 "rmax calculated incorrectly"
            end
        end
    end
    # Avoid division by zero if all rewards are 0
    if r_max <= 0.0
        error("r_max should not smaller than 0")
        r_max = 1.0
    end

    # Precompute constant factor to avoid repeated divisions
    w_slope_normalized = w_slope / r_max

    # Cache for drift rates based on reward configurations
    drift_cache = Dict{Tuple{Vararg{Float64}}, Any}()

    # Accumulate log-likelihood across all trials
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]
        key = Tuple(rewards)

        # --- MIS THEORY ---
        # Check if we've already computed drift rates for this reward configuration
        if haskey(drift_cache, key)
            drift_rates = drift_cache[key]
        else
            # Weight = exp(θ * r / r_max) as per paper
            # Exponential weighting allows winner-take-all behavior
            weights = exp.(w_slope_normalized .* rewards)
            rel_weights = weights ./ sum(weights)
            drift_rates = C .* rel_weights
            # Cache the result
            drift_cache[key] = drift_rates
        end

        # --- SINGLE LBA ---
        lba = LBA(ν=drift_rates, A=A, k=k, τ=t0)
        lik = 0.0
        if rt > t0
            try
                lik = pdf(lba, (choice=choice, rt=rt))
                if isnan(lik) || isinf(lik) lik = 1e-10 end
            catch
                lik = 1e-10
            end
        else
            lik = 1e-10
        end

        if lik <= 1e-20 lik = 1e-20 end
        total_neg_ll -= log(lik)
    end

    return total_neg_ll
end

"""
    mis_lba_allconditions_loglike(params::Vector{Float64}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, use_contaminant::Bool=false, estimate_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64

    Computes the negative log-likelihood for a single LBA model fitted to ALL conditions at once.
    Allows optional variation of C and/or t0 by single vs double cue conditions.
"""
function mis_lba_allconditions_loglike(params::Vector{<:Real}, df::DataFrame; layout=nothing, r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, use_contaminant::Bool=false, estimate_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64
    if layout !== nothing
        weighting_mode = layout.weighting_mode
        vary_C_by_cue_type = layout.vary_C_by_cue
        vary_t0_by_cue_type = layout.vary_t0_by_cue
        vary_k_by_cue_type = layout.vary_k_by_cue
        use_contaminant = layout.use_contaminant
        estimate_contaminant = layout.estimate_contaminant
    end
    vary_contam_by_cue = layout === nothing ? false : layout.vary_contam_by_cue

    # Derive cue-condition types if needed
    cond_types = nothing
    if vary_C_by_cue_type || vary_t0_by_cue_type || vary_k_by_cue_type || (use_contaminant && (estimate_contaminant || vary_contam_by_cue))
        if isnothing(cue_condition_types)
            if !("CueCondition" in names(df))
                error("CueCondition column required to vary parameters by cue condition.")
            end
            single_set = Set([1, 2, 3, 4])
            double_set = Set([5, 6, 7, 8, 9, 10])
            cond_types = Vector{Symbol}(undef, nrow(df))
            for (i, cc) in enumerate(df.CueCondition)
                @assert !ismissing(cc) "CueCondition cannot be missing when varying by cue type."
                if cc in single_set
                    cond_types[i] = :single
                elseif cc in double_set
                    cond_types[i] = :double
                else
                    error("Unexpected CueCondition value $cc. Expected 1-10.")
                end
            end
        else
            cond_types = cue_condition_types
            @assert length(cond_types) == nrow(df) "cue_condition_types length must match number of rows"
            @assert all(ct -> ct in (:single, :double), cond_types) "cue_condition_types must be :single or :double"
        end
    end

    # Helper to select cue-specific parameters
    get_by_cue(map, ct) = haskey(map, ct) ? params[map[ct]] : params[map[:all]]

    # Unpack parameters and constraints based on weighting_mode
    w_slope = 0.0
    w2 = w3 = w4 = 0.0
    Ge = Gi = 1.0
    if layout === nothing
        # Fallback to positional parsing for legacy callers
        p_idx = 1
        C_single = params[p_idx]; p_idx += 1
        C_double = vary_C_by_cue_type ? params[p_idx] : C_single
        p_idx += vary_C_by_cue_type ? 1 : 0
        w_slope = 0.0
        w2 = w3 = w4 = 0.0
        Ge = Gi = 1.0
        k_single = 0.0
        k_double = 0.0
        if weighting_mode == :exponential
            w_slope = params[p_idx]; p_idx += 1
            A = params[p_idx]; p_idx += 1
            k_single = params[p_idx]; p_idx += 1
            k_double = vary_k_by_cue_type ? params[p_idx] : k_single
            p_idx += vary_k_by_cue_type ? 1 : 0
            t0_single = params[p_idx]; p_idx += 1
            t0_double = vary_t0_by_cue_type ? params[p_idx] : t0_single
            if C_single<=0 || C_double<=0 || w_slope<0 || A<=0 || k_single<=0 || k_double<=0 || t0_single<=0 || t0_single < 0.01 || t0_double<=0 || t0_double < 0.01
                return Inf
            end
        elseif weighting_mode == :free
            w2 = params[p_idx]; p_idx += 1
            w3 = params[p_idx]; p_idx += 1
            w4 = params[p_idx]; p_idx += 1
            A = params[p_idx]; p_idx += 1
            k_single = params[p_idx]; p_idx += 1
            k_double = vary_k_by_cue_type ? params[p_idx] : k_single
            p_idx += vary_k_by_cue_type ? 1 : 0
            t0_single = params[p_idx]; p_idx += 1
            t0_double = vary_t0_by_cue_type ? params[p_idx] : t0_single
            if C_single<=0 || C_double<=0 || w2<=0 || w3<=0 || w4<=0 || A<=0 || k_single<=0 || k_double<=0 || t0_single<=0 || t0_single < 0.01 || t0_double<=0 || t0_double < 0.01
                return Inf
            end
        elseif weighting_mode == :excitation_inhibition
            w_slope = params[p_idx]; p_idx += 1
            Ge = params[p_idx]; p_idx += 1
            Gi = params[p_idx]; p_idx += 1
            A = params[p_idx]; p_idx += 1
            k_single = params[p_idx]; p_idx += 1
            k_double = vary_k_by_cue_type ? params[p_idx] : k_single
            p_idx += vary_k_by_cue_type ? 1 : 0
            t0_single = params[p_idx]; p_idx += 1
            t0_double = vary_t0_by_cue_type ? params[p_idx] : t0_single
            if C_single<=0 || C_double<=0 || w_slope<0 || Ge<=0 || Gi<=0 || A<=0 || k_single<=0 || k_double<=0 || t0_single<=0 || t0_single < 0.01 || t0_double<=0 || t0_double < 0.01
                return Inf
            end
        else
            error("Unknown weighting_mode: $weighting_mode. Use :exponential, :free, or :excitation_inhibition.")
        end

        # Optional contaminant parameters from search
        contam_alpha_use = contaminant_alpha
        contam_rt_max_use = contaminant_rt_max
        if use_contaminant && estimate_contaminant
            contam_alpha_use = params[p_idx]; p_idx += 1
            contam_rt_max_use = params[p_idx]; p_idx += 1
            if contam_alpha_use < 0.0 || contam_alpha_use > 0.5 || contam_rt_max_use <= 0.1
                return Inf
            end
        end
    else
        if weighting_mode == :exponential
            w_slope = params[layout.idx_w[:w_slope]]
        elseif weighting_mode == :free
            w2 = params[layout.idx_w[:w2]]
            w3 = params[layout.idx_w[:w3]]
            w4 = params[layout.idx_w[:w4]]
        elseif weighting_mode == :excitation_inhibition
            w_slope = params[layout.idx_w[:w_slope]]
            Ge = params[layout.idx_w[:Ge]]
            Gi = params[layout.idx_w[:Gi]]
        end
        A = params[layout.idx_A]

        # C/k/t0 retrieval
        getC(ct) = haskey(layout.idx_C, ct) ? params[layout.idx_C[ct]] : params[layout.idx_C[:all]]
        getk(ct) = haskey(layout.idx_k, ct) ? params[layout.idx_k[ct]] : params[layout.idx_k[:all]]
        gett0(ct) = haskey(layout.idx_t0, ct) ? params[layout.idx_t0[ct]] : params[layout.idx_t0[:all]]

        C_single = getC(:single)
        C_double = haskey(layout.idx_C, :double) ? getC(:double) : C_single
        k_single = getk(:single)
        k_double = haskey(layout.idx_k, :double) ? getk(:double) : k_single
        t0_single = gett0(:single)
        t0_double = haskey(layout.idx_t0, :double) ? gett0(:double) : t0_single

        if any(x -> x <= 0, (C_single, C_double, A, k_single, k_double, t0_single, t0_double)) || t0_single < 0.01 || t0_double < 0.01
            return Inf
        end
        
        # Validate Ge and Gi for excitation_inhibition mode
        if weighting_mode == :excitation_inhibition
            if Ge <= 0 || Gi <= 0
                return Inf
            end
        end

        contam_alpha_use = layout.contam_alpha_fixed
        contam_rt_max_use = layout.contam_rt_fixed
        if use_contaminant && estimate_contaminant
            if !isempty(layout.idx_contam_alpha)
                contam_alpha_use = get_by_cue(layout.idx_contam_alpha, :single)
            end
            if !isempty(layout.idx_contam_rt)
                contam_rt_max_use = get_by_cue(layout.idx_contam_rt, :single)
            end
        end
    end

    total_neg_ll = 0.0

    # Compute r_max for exponential and excitation_inhibition modes
    if weighting_mode == :exponential || weighting_mode == :excitation_inhibition
        if isnothing(r_max)
            r_max = 0.0
            for rewards in df.ParsedRewards
                if !isempty(rewards)
                    r_max = max(r_max, maximum(rewards))
                    @assert r_max == 4 "rmax calculated incorrectly"
                end
            end
        end
        if r_max <= 0.0
            error("r_max should not smaller than 0")
            r_max = 1.0
        end
    else #free mode, r_max is not used
        r_max = isnothing(r_max) ? 4.0 : r_max
    end

    w_slope_normalized = (weighting_mode == :exponential || weighting_mode == :excitation_inhibition) ? (w_slope / r_max) : 0.0
    weight_lookup = nothing #weight_lookup is only used for free mode
    if weighting_mode == :free
        val_type = typeof(C_single)
        weight_lookup = Dict{Float64, val_type}(
            1.0 => one(val_type),
            2.0 => w2,
            3.0 => w3,
            4.0 => w4,
            0.0 => convert(val_type, 1e-10)
        )
    end
    default_weight = weighting_mode == :free ? weight_lookup[0.0] : 1e-10

    # Cache for drift rates based on reward configurations
    drift_cache = Dict{Tuple{Vararg{Float64}}, Any}()

    # Accumulate log-likelihood across ALL trials from ALL conditions
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]
        cond_type = isnothing(cond_types) ? :single : cond_types[i]
        key = vary_C_by_cue_type ? (cond_type, Tuple(rewards)) : Tuple(rewards)

        # --- MIS THEORY ---
        if haskey(drift_cache, key)
            drift_rates = drift_cache[key]
        else
            # Calculate baseline weights
            if weighting_mode == :exponential
                weights = exp.(w_slope_normalized .* rewards)
            elseif weighting_mode == :free
                weights = [get(weight_lookup, r, default_weight) for r in rewards]
            elseif weighting_mode == :excitation_inhibition
                # For excitation_inhibition mode, use exponential as baseline
                # Compute baseline weights first (no Ge/Gi applied yet)
                weights = exp.(w_slope_normalized .* rewards)
            else
                error("Unknown weighting_mode: $weighting_mode")
            end
            
            rel_weights = weights ./ sum(weights)
            C_use = cond_type == :double ? C_double : C_single
            drift_rates = C_use .* rel_weights
            
            # Apply excitation/inhibition gains DIRECTLY to drift rates for 2-cue conditions
            # This ensures the control mechanism has a direct, un-diluted effect on accumulation rates
            # (Decision Control Theory: reactive control modulates drift rates directly)
            if weighting_mode == :excitation_inhibition && cond_type == :double
                # Find highest reward cue (target)
                max_reward_val = maximum(rewards)
                max_reward_idx = findfirst(r -> r == max_reward_val, rewards)
                
                # Apply Ge (excitatory gain) to highest reward cue's drift rate
                if !isnothing(max_reward_idx)
                    drift_rates[max_reward_idx] *= Ge
                end
                
                # Apply Gi (inhibitory gain) to all other cues with reward > 0 (distractors)
                for (idx, r) in enumerate(rewards)
                    if r > 0 && r < max_reward_val
                        drift_rates[idx] *= Gi
                    end
                    # Non-cued items (reward 0) keep baseline drift rate (already set above)
                end
            end
            
            drift_cache[key] = drift_rates
        end

        # --- SINGLE LBA ---
        k_use = cond_type == :double ? k_double : k_single
        t0_use = cond_type == :double ? t0_double : t0_single
        lba = LBA(ν=drift_rates, A=A, k=k_use, τ=t0_use)
        lik = 0.0
        if rt > t0_use
            try
                lik = pdf(lba, (choice=choice, rt=rt))
                if isnan(lik) || isinf(lik) lik = 1e-10 end
            catch
                lik = 1e-10
            end
        else
            lik = 1e-10
        end

        if use_contaminant
            alpha_use = layout === nothing ? contam_alpha_use : (estimate_contaminant && vary_contam_by_cue && !isnothing(cond_types) && haskey(layout.idx_contam_alpha, cond_type) ? params[layout.idx_contam_alpha[cond_type]] : contam_alpha_use)
            rtmax_use = layout === nothing ? contaminant_rt_max : (estimate_contaminant && vary_contam_by_cue && !isnothing(cond_types) && haskey(layout.idx_contam_rt, cond_type) ? params[layout.idx_contam_rt[cond_type]] : contam_rt_max_use)
            uniform_density = 1.0 / rtmax_use
            lik = (1 - alpha_use) * lik + alpha_use * uniform_density
        end

        if lik <= 1e-20 lik = 1e-20 end
        total_neg_ll -= log(lik)
    end

    return total_neg_ll
end

"""
    mis_lba_allconditions_loglike(params::Vector{Float64}, preprocessed::PreprocessedData; r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, use_contaminant::Bool=false, estimate_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64

Ultra-fast likelihood computation using preprocessed data (method overload).
Computes drift rates only once per unique reward configuration.
This version is 3-5x faster than the DataFrame version.
"""
function mis_lba_allconditions_loglike(params::Vector{<:Real}, preprocessed::PreprocessedData; layout=nothing, r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, use_contaminant::Bool=false, estimate_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64
    if layout !== nothing
        weighting_mode = layout.weighting_mode
        vary_C_by_cue_type = layout.vary_C_by_cue
        vary_t0_by_cue_type = layout.vary_t0_by_cue
        vary_k_by_cue_type = layout.vary_k_by_cue
        use_contaminant = layout.use_contaminant
        estimate_contaminant = layout.estimate_contaminant
    end
    vary_contam_by_cue = layout === nothing ? false : layout.vary_contam_by_cue

    # Unpack parameters and constraints based on weighting_mode
    w_slope = 0.0
    w2 = w3 = w4 = 0.0
    if layout === nothing
        p_idx = 1
        C_single = params[p_idx]; p_idx += 1
        C_double = vary_C_by_cue_type ? params[p_idx] : C_single
        p_idx += vary_C_by_cue_type ? 1 : 0
        k_single = 0.0
        k_double = 0.0
        if weighting_mode == :exponential
            w_slope = params[p_idx]; p_idx += 1
            A = params[p_idx]; p_idx += 1
            k_single = params[p_idx]; p_idx += 1
            k_double = vary_k_by_cue_type ? params[p_idx] : k_single
            p_idx += vary_k_by_cue_type ? 1 : 0
            t0_single = params[p_idx]; p_idx += 1
            t0_double = vary_t0_by_cue_type ? params[p_idx] : t0_single
            if C_single<=0 || C_double<=0 || w_slope<0 || A<=0 || k_single<=0 || k_double<=0 || t0_single<=0 || t0_single < 0.01 || t0_double<=0 || t0_double < 0.01
                return Inf
            end
        elseif weighting_mode == :free
            w2 = params[p_idx]; p_idx += 1
            w3 = params[p_idx]; p_idx += 1
            w4 = params[p_idx]; p_idx += 1
            A = params[p_idx]; p_idx += 1
            k_single = params[p_idx]; p_idx += 1
            k_double = vary_k_by_cue_type ? params[p_idx] : k_single
            p_idx += vary_k_by_cue_type ? 1 : 0
            t0_single = params[p_idx]; p_idx += 1
            t0_double = vary_t0_by_cue_type ? params[p_idx] : t0_single
            if C_single<=0 || C_double<=0 || w2<=0 || w3<=0 || w4<=0 || A<=0 || k_single<=0 || k_double<=0 || t0_single<=0 || t0_single < 0.01 || t0_double<=0 || t0_double < 0.01
                return Inf
            end
        elseif weighting_mode == :excitation_inhibition
            w_slope = params[p_idx]; p_idx += 1
            Ge = params[p_idx]; p_idx += 1
            Gi = params[p_idx]; p_idx += 1
            A = params[p_idx]; p_idx += 1
            k_single = params[p_idx]; p_idx += 1
            k_double = vary_k_by_cue_type ? params[p_idx] : k_single
            p_idx += vary_k_by_cue_type ? 1 : 0
            t0_single = params[p_idx]; p_idx += 1
            t0_double = vary_t0_by_cue_type ? params[p_idx] : t0_single
            if C_single<=0 || C_double<=0 || w_slope<0 || Ge<=0 || Gi<=0 || A<=0 || k_single<=0 || k_double<=0 || t0_single<=0 || t0_single < 0.01 || t0_double<=0 || t0_double < 0.01
                return Inf
            end
        else
            error("Unknown weighting_mode: $weighting_mode. Use :exponential, :free, or :excitation_inhibition.")
        end

        contam_alpha_use = contaminant_alpha
        contam_rt_max_use = contaminant_rt_max
        if use_contaminant && estimate_contaminant
            contam_alpha_use = params[p_idx]; p_idx += 1
            contam_rt_max_use = params[p_idx]; p_idx += 1
            if contam_alpha_use < 0.0 || contam_alpha_use > 0.5 || contam_rt_max_use <= 0.1
                return Inf
            end
        end
    else
        if weighting_mode == :exponential
            w_slope = params[layout.idx_w[:w_slope]]
        elseif weighting_mode == :free
            w2 = params[layout.idx_w[:w2]]
            w3 = params[layout.idx_w[:w3]]
            w4 = params[layout.idx_w[:w4]]
        elseif weighting_mode == :excitation_inhibition
            w_slope = params[layout.idx_w[:w_slope]]
            Ge = params[layout.idx_w[:Ge]]
            Gi = params[layout.idx_w[:Gi]]
        end
        A = params[layout.idx_A]

        getC(ct) = haskey(layout.idx_C, ct) ? params[layout.idx_C[ct]] : params[layout.idx_C[:all]]
        getk(ct) = haskey(layout.idx_k, ct) ? params[layout.idx_k[ct]] : params[layout.idx_k[:all]]
        gett0(ct) = haskey(layout.idx_t0, ct) ? params[layout.idx_t0[ct]] : params[layout.idx_t0[:all]]

        C_single = getC(:single)
        C_double = haskey(layout.idx_C, :double) ? getC(:double) : C_single
        k_single = getk(:single)
        k_double = haskey(layout.idx_k, :double) ? getk(:double) : k_single
        t0_single = gett0(:single)
        t0_double = haskey(layout.idx_t0, :double) ? gett0(:double) : t0_single

        if any(x -> x <= 0, (C_single, C_double, A, k_single, k_double, t0_single, t0_double)) || t0_single < 0.01 || t0_double < 0.01
            return Inf
        end
        
        # Validate Ge and Gi for excitation_inhibition mode
        if weighting_mode == :excitation_inhibition
            if Ge <= 0 || Gi <= 0
                return Inf
            end
        end

        contam_alpha_use = layout.contam_alpha_fixed
        contam_rt_max_use = layout.contam_rt_fixed
        if use_contaminant && estimate_contaminant
            if !isempty(layout.idx_contam_alpha)
                cue_key = haskey(layout.idx_contam_alpha, :all) ? :all : :single
                contam_alpha_use = params[layout.idx_contam_alpha[cue_key]]
            end
            if !isempty(layout.idx_contam_rt)
                cue_key = haskey(layout.idx_contam_rt, :all) ? :all : :single
                contam_rt_max_use = params[layout.idx_contam_rt[cue_key]]
            end
        end
    end

    # r_max is used for exponential and excitation_inhibition modes
    if weighting_mode == :exponential || weighting_mode == :excitation_inhibition
        if isnothing(r_max)
            r_max = 4.0
        end
    else
        r_max = isnothing(r_max) ? 1.0 : r_max
    end

    total_neg_ll = 0.0
    w_slope_normalized = (weighting_mode == :exponential || weighting_mode == :excitation_inhibition) ? (w_slope / r_max) : 0.0
    weight_lookup = nothing
    if weighting_mode == :free
        val_type = typeof(C_single)
        weight_lookup = Dict{Float64, val_type}(
            1.0 => one(val_type),
            2.0 => w2,
            3.0 => w3,
            4.0 => w4,
            0.0 => convert(val_type, 1e-10)
        )
    end
    default_weight = weighting_mode == :free ? weight_lookup[0.0] : 1e-10

    # Process each unique reward/condition configuration
    @inbounds for (idx, rewards) in enumerate(preprocessed.unique_rewards)
        trial_indices = preprocessed.trial_groups[idx]
        cond_type = preprocessed.group_condition_types[idx]
        if (vary_C_by_cue_type || vary_t0_by_cue_type || vary_k_by_cue_type || vary_contam_by_cue) && cond_type in (:all, :mixed)
            error("Preprocessed data missing cue-condition grouping. Call preprocess_data_for_fitting with group_by_condition=true.")
        end
        cond_type_use = cond_type == :all ? :single : cond_type

        # Calculate baseline weights
        if weighting_mode == :exponential
            weights = exp.(w_slope_normalized .* rewards)
        elseif weighting_mode == :free
            weights = [get(weight_lookup, r, default_weight) for r in rewards]
        elseif weighting_mode == :excitation_inhibition
            # For excitation_inhibition mode, use exponential as baseline
            # Compute baseline weights first (no Ge/Gi applied yet)
            weights = exp.(w_slope_normalized .* rewards)
        else
            error("Unknown weighting_mode: $weighting_mode")
        end
        
        rel_weights = weights ./ sum(weights)
        C_use = cond_type_use == :double ? C_double : C_single
        drift_rates = C_use .* rel_weights
        
        # Apply excitation/inhibition gains DIRECTLY to drift rates for 2-cue conditions
        # This ensures the control mechanism has a direct, un-diluted effect on accumulation rates
        # (Decision Control Theory: reactive control modulates drift rates directly)
        if weighting_mode == :excitation_inhibition && cond_type_use == :double
            # Find highest reward cue (target)
            max_reward_val = maximum(rewards)
            max_reward_idx = findfirst(r -> r == max_reward_val, rewards)
            
            # Apply Ge (excitatory gain) to highest reward cue's drift rate
            if !isnothing(max_reward_idx)
                drift_rates[max_reward_idx] *= Ge
            end
            
            # Apply Gi (inhibitory gain) to all other cues with reward > 0 (distractors)
            for (idx, r) in enumerate(rewards)
                if r > 0 && r < max_reward_val
                    drift_rates[idx] *= Gi
                end
                # Non-cued items (reward 0) keep baseline drift rate (already set above)
            end
        end

        # Create LBA once for this configuration
        k_use = cond_type_use == :double ? k_double : k_single
        t0_use = cond_type_use == :double ? t0_double : t0_single
        lba = LBA(ν=drift_rates, A=A, k=k_use, τ=t0_use)

        # Process all trials with this reward configuration
        for trial_idx in trial_indices
            rt = preprocessed.rts[trial_idx]
            choice = preprocessed.choices[trial_idx]

            lik = 0.0
            if rt > t0_use
                try
                    lik = pdf(lba, (choice=choice, rt=rt))
                    if isnan(lik) || isinf(lik) lik = 1e-10 end
                catch
                    lik = 1e-10
                end
            else
                lik = 1e-10
            end

            if use_contaminant
                alpha_use = if layout === nothing
                    contam_alpha_use
                elseif estimate_contaminant && vary_contam_by_cue && haskey(layout.idx_contam_alpha, cond_type_use)
                    params[layout.idx_contam_alpha[cond_type_use]]
                else
                    contam_alpha_use
                end
                rtmax_use = if layout === nothing
                    contam_rt_max_use
                elseif estimate_contaminant && vary_contam_by_cue && haskey(layout.idx_contam_rt, cond_type_use)
                    params[layout.idx_contam_rt[cond_type_use]]
                else
                    contam_rt_max_use
                end
                uniform_density = 1.0 / rtmax_use
                lik = (1 - alpha_use) * lik + alpha_use * uniform_density
            end

            if lik <= 1e-20 lik = 1e-20 end
            total_neg_ll -= log(lik)
        end
    end

    return total_neg_ll
end
