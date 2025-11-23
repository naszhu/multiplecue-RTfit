# ==========================================================================
# Model Utilities Module
# MIS-LBA mixture model logic and likelihood calculations
# ==========================================================================

module ModelUtils

using DataFrames
using Distributions
using SequentialSamplingModels

export mis_lba_mixture_loglike, mis_lba_dual_mixture_loglike, mis_lba_single_loglike, mis_lba_allconditions_loglike
export PreprocessedData, preprocess_data_for_fitting

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
    mis_lba_allconditions_loglike(params::Vector{Float64}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, use_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64

    Computes the negative log-likelihood for a single LBA model fitted to ALL conditions at once.
    Allows optional variation of C and/or t0 by single vs double cue conditions.
"""
function mis_lba_allconditions_loglike(params::Vector{<:Real}, df::DataFrame; r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, cue_condition_types::Union{Nothing,Vector{Symbol}}=nothing, use_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64
    # Derive cue-condition types if needed
    cond_types = nothing
    if vary_C_by_cue_type || vary_t0_by_cue_type || vary_k_by_cue_type
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

    # Unpack parameters and constraints based on weighting_mode
    p_idx = 1
    C_single = params[p_idx]; p_idx += 1
    C_double = vary_C_by_cue_type ? params[p_idx] : C_single
    p_idx += vary_C_by_cue_type ? 1 : 0
    w_slope = 0.0
    w2 = w3 = w4 = 0.0
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
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free.")
    end

    total_neg_ll = 0.0

    # Compute r_max only for exponential mode
    if weighting_mode == :exponential
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

    w_slope_normalized = weighting_mode == :exponential ? (w_slope / r_max) : 0.0 # only used for exponential mode
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
            weights = weighting_mode == :exponential ?
                      exp.(w_slope_normalized .* rewards) :
                      [get(weight_lookup, r, default_weight) for r in rewards]
            rel_weights = weights ./ sum(weights)
            C_use = cond_type == :double ? C_double : C_single
            drift_rates = C_use .* rel_weights
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
            uniform_density = 1.0 / contaminant_rt_max
            lik = (1 - contaminant_alpha) * lik + contaminant_alpha * uniform_density
        end

        if lik <= 1e-20 lik = 1e-20 end
        total_neg_ll -= log(lik)
    end

    return total_neg_ll
end

"""
    mis_lba_allconditions_loglike(params::Vector{Float64}, preprocessed::PreprocessedData; r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, use_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64

Ultra-fast likelihood computation using preprocessed data (method overload).
Computes drift rates only once per unique reward configuration.
This version is 3-5x faster than the DataFrame version.
"""
function mis_lba_allconditions_loglike(params::Vector{<:Real}, preprocessed::PreprocessedData; r_max::Union{Nothing,Float64}=nothing, weighting_mode::Symbol=:exponential, vary_C_by_cue_type::Bool=false, vary_t0_by_cue_type::Bool=false, vary_k_by_cue_type::Bool=false, use_contaminant::Bool=false, contaminant_alpha::Float64=0.0, contaminant_rt_max::Float64=3.0)::Float64
    # Unpack parameters and constraints based on weighting_mode
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
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free.")
    end

    # r_max is only used for exponential mode
    if weighting_mode == :exponential
        if isnothing(r_max)
            r_max = 4.0
        end
    else
        r_max = isnothing(r_max) ? 1.0 : r_max
    end

    total_neg_ll = 0.0
    w_slope_normalized = weighting_mode == :exponential ? (w_slope / r_max) : 0.0
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
        if (vary_C_by_cue_type || vary_t0_by_cue_type || vary_k_by_cue_type) && cond_type in (:all, :mixed)
            error("Preprocessed data missing cue-condition grouping. Call preprocess_data_for_fitting with group_by_condition=true.")
        end
        cond_type_use = cond_type == :all ? :single : cond_type

        weights = weighting_mode == :exponential ?
                  exp.(w_slope_normalized .* rewards) :
                  [get(weight_lookup, r, default_weight) for r in rewards]
        rel_weights = weights ./ sum(weights)
        C_use = cond_type_use == :double ? C_double : C_single
        drift_rates = C_use .* rel_weights

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
                uniform_density = 1.0 / contaminant_rt_max
                lik = (1 - contaminant_alpha) * lik + contaminant_alpha * uniform_density
            end

            if lik <= 1e-20 lik = 1e-20 end
            total_neg_ll -= log(lik)
        end
    end

    return total_neg_ll
end

end # module
