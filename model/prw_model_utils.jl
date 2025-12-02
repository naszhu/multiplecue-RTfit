module PRWModel

using LinearAlgebra
using Distributions
using DataFrames

import Main.ModelUtils: PreprocessedData

export mis_prw_allconditions_loglike, PRWLayout

"""
    PRWLayout

Index layout for MIS Poisson random walk parameters (all-conditions fit).
Matches the parameter order produced by the PRW builder in the fit script.
"""
struct PRWLayout
    weighting_mode::Symbol
    vary_C_by_cue::Bool
    vary_t0_by_cue::Bool
    vary_k_by_cue::Bool
    idx_C::Dict{Symbol,Int}
    idx_t0::Dict{Symbol,Int}
    idx_k::Dict{Symbol,Int}
    idx_w::Dict{Symbol,Int}
end

# ==========================================================================
# Core PRW building blocks
# ==========================================================================

"""
    build_transition_matrix(probs::Vector{Float64}, k::Int)

Construct the transition matrix for a Star Random Walk with `length(probs)` branches
and integer threshold `k`.
"""
function build_transition_matrix(probs::Vector{Float64}, k::Int)
    n_opts = length(probs)
    n_states = 1 + n_opts * (k - 1)  # transient states only

    T = zeros(Float64, n_states, n_states)

    # Transitions from origin (state 1) into the first step of each branch
    if k > 1
        for i in 1:n_opts
            target_state = 1 + (i - 1) * (k - 1) + 1
            T[1, target_state] = probs[i]
        end
    end

    # Transitions within each branch
    for i in 1:n_opts
        branch_start = 1 + (i - 1) * (k - 1)
        for step in 1:(k - 1)
            row_idx = branch_start + step
            p_forward = probs[i]
            p_backward = 1.0 - p_forward

            # forward
            if step < k - 1
                T[row_idx, row_idx + 1] = p_forward
            end

            # backward
            if step > 1
                T[row_idx, row_idx - 1] = p_backward
            else
                T[row_idx, 1] = p_backward
            end
        end
    end

    return T
end

"""
    compute_absorption_probs(T, probs, k, max_steps)

Compute absorption probabilities at each option for steps 1..max_steps.
Returns a matrix of size (max_steps, n_opts).
"""
function compute_absorption_probs(T::Matrix{Float64}, probs::Vector{Float64}, k::Int, max_steps::Int)
    n_opts = length(probs)
    n_states = size(T, 1)
    current_state_probs = zeros(Float64, n_states)
    current_state_probs[1] = 1.0

    abs_probs = zeros(Float64, max_steps, n_opts)

    for n in 1:max_steps
        for i in 1:n_opts
            if k > 1
                pre_abs_state = 1 + i * (k - 1)
                abs_probs[n, i] = current_state_probs[pre_abs_state] * probs[i]
            elseif n == 1
                abs_probs[n, i] = probs[i]
            end
        end
        current_state_probs = transpose(T) * current_state_probs
    end

    return abs_probs
end

"""
    prw_pdf_point(rt, t0, C, abs_probs, choice)

Evaluate the PRW PDF for a single RT/choice by convolving step counts with an Erlang.
"""
function prw_pdf_point(rt::Float64, t0::Float64, C::Float64, abs_probs::Matrix{Float64}, choice::Int)
    decision_time = rt - t0
    if decision_time <= 0
        return 1e-20
    end

    density = 0.0
    max_steps = size(abs_probs, 1)
    log_Ct = log(C) + log(decision_time)

    @inbounds for n in 1:max_steps
        prob_absorb_n = abs_probs[n, choice]
        if prob_absorb_n > 1e-10
            # log(erlang_pdf) with shape=n, rate=C
            log_erlang = log(C) + (n - 1) * log_Ct - log(factorial(big(n - 1))) - (C * decision_time)
            density += exp(Float64(log_erlang)) * prob_absorb_n
        end
    end

    return max(density, 1e-20)
end

# ==========================================================================
# MIS-PRW likelihood (all conditions)
# ==========================================================================

"""
    mis_prw_allconditions_loglike(params, preprocessed::PreprocessedData; layout, r_max=4.0, max_steps=60)

Negative log-likelihood for the MIS Poisson random walk model using preprocessed data.
Supports cue-type specific C/k/t0 through the provided `layout`.
"""
function mis_prw_allconditions_loglike(params::Vector{<:Real}, preprocessed::PreprocessedData;
    layout::PRWLayout,
    r_max::Float64=4.0,
    max_steps::Int=60)

    weighting_mode = layout.weighting_mode
    weights_exponential = weighting_mode == :exponential

    w_slope = 0.0
    w2 = w3 = w4 = 0.0
    if weights_exponential
        w_slope = params[layout.idx_w[:w_slope]]
    elseif weighting_mode == :free
        w2 = params[layout.idx_w[:w2]]
        w3 = params[layout.idx_w[:w3]]
        w4 = params[layout.idx_w[:w4]]
        if w2 <= 0 || w3 <= 0 || w4 <= 0
            return Inf
        end
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free.")
    end

    C_single = haskey(layout.idx_C, :single) ? params[layout.idx_C[:single]] : params[layout.idx_C[:all]]
    C_double = haskey(layout.idx_C, :double) ? params[layout.idx_C[:double]] : C_single
    k_single = haskey(layout.idx_k, :single) ? params[layout.idx_k[:single]] : params[layout.idx_k[:all]]
    k_double = haskey(layout.idx_k, :double) ? params[layout.idx_k[:double]] : k_single
    t0_single = haskey(layout.idx_t0, :single) ? params[layout.idx_t0[:single]] : params[layout.idx_t0[:all]]
    t0_double = haskey(layout.idx_t0, :double) ? params[layout.idx_t0[:double]] : t0_single

    if any(x -> x <= 0, (C_single, C_double, k_single, k_double, t0_single, t0_double)) || any(x -> x < 0.05, (t0_single, t0_double)) || k_single < 1.0 || k_double < 1.0
        return Inf
    end

    weight_lookup = weights_exponential ? nothing : Dict(
        1.0 => 1.0,
        2.0 => w2,
        3.0 => w3,
        4.0 => w4,
        0.0 => 1e-10,
    )
    default_weight = weights_exponential ? 1.0 : weight_lookup[0.0]

    total_neg_ll = 0.0
    w_slope_normalized = weights_exponential ? (w_slope / r_max) : 0.0

    @inbounds for (idx, rewards) in enumerate(preprocessed.unique_rewards)
        trial_indices = preprocessed.trial_groups[idx]
        cond_type = preprocessed.group_condition_types[idx]
        if (layout.vary_C_by_cue || layout.vary_t0_by_cue || layout.vary_k_by_cue) && cond_type in (:all, :mixed)
            error("Preprocessed data missing cue-condition grouping. Call preprocess_data_for_fitting with group_by_condition=true.")
        end
        cond_type_use = cond_type == :all ? :single : cond_type

        weights = weights_exponential ?
            exp.(w_slope_normalized .* rewards) :
            [get(weight_lookup, r, default_weight) for r in rewards]
        step_probs = weights ./ sum(weights)

        k_use = cond_type_use == :double ? k_double : k_single
        t0_use = cond_type_use == :double ? t0_double : t0_single
        C_use = cond_type_use == :double ? C_double : C_single

        k_floor = floor(Int, k_use)
        k_ceil = ceil(Int, k_use)
        p_ceil = k_use - k_floor
        p_floor = 1.0 - p_ceil

        T_floor = build_transition_matrix(step_probs, k_floor)
        abs_floor = compute_absorption_probs(T_floor, step_probs, k_floor, max_steps)
        abs_ceil = if k_ceil > k_floor
            T_ceil = build_transition_matrix(step_probs, k_ceil)
            compute_absorption_probs(T_ceil, step_probs, k_ceil, max_steps)
        else
            abs_floor
        end

        for trial_idx in trial_indices
            rt = preprocessed.rts[trial_idx]
            choice = preprocessed.choices[trial_idx]

            lik_floor = prw_pdf_point(rt, t0_use, C_use, abs_floor, choice)
            lik_ceil = prw_pdf_point(rt, t0_use, C_use, abs_ceil, choice)
            lik = p_floor * lik_floor + p_ceil * lik_ceil

            total_neg_ll -= log(max(lik, 1e-20))
        end
    end

    return total_neg_ll
end

end # module
