# ==========================================================================
# Model Utilities Module
# MIS-LBA mixture model logic and likelihood calculations
# ==========================================================================

module ModelUtils

using DataFrames
using Distributions
using SequentialSamplingModels

export mis_lba_mixture_loglike, mis_lba_dual_mixture_loglike, mis_lba_single_loglike

"""
    mis_lba_mixture_loglike(params, df::DataFrame)

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
function mis_lba_mixture_loglike(params, df::DataFrame)
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
    mis_lba_dual_mixture_loglike(params, df::DataFrame; r_max=nothing)

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
function mis_lba_dual_mixture_loglike(params, df::DataFrame; r_max=nothing)
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

    # Accumulate log-likelihood across all trials
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]

        # --- MIS THEORY ---
        # Weight = exp(θ * r / r_max) as per paper
        # Exponential weighting allows winner-take-all behavior
        weights = exp.(w_slope .* rewards ./ r_max)
        rel_weights = weights ./ sum(weights)
        drift_rates = C .* rel_weights

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
    mis_lba_single_loglike(params, df::DataFrame; r_max=nothing)

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
function mis_lba_single_loglike(params, df::DataFrame; r_max=nothing)
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

    # Accumulate log-likelihood across all trials
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]

        # --- MIS THEORY ---
        # Weight = exp(θ * r / r_max) as per paper
        # Exponential weighting allows winner-take-all behavior
        weights = exp.(w_slope .* rewards ./ r_max)
        rel_weights = weights ./ sum(weights)
        drift_rates = C .* rel_weights

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

end # module
