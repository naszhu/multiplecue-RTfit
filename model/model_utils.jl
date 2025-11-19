# ==========================================================================
# Model Utilities Module
# MIS-LBA mixture model logic and likelihood calculations
# ==========================================================================

module ModelUtils

using DataFrames
using Distributions
using SequentialSamplingModels

export mis_lba_mixture_loglike

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

end # module
