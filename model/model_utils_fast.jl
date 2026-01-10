# ==========================================================================
# EXPERIMENTAL: Ultra-Fast Model Utilities
# Pre-indexed data structures for maximum speed
# ==========================================================================

module ModelUtilsFast

using DataFrames
using Distributions
using SequentialSamplingModels

export mis_lba_allconditions_loglike_fast, preprocess_data_for_fast_fitting

"""
    PreprocessedData

Pre-indexed data structure for ultra-fast likelihood computation.
Groups trials by unique reward configurations to minimize redundant computations.
"""
struct PreprocessedData
    unique_rewards::Vector{Vector{Float64}}
    trial_groups::Vector{Vector{Int}}  # Indices of trials for each unique reward config
    rts::Vector{Float64}
    choices::Vector{Int}
    n_trials::Int
end

"""
    preprocess_data_for_fast_fitting(df::DataFrame)

Preprocess data to group trials by unique reward configurations.
This enables ultra-fast likelihood computation by avoiding redundant drift rate calculations.
"""
function preprocess_data_for_fast_fitting(df::DataFrame)
    # Find unique reward configurations
    unique_rewards = unique(df.ParsedRewards)
    n_unique = length(unique_rewards)

    # Group trial indices by reward configuration
    trial_groups = Vector{Vector{Int}}(undef, n_unique)
    for i in 1:n_unique
        trial_groups[i] = Int[]
    end

    # Map each trial to its reward configuration group
    reward_to_idx = Dict{Vector{Float64}, Int}()
    for (idx, rewards) in enumerate(unique_rewards)
        reward_to_idx[rewards] = idx
    end

    for trial_idx in 1:nrow(df)
        group_idx = reward_to_idx[df.ParsedRewards[trial_idx]]
        push!(trial_groups[group_idx], trial_idx)
    end

    # Extract RT and choice arrays for fast access
    rts = Vector{Float64}(df.CleanRT)
    choices = Vector{Int}(df.Choice)

    return PreprocessedData(unique_rewards, trial_groups, rts, choices, nrow(df))
end

"""
    mis_lba_allconditions_loglike_fast(params, preprocessed_data::PreprocessedData; r_max=4.0)

Ultra-fast likelihood computation using preprocessed data.
Computes drift rates only once per unique reward configuration.
"""
function mis_lba_allconditions_loglike_fast(params, preprocessed_data::PreprocessedData; r_max=4.0)
    # Unpack parameters
    C, w_slope, A, k, t0 = params

    # Constraints
    # k > A ensures threshold (A+k) is meaningfully above max starting point (A)
    if C<=0 || w_slope<0 || A<=0 || k<=0 || k<=A || t0<=0 || t0 < 0.01
        return Inf
    end

    total_neg_ll = 0.0
    w_slope_normalized = w_slope / r_max

    # Process each unique reward configuration
    for (rewards, trial_indices) in zip(preprocessed_data.unique_rewards, preprocessed_data.trial_groups)
        # Compute drift rates once for this configuration
        weights = exp.(w_slope_normalized .* rewards)
        rel_weights = weights ./ sum(weights)
        drift_rates = C .* rel_weights

        # Create LBA once for this configuration
        lba = LBA(ν=drift_rates, A=A, k=k, τ=t0)

        # Process all trials with this reward configuration
        for trial_idx in trial_indices
            rt = preprocessed_data.rts[trial_idx]
            choice = preprocessed_data.choices[trial_idx]

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
    end

    return total_neg_ll
end

end # module
