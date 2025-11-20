# ==========================================================================
# Plotting Utilities Module
# Visualization functions for model fitting results
# ==========================================================================

module PlottingUtils

using DataFrames
using Distributions
using SequentialSamplingModels
using Statistics
using Plots

export generate_plot, generate_plot_dual, generate_plot_single
export generate_accuracy_plot_dual, generate_overall_accuracy_plot, generate_overall_accuracy_plot_single

"""
    generate_plot(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing)

    Generates a plot comparing observed RT distribution with model predictions.
    Shows the mixture components separately to visualize bimodality.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - params: Vector of model parameters [C, w, A, k, t0, p_exp, mu_exp, sig_exp]
    - output_plot: Output filename for plot
    - cue_condition: Optional cue condition identifier for plot title
"""
function generate_plot(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing)
    println("Generating plot...")

    # Unpack parameters
    C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp = params

    # Create title with cue condition and mixture info
    title_str = "MIS-LBA Mixture Fit"
    if !isnothing(cue_condition)
        title_str = "MIS-LBA Mixture Fit - Cue Condition: $cue_condition"
    end
    title_str *= "\n(p_exp=$(round(p_exp, digits=4)), μ_exp=$(round(mu_exp, digits=3)), σ_exp=$(round(sig_exp, digits=3)))"

    # Compute kernel density estimate (KDE) for observed data
    rt_min = minimum(data.CleanRT)
    rt_max = maximum(data.CleanRT)
    rt_range = rt_max - rt_min
    kde_grid = range(max(0.05, rt_min - 0.1*rt_range), min(1.5, rt_max + 0.1*rt_range), length=200)

    # Adaptive bandwidth using Silverman's rule of thumb
    n = length(data.CleanRT)
    rt_std = std(data.CleanRT)
    rt_quantiles = quantile(data.CleanRT, [0.25, 0.75])
    rt_iqr = rt_quantiles[2] - rt_quantiles[1]
    # Silverman's rule: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    bandwidth = 0.9 * min(rt_std, rt_iqr / 1.34) * (n ^ (-1/5))
    # Ensure minimum bandwidth to avoid overfitting
    bandwidth = max(bandwidth, 0.01)

    # Simple KDE using Gaussian kernel
    kde_dens = zeros(length(kde_grid))
    for (i, t) in enumerate(kde_grid)
        kde_dens[i] = mean([pdf(Normal(t, bandwidth), rt) for rt in data.CleanRT])
    end

    # Normalize to proper density (integral = 1)
    dx = kde_grid[2] - kde_grid[1]
    kde_dens ./= sum(kde_dens) * dx

    # Create plot with KDE line
    p = plot(kde_grid, kde_dens, label="Observed", linewidth=2.5,
             color=:darkblue, linestyle=:solid, alpha=0.8,
             xlabel="Reaction Time (s)", ylabel="Density", title=title_str,
             legend=:topright, size=(800, 600))

    # Simulate Model Curve
    # We calculate the unconditional PDF by averaging across all unique reward structures
    t_grid = range(0.05, 1.5, length=300)
    y_pred_total = zeros(length(t_grid))
    y_pred_express = zeros(length(t_grid))
    y_pred_lba = zeros(length(t_grid))

    # Get unique reward structures to properly weight the unconditional PDF
    # Use string representation as key since arrays can't be dict keys directly
    reward_counts = Dict()
    reward_arrays = Dict()  # Store actual arrays
    for rewards in data.ParsedRewards
        key = string(rewards)  # Use string representation as key
        if !haskey(reward_counts, key)
            reward_counts[key] = 0
            reward_arrays[key] = rewards
        end
        reward_counts[key] += 1
    end

    # Compute PDF for each unique reward structure and weight by frequency
    total_weight = 0.0
    for (key, rewards) in reward_arrays
        weight = reward_counts[key]
        total_weight += weight

        # Reconstruct parameters for this reward structure
        ws = 1.0 .+ (w_slope .* rewards)
        vs = C .* (ws ./ sum(ws))

        # LBA PDF (summed over all choices)
        lba = LBA(ν=vs, A=A, k=k, τ=t0)

        for (j, t) in enumerate(t_grid)
            # LBA component
            lba_dens = 0.0
            if t > t0
                try
                    # Sum pdf of all possible choices
                    lba_dens = sum([pdf(lba, (choice=c, rt=t)) for c in 1:length(vs)])
                    if isnan(lba_dens) || isinf(lba_dens)
                        lba_dens = 0.0
                    end
                catch
                    lba_dens = 0.0
                end
            end

            # Express component
            exp_dens = pdf(Normal(mu_exp, sig_exp), t)
            if isnan(exp_dens) || isinf(exp_dens)
                exp_dens = 0.0
            end

            # Weighted mixture components
            y_pred_express[j] += weight * p_exp * exp_dens
            y_pred_lba[j] += weight * (1-p_exp) * lba_dens
            y_pred_total[j] += weight * ((p_exp * exp_dens) + ((1-p_exp) * lba_dens))
        end
    end

    # Normalize by total weight
    if total_weight > 0
        y_pred_total ./= total_weight
        y_pred_express ./= total_weight
        y_pred_lba ./= total_weight
    end

    # Find peak locations for diagnostics
    max_total_idx = argmax(y_pred_total)
    max_total_rt = t_grid[max_total_idx]
    max_total_dens = y_pred_total[max_total_idx]

    max_exp_idx = p_exp > 1e-6 ? argmax(y_pred_express) : nothing
    max_exp_rt = !isnothing(max_exp_idx) ? t_grid[max_exp_idx] : nothing
    max_exp_dens = !isnothing(max_exp_idx) ? y_pred_express[max_exp_idx] : 0.0

    # Check if components are well-separated
    separation = !isnothing(max_exp_rt) ? abs(max_exp_rt - max_total_rt) : 0.0
    well_separated = separation > 0.1  # At least 100ms separation
    component_visible = p_exp > 1e-6 && max_exp_dens > max_total_dens * 0.1  # Express peak is at least 10% of main peak

    # Plot components separately to show bimodality
    if p_exp > 1e-6  # Only show express component if it's non-negligible
        plot!(p, t_grid, y_pred_express, label="Express Component (p=$(round(p_exp, digits=3)))",
              linewidth=2, color=:orange, linestyle=:dash, alpha=0.7)
    end

    plot!(p, t_grid, y_pred_lba, label="LBA Component (p=$(round(1-p_exp, digits=3)))",
          linewidth=2, color=:green, linestyle=:dash, alpha=0.7)

    plot!(p, t_grid, y_pred_total, label="Total Mixture", linewidth=3, color=:red)

    # Add diagnostic text
    y_pos = maximum(y_pred_total) * 0.85
    if p_exp < 1e-6
        annotate!(p, 0.7, y_pos,
                  text("⚠ Express component collapsed (p_exp ≈ 0)", :red, :left, 10))
    elseif !well_separated || !component_visible
        warning_msg = "⚠ Express component too small/close to LBA mode"
        if !well_separated
            warning_msg *= "\n  (separation: $(round(separation*1000, digits=0))ms < 100ms)"
        end
        if !component_visible
            warning_msg *= "\n  (express peak: $(round(max_exp_dens, digits=3)) vs main: $(round(max_total_dens, digits=3)))"
        end
        annotate!(p, 0.7, y_pos,
                  text(warning_msg, :orange, :left, 9))
    end

    # Add vertical lines at component peaks
    if p_exp > 1e-6 && !isnothing(max_exp_rt)
        vline!(p, [max_exp_rt], color=:orange, linestyle=:dot, linewidth=1, alpha=0.5, label="")
    end
    vline!(p, [max_total_rt], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="")

    savefig(p, output_plot)
    println("Saved plot to $output_plot")
    println("  KDE bandwidth (adaptive): $(round(bandwidth, digits=4))s (n=$n, std=$(round(rt_std, digits=3)), IQR=$(round(rt_iqr, digits=3)))")
    println("  Express probability: $(round(p_exp, digits=6))")
    println("  Express mean: $(round(mu_exp, digits=3))s, std: $(round(sig_exp, digits=3))s")
    println("  Non-decision time (t0): $(round(t0, digits=3))s")
    if p_exp > 1e-6
        println("  Express-LBA separation: $(round(separation*1000, digits=0))ms")
        println("  Express peak density: $(round(max_exp_dens, digits=4))")
        println("  Main peak density: $(round(max_total_dens, digits=4))")
        if !well_separated || !component_visible
            println("  ⚠ WARNING: Express component may not create visible bimodality")
        end
    end
end

"""
    generate_plot_dual(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing, r_max=nothing, config=nothing)

    Generates a plot for dual-LBA mixture model showing both LBA components separately.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - params: Vector of model parameters [C, w, A1, k1, t0_1, A2, k2, t0_2, p_mix]
    - output_plot: Output filename for plot
    - cue_condition: Optional cue condition identifier for plot title
    - r_max: Optional maximum reward value across entire experiment (for consistent normalization)
    - config: Optional ModelConfig object with display flags
"""
function generate_plot_dual(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing, r_max=nothing, config=nothing)
    println("Generating plot for dual-LBA model...")

    # Unpack parameters
    C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix = params

    # Create title
    title_str = "Dual-LBA Mixture Fit"
    if !isnothing(cue_condition)
        title_str = "Dual-LBA Mixture Fit - Cue Condition: $cue_condition"
    end
    title_str *= "\n(p_mix=$(round(p_mix, digits=3)), t0_1=$(round(t0_1, digits=3))s, t0_2=$(round(t0_2, digits=3))s)"

    # Compute kernel density estimate (KDE) for observed data
    rt_min = minimum(data.CleanRT)
    rt_max = maximum(data.CleanRT)
    rt_range = rt_max - rt_min
    kde_grid = range(max(0.05, rt_min - 0.1*rt_range), min(1.5, rt_max + 0.1*rt_range), length=200)

    # Adaptive bandwidth using Silverman's rule of thumb
    n = length(data.CleanRT)
    rt_std = std(data.CleanRT)
    rt_quantiles = quantile(data.CleanRT, [0.25, 0.75])
    rt_iqr = rt_quantiles[2] - rt_quantiles[1]
    # Silverman's rule: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    bandwidth = 0.9 * min(rt_std, rt_iqr / 1.34) * (n ^ (-1/5))
    # Ensure minimum bandwidth to avoid overfitting
    bandwidth = max(bandwidth, 0.01)

    # Simple KDE using Gaussian kernel
    kde_dens = zeros(length(kde_grid))
    for (i, t) in enumerate(kde_grid)
        kde_dens[i] = mean([pdf(Normal(t, bandwidth), rt) for rt in data.CleanRT])
    end

    # Normalize to proper density (integral = 1)
    dx = kde_grid[2] - kde_grid[1]
    kde_dens ./= sum(kde_dens) * dx

    # Create plot with KDE line
    p = plot(kde_grid, kde_dens, label="Observed", linewidth=2.5,
             color=:darkblue, linestyle=:solid, alpha=0.8,
             xlabel="Reaction Time (s)", ylabel="Density", title=title_str,
             legend=:topright, size=(800, 600))

    # Compute unconditional PDF
    t_grid = range(0.05, 1.5, length=300)
    y_pred_total = zeros(length(t_grid))
    y_pred_lba1 = zeros(length(t_grid))
    y_pred_lba2 = zeros(length(t_grid))
    # Choice-specific densities: target (highest reward) vs distractors
    y_pred_target_total = zeros(length(t_grid))
    y_pred_target_lba1 = zeros(length(t_grid))
    y_pred_target_lba2 = zeros(length(t_grid))
    y_pred_distractor_total = zeros(length(t_grid))
    y_pred_distractor_lba1 = zeros(length(t_grid))
    y_pred_distractor_lba2 = zeros(length(t_grid))

    # Compute r_max: use provided value, or compute from dataset if not provided
    if isnothing(r_max)
        r_max = 0.0
        for rewards in data.ParsedRewards
            if !isempty(rewards)
                r_max = max(r_max, maximum(rewards))
            end
        end
        # Avoid division by zero if all rewards are 0
        if r_max <= 0.0
            r_max = 1.0
        end
    end

    # Get unique reward structures
    reward_counts = Dict()
    reward_arrays = Dict()
    for rewards in data.ParsedRewards
        key = string(rewards)
        if !haskey(reward_counts, key)
            reward_counts[key] = 0
            reward_arrays[key] = rewards
        end
        reward_counts[key] += 1
    end

    total_weight = 0.0
    for (key, rewards) in reward_arrays
        weight = reward_counts[key]
        total_weight += weight

        # Reconstruct drift rates using exponential weight function
        # Weight = exp(θ * r / r_max) as per paper
        ws = exp.(w_slope .* rewards ./ r_max)
        vs = C .* (ws ./ sum(ws))

        # LBA components
        lba1 = LBA(ν=vs, A=A1, k=k1, τ=t0_1)
        lba2 = LBA(ν=vs, A=A2, k=k2, τ=t0_2)

        # Identify target choice (highest reward option)
        target_choice = argmax(rewards)
        distractor_choices = [c for c in 1:length(vs) if c != target_choice]

        for (j, t) in enumerate(t_grid)
            # LBA component 1 - total
            lba1_dens = 0.0
            lba1_target_dens = 0.0
            lba1_distractor_dens = 0.0
            if t > t0_1
                try
                    lba1_dens = sum([pdf(lba1, (choice=c, rt=t)) for c in 1:length(vs)])
                    lba1_target_dens = pdf(lba1, (choice=target_choice, rt=t))
                    lba1_distractor_dens = sum([pdf(lba1, (choice=c, rt=t)) for c in distractor_choices])
                    if isnan(lba1_dens) || isinf(lba1_dens) lba1_dens = 0.0 end
                    if isnan(lba1_target_dens) || isinf(lba1_target_dens) lba1_target_dens = 0.0 end
                    if isnan(lba1_distractor_dens) || isinf(lba1_distractor_dens) lba1_distractor_dens = 0.0 end
                catch
                    lba1_dens = 0.0
                    lba1_target_dens = 0.0
                    lba1_distractor_dens = 0.0
                end
            end

            # LBA component 2 - total
            lba2_dens = 0.0
            lba2_target_dens = 0.0
            lba2_distractor_dens = 0.0
            if t > t0_2
                try
                    lba2_dens = sum([pdf(lba2, (choice=c, rt=t)) for c in 1:length(vs)])
                    lba2_target_dens = pdf(lba2, (choice=target_choice, rt=t))
                    lba2_distractor_dens = sum([pdf(lba2, (choice=c, rt=t)) for c in distractor_choices])
                    if isnan(lba2_dens) || isinf(lba2_dens) lba2_dens = 0.0 end
                    if isnan(lba2_target_dens) || isinf(lba2_target_dens) lba2_target_dens = 0.0 end
                    if isnan(lba2_distractor_dens) || isinf(lba2_distractor_dens) lba2_distractor_dens = 0.0 end
                catch
                    lba2_dens = 0.0
                    lba2_target_dens = 0.0
                    lba2_distractor_dens = 0.0
                end
            end

            # Weighted mixture - total
            y_pred_lba1[j] += weight * p_mix * lba1_dens
            y_pred_lba2[j] += weight * (1-p_mix) * lba2_dens
            y_pred_total[j] += weight * ((p_mix * lba1_dens) + ((1-p_mix) * lba2_dens))

            # Weighted mixture - target choice
            y_pred_target_lba1[j] += weight * p_mix * lba1_target_dens
            y_pred_target_lba2[j] += weight * (1-p_mix) * lba2_target_dens
            y_pred_target_total[j] += weight * ((p_mix * lba1_target_dens) + ((1-p_mix) * lba2_target_dens))

            # Weighted mixture - distractor choices
            y_pred_distractor_lba1[j] += weight * p_mix * lba1_distractor_dens
            y_pred_distractor_lba2[j] += weight * (1-p_mix) * lba2_distractor_dens
            y_pred_distractor_total[j] += weight * ((p_mix * lba1_distractor_dens) + ((1-p_mix) * lba2_distractor_dens))
        end
    end

    # Normalize
    if total_weight > 0
        y_pred_total ./= total_weight
        y_pred_lba1 ./= total_weight
        y_pred_lba2 ./= total_weight
        y_pred_target_total ./= total_weight
        y_pred_target_lba1 ./= total_weight
        y_pred_target_lba2 ./= total_weight
        y_pred_distractor_total ./= total_weight
        y_pred_distractor_lba1 ./= total_weight
        y_pred_distractor_lba2 ./= total_weight
    end

    # Find peaks
    max_total_idx = argmax(y_pred_total)
    max_total_rt = t_grid[max_total_idx]

    max_lba1_idx = argmax(y_pred_lba1)
    max_lba1_rt = t_grid[max_lba1_idx]
    max_lba1_dens = y_pred_lba1[max_lba1_idx]

    max_lba2_idx = argmax(y_pred_lba2)
    max_lba2_rt = t_grid[max_lba2_idx]
    max_lba2_dens = y_pred_lba2[max_lba2_idx]

    # Plot components - total
    plot!(p, t_grid, y_pred_lba1, label="LBA Component 1 (Fast, p=$(round(p_mix, digits=3)))",
          linewidth=2, color=:orange, linestyle=:dash, alpha=0.7)
    plot!(p, t_grid, y_pred_lba2, label="LBA Component 2 (Slow, p=$(round(1-p_mix, digits=3)))",
          linewidth=2, color=:green, linestyle=:dash, alpha=0.7)
    plot!(p, t_grid, y_pred_total, label="Total Mixture", linewidth=3, color=:red)

    # Plot choice-specific densities (controlled by config flags)
    if isnothing(config) || config.show_target_choice
        plot!(p, t_grid, y_pred_target_total, label="Target Choice (Highest Reward)",
              linewidth=2.5, color=:purple, linestyle=:solid, alpha=0.8)
    end
    if isnothing(config) || config.show_distractor_choice
        plot!(p, t_grid, y_pred_distractor_total, label="Distractor Choices",
              linewidth=2.5, color=:brown, linestyle=:solid, alpha=0.8)
    end

    # Add vertical lines at peaks
    vline!(p, [max_lba1_rt], color=:orange, linestyle=:dot, linewidth=1, alpha=0.5, label="")
    vline!(p, [max_lba2_rt], color=:green, linestyle=:dot, linewidth=1, alpha=0.5, label="")
    vline!(p, [max_total_rt], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="")

    savefig(p, output_plot)
    println("Saved plot to $output_plot")
    println("  KDE bandwidth (adaptive): $(round(bandwidth, digits=4))s (n=$n, std=$(round(rt_std, digits=3)), IQR=$(round(rt_iqr, digits=3)))")
    println("  Mixing probability: $(round(p_mix, digits=4))")
    println("  LBA1 (fast) - t0: $(round(t0_1, digits=3))s, peak at: $(round(max_lba1_rt, digits=3))s")
    println("  LBA2 (slow) - t0: $(round(t0_2, digits=3))s, peak at: $(round(max_lba2_rt, digits=3))s")
    println("  Component separation: $(round(abs(max_lba1_rt - max_lba2_rt)*1000, digits=0))ms")

    return p
end

"""
    generate_plot_single(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing, r_max=nothing, config=nothing)

    Generates a plot for single LBA model showing the fit to RT distribution.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - params: Vector of model parameters [C, w, A, k, t0]
    - output_plot: Output filename for plot
    - cue_condition: Optional cue condition identifier for plot title
    - r_max: Optional maximum reward value across entire experiment (for consistent normalization)
    - config: Optional ModelConfig object with display flags
"""
function generate_plot_single(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing, r_max=nothing, config=nothing)
    println("Generating plot for single LBA model...")

    # Unpack parameters
    C, w_slope, A, k, t0 = params

    # Create title
    title_str = "Single LBA Fit"
    if !isnothing(cue_condition)
        title_str = "Single LBA Fit - Cue Condition: $cue_condition"
    end
    title_str *= "\n(C=$(round(C, digits=2)), w=$(round(w_slope, digits=2)), t0=$(round(t0, digits=3))s)"

    # Compute kernel density estimate (KDE) for observed data
    rt_min = minimum(data.CleanRT)
    rt_max = maximum(data.CleanRT)
    rt_range = rt_max - rt_min
    kde_grid = range(max(0.05, rt_min - 0.1*rt_range), min(1.5, rt_max + 0.1*rt_range), length=200)

    # Adaptive bandwidth using Silverman's rule of thumb
    n = length(data.CleanRT)
    rt_std = std(data.CleanRT)
    rt_quantiles = quantile(data.CleanRT, [0.25, 0.75])
    rt_iqr = rt_quantiles[2] - rt_quantiles[1]
    bandwidth = 0.9 * min(rt_std, rt_iqr / 1.34) * (n ^ (-1/5))
    bandwidth = max(bandwidth, 0.01)

    # Simple KDE using Gaussian kernel
    kde_dens = zeros(length(kde_grid))
    for (i, t) in enumerate(kde_grid)
        kde_dens[i] = mean([pdf(Normal(t, bandwidth), rt) for rt in data.CleanRT])
    end

    # Normalize to proper density (integral = 1)
    dx = kde_grid[2] - kde_grid[1]
    kde_dens ./= sum(kde_dens) * dx

    # Create plot with KDE line
    p = plot(kde_grid, kde_dens, label="Observed", linewidth=2.5,
             color=:darkblue, linestyle=:solid, alpha=0.8,
             xlabel="Reaction Time (s)", ylabel="Density", title=title_str,
             legend=:topright, size=(800, 600))

    # Compute unconditional PDF
    t_grid = range(0.05, 1.5, length=300)
    y_pred_total = zeros(length(t_grid))
    # Choice-specific densities: target (highest reward) vs distractors
    y_pred_target = zeros(length(t_grid))
    y_pred_distractor = zeros(length(t_grid))

    # Compute r_max: use provided value, or compute from dataset if not provided
    if isnothing(r_max)
        r_max = 0.0
        for rewards in data.ParsedRewards
            if !isempty(rewards)
                r_max = max(r_max, maximum(rewards))
            end
        end
        if r_max <= 0.0
            r_max = 1.0
        end
    end

    # Get unique reward structures
    reward_counts = Dict()
    reward_arrays = Dict()
    for rewards in data.ParsedRewards
        key = string(rewards)
        if !haskey(reward_counts, key)
            reward_counts[key] = 0
            reward_arrays[key] = rewards
        end
        reward_counts[key] += 1
    end

    total_weight = 0.0
    for (key, rewards) in reward_arrays
        weight = reward_counts[key]
        total_weight += weight

        # Reconstruct drift rates using exponential weight function
        ws = exp.(w_slope .* rewards ./ r_max)
        vs = C .* (ws ./ sum(ws))

        # Single LBA component
        lba = LBA(ν=vs, A=A, k=k, τ=t0)

        # Identify target choice (highest reward option)
        target_choice = argmax(rewards)
        distractor_choices = [c for c in 1:length(vs) if c != target_choice]

        for (j, t) in enumerate(t_grid)
            # LBA density
            lba_dens = 0.0
            lba_target_dens = 0.0
            lba_distractor_dens = 0.0
            if t > t0
                try
                    lba_dens = sum([pdf(lba, (choice=c, rt=t)) for c in 1:length(vs)])
                    lba_target_dens = pdf(lba, (choice=target_choice, rt=t))
                    lba_distractor_dens = sum([pdf(lba, (choice=c, rt=t)) for c in distractor_choices])
                    if isnan(lba_dens) || isinf(lba_dens) lba_dens = 0.0 end
                    if isnan(lba_target_dens) || isinf(lba_target_dens) lba_target_dens = 0.0 end
                    if isnan(lba_distractor_dens) || isinf(lba_distractor_dens) lba_distractor_dens = 0.0 end
                catch
                    lba_dens = 0.0
                    lba_target_dens = 0.0
                    lba_distractor_dens = 0.0
                end
            end

            # Accumulate weighted
            y_pred_total[j] += weight * lba_dens
            y_pred_target[j] += weight * lba_target_dens
            y_pred_distractor[j] += weight * lba_distractor_dens
        end
    end

    # Normalize
    if total_weight > 0
        y_pred_total ./= total_weight
        y_pred_target ./= total_weight
        y_pred_distractor ./= total_weight
    end

    # Find peak
    max_total_idx = argmax(y_pred_total)
    max_total_rt = t_grid[max_total_idx]

    # Plot model fit
    plot!(p, t_grid, y_pred_total, label="Model Fit", linewidth=3, color=:red)

    # Plot choice-specific densities (controlled by config flags)
    if isnothing(config) || config.show_target_choice
        plot!(p, t_grid, y_pred_target, label="Target Choice (Highest Reward)",
              linewidth=2.5, color=:purple, linestyle=:solid, alpha=0.8)
    end
    if isnothing(config) || config.show_distractor_choice
        plot!(p, t_grid, y_pred_distractor, label="Distractor Choices",
              linewidth=2.5, color=:brown, linestyle=:solid, alpha=0.8)
    end

    # Add vertical line at peak
    vline!(p, [max_total_rt], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="")

    savefig(p, output_plot)
    println("Saved plot to $output_plot")
    println("  KDE bandwidth (adaptive): $(round(bandwidth, digits=4))s (n=$n, std=$(round(rt_std, digits=3)), IQR=$(round(rt_iqr, digits=3)))")
    println("  Non-decision time (t0): $(round(t0, digits=3))s")
    println("  Peak RT: $(round(max_total_rt, digits=3))s")

    return p
end

"""
    generate_accuracy_plot_dual(data::DataFrame, params, output_plot="accuracy_plot.png"; cue_condition=nothing, r_max=nothing)

    Generates a plot showing observed vs predicted choice probability for the target option
    (highest reward option). Groups by CueCondition to match experimental design.

    Arguments:
    - data: DataFrame with CleanRT, Choice, ParsedRewards, and CueCondition columns
    - params: Vector of model parameters [C, w, A1, k1, t0_1, A2, k2, t0_2, p_mix]
    - output_plot: Output filename for plot
    - cue_condition: Optional cue condition identifier for plot title (if provided, shows only that condition)
    - r_max: Optional maximum reward value across entire experiment (for consistent normalization)
"""
function generate_accuracy_plot_dual(data::DataFrame, params, output_plot="accuracy_plot.png"; cue_condition=nothing, r_max=nothing)
    println("Generating accuracy plot for dual-LBA model...")

    # Unpack parameters
    C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix = params

    # Compute r_max: use provided value, or compute from dataset if not provided
    if isnothing(r_max)
        r_max = 0.0
        for rewards in data.ParsedRewards
            if !isempty(rewards)
                r_max = max(r_max, maximum(rewards))
            end
        end
        if r_max <= 0.0
            r_max = 1.0
        end
    end

    # Check if CueCondition column exists
    has_cue_condition = "CueCondition" in names(data)

    if has_cue_condition && isnothing(cue_condition)
        # Group by CueCondition for overall plot
        cue_conditions = unique(data.CueCondition)
        filter!(x -> !ismissing(x), cue_conditions)
        sort!(cue_conditions)

        observed_acc = Float64[]
        predicted_acc = Float64[]
        condition_labels = String[]
        n_trials_per_cond = Int[]

        # RT grid for numerical integration
        t_grid = range(0.05, 3.0, length=1000)
        dt = t_grid[2] - t_grid[1]

        for cc in cue_conditions
            condition_data = filter(row -> row.CueCondition == cc, data)
            if nrow(condition_data) < 5
                continue  # Skip conditions with too few trials
            end

            # Group by unique reward structures within this condition
            reward_groups = Dict()
            for (i, row) in enumerate(eachrow(condition_data))
                rewards = row.ParsedRewards
                key = string(rewards)
                if !haskey(reward_groups, key)
                    reward_groups[key] = Dict(
                        "rewards" => rewards,
                        "target_choice" => argmax(rewards),
                        "choices" => Int[],
                        "n_trials" => 0
                    )
                end
                push!(reward_groups[key]["choices"], row.Choice)
                reward_groups[key]["n_trials"] += 1
            end

            # Compute weighted average accuracy across all reward structures in this condition
            total_obs_correct = 0
            total_trials = 0
            total_pred_prob = 0.0
            total_weight = 0.0

            for (key, group) in reward_groups
                rewards = group["rewards"]
                target_choice = group["target_choice"]
                choices = group["choices"]
                n_trials = group["n_trials"]

                # Observed accuracy for this reward structure
                n_correct = sum(choices .== target_choice)
                total_obs_correct += n_correct
                total_trials += n_trials

                # Predicted accuracy for this reward structure
                ws = exp.(w_slope .* rewards ./ r_max)
                vs = C .* (ws ./ sum(ws))

                lba1 = LBA(ν=vs, A=A1, k=k1, τ=t0_1)
                lba2 = LBA(ν=vs, A=A2, k=k2, τ=t0_2)

                # Compute choice probability for each LBA component separately, then mix
                # This is more numerically stable than mixing PDFs first
                pred_prob1 = 0.0
                pred_prob2 = 0.0

                for t in t_grid
                    if t > t0_1
                        try
                            prob1 = pdf(lba1, (choice=target_choice, rt=t))
                            if !isnan(prob1) && !isinf(prob1) && prob1 > 0
                                pred_prob1 += prob1 * dt
                            end
                        catch
                        end
                    end
                    if t > t0_2
                        try
                            prob2 = pdf(lba2, (choice=target_choice, rt=t))
                            if !isnan(prob2) && !isinf(prob2) && prob2 > 0
                                pred_prob2 += prob2 * dt
                            end
                        catch
                        end
                    end
                end

                # Mix the choice probabilities (not the PDFs)
                pred_prob = p_mix * pred_prob1 + (1 - p_mix) * pred_prob2

                # Weight by number of trials
                total_pred_prob += pred_prob * n_trials
                total_weight += n_trials
            end

            if total_trials > 0
                obs_acc = total_obs_correct / total_trials
                pred_acc = total_weight > 0 ? total_pred_prob / total_weight : 0.0

                push!(observed_acc, obs_acc)
                push!(predicted_acc, pred_acc)
                push!(condition_labels, string(cc))
                push!(n_trials_per_cond, total_trials)
            end
        end

        # Create plot
        title_str = "Choice Accuracy: Observed vs Predicted (All Cue Conditions)"
        p = plot(size=(1200, 700), title=title_str,
                 xlabel="Cue Condition", ylabel="Choice Probability (Target Option)",
                 ylim=(0, 1.05), legend=:topright)

        x_pos = 1:length(condition_labels)
        scatter!(p, x_pos, observed_acc, label="Observed", color=:blue, markersize=8, alpha=0.8)
        scatter!(p, x_pos, predicted_acc, label="Predicted", color=:red, markersize=8, alpha=0.8, marker=:x)

        # Add lines
        plot!(p, x_pos, observed_acc, linestyle=:dash, color=:blue, alpha=0.3, label="")
        plot!(p, x_pos, predicted_acc, linestyle=:dash, color=:red, alpha=0.3, label="")

        # Add perfect accuracy line
        plot!(p, [0, length(condition_labels)+1], [1, 1], linestyle=:dot, color=:gray, alpha=0.5, label="Perfect Accuracy", linewidth=1)

        # Set x-axis labels
        plot!(p, xticks=(x_pos, condition_labels), xrotation=45)

        # Add trial count annotations
        for (i, n) in enumerate(n_trials_per_cond)
            annotate!(p, i, 0.05, text("n=$n", :gray, :center, 8))
        end

    else
        # Fallback: group by reward structure (original method) if no CueCondition or specific condition requested
        reward_groups = Dict()
        for (i, row) in enumerate(eachrow(data))
            rewards = row.ParsedRewards
            key = string(rewards)
            if !haskey(reward_groups, key)
                reward_groups[key] = Dict(
                    "rewards" => rewards,
                    "target_choice" => argmax(rewards),
                    "trials" => Int[],
                    "choices" => Int[]
                )
            end
            push!(reward_groups[key]["trials"], i)
            push!(reward_groups[key]["choices"], row.Choice)
        end

        reward_keys = sort(collect(keys(reward_groups)))
        observed_acc = Float64[]
        predicted_acc = Float64[]
        reward_labels = String[]

        t_grid = range(0.05, 3.0, length=1000)
        dt = t_grid[2] - t_grid[1]

        for key in reward_keys
            group = reward_groups[key]
            rewards = group["rewards"]
            target_choice = group["target_choice"]
            choices = group["choices"]

            n_trials = length(choices)
            n_correct = sum(choices .== target_choice)
            obs_acc = n_trials > 0 ? n_correct / n_trials : 0.0
            push!(observed_acc, obs_acc)

            ws = exp.(w_slope .* rewards ./ r_max)
            vs = C .* (ws ./ sum(ws))

            lba1 = LBA(ν=vs, A=A1, k=k1, τ=t0_1)
            lba2 = LBA(ν=vs, A=A2, k=k2, τ=t0_2)

            pred_prob = 0.0
            for t in t_grid
                if t > max(t0_1, t0_2)
                    try
                        prob1 = pdf(lba1, (choice=target_choice, rt=t))
                        prob2 = pdf(lba2, (choice=target_choice, rt=t))
                        if !isnan(prob1) && !isinf(prob1) && !isnan(prob2) && !isinf(prob2)
                            pred_prob += (p_mix * prob1 + (1 - p_mix) * prob2) * dt
                        end
                    catch
                    end
                end
            end
            push!(predicted_acc, pred_prob)
            push!(reward_labels, key)
        end

        title_str = "Choice Accuracy: Target (Highest Reward) Option"
        if !isnothing(cue_condition)
            title_str = "Choice Accuracy - Cue Condition: $cue_condition"
        end

        p = plot(size=(1000, 600), title=title_str, xlabel="Reward Structure",
                 ylabel="Choice Probability (Target)", ylim=(0, 1.05), legend=:topright)

        x_pos = 1:length(reward_keys)
        scatter!(p, x_pos, observed_acc, label="Observed", color=:blue, markersize=6, alpha=0.7)
        scatter!(p, x_pos, predicted_acc, label="Predicted", color=:red, markersize=6, alpha=0.7, marker=:x)

        plot!(p, x_pos, observed_acc, linestyle=:dash, color=:blue, alpha=0.3, label="")
        plot!(p, x_pos, predicted_acc, linestyle=:dash, color=:red, alpha=0.3, label="")
        plot!(p, [0, length(reward_keys)+1], [1, 1], linestyle=:dot, color=:gray, alpha=0.5, label="Perfect Accuracy", linewidth=1)

        if length(reward_keys) > 10
            plot!(p, xticks=(x_pos, ["" for _ in x_pos]), xlabel="Reward Structure (index)")
        else
            plot!(p, xticks=(x_pos, [string(i) for i in 1:length(reward_keys)]))
        end
    end

    savefig(p, output_plot)
    println("Saved accuracy plot to $output_plot")

    if length(observed_acc) > 0
        mean_obs = mean(observed_acc)
        mean_pred = mean(predicted_acc)
        rmse = sqrt(mean((observed_acc .- predicted_acc).^2))
        println("  Mean observed accuracy: $(round(mean_obs, digits=3))")
        println("  Mean predicted accuracy: $(round(mean_pred, digits=3))")
        println("  RMSE: $(round(rmse, digits=3))")
    end
end

"""
    generate_overall_accuracy_plot(condition_fits::Dict, output_plot="accuracy_plot_all_conditions.png"; r_max=nothing)

    Generates one overall accuracy plot showing all CueConditions together.
    Uses condition-specific fitted parameters for each condition.

    Arguments:
    - condition_fits: Dictionary mapping CueCondition => (data=DataFrame, params=Vector)
    - output_plot: Output filename for plot
    - r_max: Optional maximum reward value across entire experiment (for consistent normalization)
"""
function generate_overall_accuracy_plot(condition_fits::Dict, output_plot="accuracy_plot_all_conditions.png"; r_max=nothing)
    println("Generating overall accuracy plot for all conditions...")

    observed_acc = Float64[]
    predicted_acc = Float64[]
    condition_labels = String[]
    n_trials_per_cond = Int[]

    # RT grid for numerical integration
    t_grid = range(0.05, 3.0, length=1000)
    dt = t_grid[2] - t_grid[1]

    # Sort conditions for consistent ordering
    sorted_conditions = sort(collect(keys(condition_fits)))

    for cc in sorted_conditions
        condition_data = condition_fits[cc].data
        params = condition_fits[cc].params

        # Unpack parameters for this condition
        C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix = params

        # Compute r_max: use provided value, or compute from condition data if not provided
        if isnothing(r_max)
            r_max_cond = 0.0
            for rewards in condition_data.ParsedRewards
                if !isempty(rewards)
                    r_max_cond = max(r_max_cond, maximum(rewards))
                end
            end
            if r_max_cond <= 0.0
                r_max_cond = 1.0
            end
            r_max_use = r_max_cond
        else
            r_max_use = r_max
        end

        # Group by unique reward structures within this condition
        reward_groups = Dict()
        for (i, row) in enumerate(eachrow(condition_data))
            rewards = row.ParsedRewards
            key = string(rewards)
            if !haskey(reward_groups, key)
                reward_groups[key] = Dict(
                    "rewards" => rewards,
                    "target_choice" => argmax(rewards),
                    "choices" => Int[],
                    "n_trials" => 0
                )
            end
            push!(reward_groups[key]["choices"], row.Choice)
            reward_groups[key]["n_trials"] += 1
        end

        # Compute weighted average accuracy across all reward structures in this condition
        total_obs_correct = 0
        total_trials = 0
        total_pred_prob = 0.0
        total_weight = 0.0

        for (key, group) in reward_groups
            rewards = group["rewards"]
            target_choice = group["target_choice"]
            choices = group["choices"]
            n_trials = group["n_trials"]

            # Observed accuracy for this reward structure
            n_correct = sum(choices .== target_choice)
            total_obs_correct += n_correct
            total_trials += n_trials

            # Predicted accuracy for this reward structure using THIS condition's parameters
            ws = exp.(w_slope .* rewards ./ r_max_use)
            vs = C .* (ws ./ sum(ws))

            lba1 = LBA(ν=vs, A=A1, k=k1, τ=t0_1)
            lba2 = LBA(ν=vs, A=A2, k=k2, τ=t0_2)

            # Compute choice probability for each LBA component separately, then mix
            # This is more numerically stable than mixing PDFs first
            pred_prob1 = 0.0
            pred_prob2 = 0.0

            for t in t_grid
                if t > t0_1
                    try
                        prob1 = pdf(lba1, (choice=target_choice, rt=t))
                        if !isnan(prob1) && !isinf(prob1) && prob1 > 0
                            pred_prob1 += prob1 * dt
                        end
                    catch
                    end
                end
                if t > t0_2
                    try
                        prob2 = pdf(lba2, (choice=target_choice, rt=t))
                        if !isnan(prob2) && !isinf(prob2) && prob2 > 0
                            pred_prob2 += prob2 * dt
                        end
                    catch
                    end
                end
            end

            # Mix the choice probabilities (not the PDFs)
            pred_prob = p_mix * pred_prob1 + (1 - p_mix) * pred_prob2

            # Weight by number of trials
            total_pred_prob += pred_prob * n_trials
            total_weight += n_trials
        end

        if total_trials > 0
            obs_acc = total_obs_correct / total_trials
            pred_acc = total_weight > 0 ? total_pred_prob / total_weight : 0.0

            push!(observed_acc, obs_acc)
            push!(predicted_acc, pred_acc)
            push!(condition_labels, string(cc))
            push!(n_trials_per_cond, total_trials)
        end
    end

    # Create plot
    title_str = "Choice Accuracy: Observed vs Predicted (All Cue Conditions)\nUsing Condition-Specific Fitted Parameters"
    p = plot(size=(1200, 700), title=title_str,
             xlabel="Cue Condition", ylabel="Choice Probability (Target Option)",
             ylim=(0, 1.05), legend=:topright)

    x_pos = 1:length(condition_labels)
    scatter!(p, x_pos, observed_acc, label="Observed", color=:blue, markersize=10, alpha=0.8)
    scatter!(p, x_pos, predicted_acc, label="Predicted", color=:red, markersize=10, alpha=0.8, marker=:x)

    # Add lines connecting points
    plot!(p, x_pos, observed_acc, linestyle=:dash, color=:blue, alpha=0.3, label="")
    plot!(p, x_pos, predicted_acc, linestyle=:dash, color=:red, alpha=0.3, label="")

    # Add perfect accuracy line
    plot!(p, [0, length(condition_labels)+1], [1, 1], linestyle=:dot, color=:gray, alpha=0.5, label="Perfect Accuracy", linewidth=1)

    # Add diagonal line (perfect prediction)
    plot!(p, [0, length(condition_labels)+1], [0, 1], linestyle=:dot, color=:black, alpha=0.3, label="Perfect Prediction", linewidth=1)

    # Set x-axis labels
    plot!(p, xticks=(x_pos, condition_labels), xrotation=45)

    # Add trial count annotations
    for (i, n) in enumerate(n_trials_per_cond)
        annotate!(p, i, 0.05, text("n=$n", :gray, :center, 8))
    end

    savefig(p, output_plot)
    println("Saved overall accuracy plot to $output_plot")

    if length(observed_acc) > 0
        mean_obs = mean(observed_acc)
        mean_pred = mean(predicted_acc)
        rmse = sqrt(mean((observed_acc .- predicted_acc).^2))
        println("  Mean observed accuracy: $(round(mean_obs, digits=3))")
        println("  Mean predicted accuracy: $(round(mean_pred, digits=3))")
        println("  RMSE: $(round(rmse, digits=3))")
    end
end

"""
    generate_overall_accuracy_plot_single(condition_fits::Dict, output_plot="accuracy_plot_all_conditions.png"; r_max=nothing)

    Generates one overall accuracy plot showing all CueConditions together for single LBA model.
    Uses condition-specific fitted parameters for each condition.

    Arguments:
    - condition_fits: Dictionary mapping CueCondition => (data=DataFrame, params=Vector)
    - output_plot: Output filename for plot
    - r_max: Optional maximum reward value across entire experiment (for consistent normalization)
"""
function generate_overall_accuracy_plot_single(condition_fits::Dict, output_plot="accuracy_plot_all_conditions.png"; r_max=nothing)
    println("Generating overall accuracy plot for all conditions (single LBA)...")

    observed_acc = Float64[]
    predicted_acc = Float64[]
    condition_labels = String[]
    n_trials_per_cond = Int[]

    # RT grid for numerical integration
    t_grid = range(0.05, 3.0, length=1000)
    dt = t_grid[2] - t_grid[1]

    # Sort conditions for consistent ordering
    sorted_conditions = sort(collect(keys(condition_fits)))

    for cc in sorted_conditions
        condition_data = condition_fits[cc].data
        params = condition_fits[cc].params

        # Unpack parameters for this condition
        C, w_slope, A, k, t0 = params

        # Compute r_max: use provided value, or compute from condition data if not provided
        if isnothing(r_max)
            r_max_cond = 0.0
            for rewards in condition_data.ParsedRewards
                if !isempty(rewards)
                    r_max_cond = max(r_max_cond, maximum(rewards))
                end
            end
            if r_max_cond <= 0.0
                r_max_cond = 1.0
            end
            r_max_use = r_max_cond
        else
            r_max_use = r_max
        end

        # Group by unique reward structures within this condition
        reward_groups = Dict()
        for (i, row) in enumerate(eachrow(condition_data))
            rewards = row.ParsedRewards
            key = string(rewards)
            if !haskey(reward_groups, key)
                reward_groups[key] = Dict(
                    "rewards" => rewards,
                    "target_choice" => argmax(rewards),
                    "choices" => Int[],
                    "n_trials" => 0
                )
            end
            push!(reward_groups[key]["choices"], row.Choice)
            reward_groups[key]["n_trials"] += 1
        end

        # Compute weighted average accuracy across all reward structures in this condition
        total_obs_correct = 0
        total_trials = 0
        total_pred_prob = 0.0
        total_weight = 0.0

        for (key, group) in reward_groups
            rewards = group["rewards"]
            target_choice = group["target_choice"]
            choices = group["choices"]
            n_trials = group["n_trials"]

            # Observed accuracy for this reward structure
            n_correct = sum(choices .== target_choice)
            total_obs_correct += n_correct
            total_trials += n_trials

            # Predicted accuracy for this reward structure using THIS condition's parameters
            ws = exp.(w_slope .* rewards ./ r_max_use)
            vs = C .* (ws ./ sum(ws))

            lba = LBA(ν=vs, A=A, k=k, τ=t0)

            # Compute choice probability
            pred_prob = 0.0

            for t in t_grid
                if t > t0
                    try
                        prob = pdf(lba, (choice=target_choice, rt=t))
                        if !isnan(prob) && !isinf(prob) && prob > 0
                            pred_prob += prob * dt
                        end
                    catch
                    end
                end
            end

            # Weight by number of trials
            total_pred_prob += pred_prob * n_trials
            total_weight += n_trials
        end

        if total_trials > 0
            obs_acc = total_obs_correct / total_trials
            pred_acc = total_weight > 0 ? total_pred_prob / total_weight : 0.0

            push!(observed_acc, obs_acc)
            push!(predicted_acc, pred_acc)
            push!(condition_labels, string(cc))
            push!(n_trials_per_cond, total_trials)
        end
    end

    # Create plot
    title_str = "Choice Accuracy: Observed vs Predicted (All Cue Conditions)\nSingle LBA Model - Condition-Specific Fitted Parameters"
    p = plot(size=(1200, 700), title=title_str,
             xlabel="Cue Condition", ylabel="Choice Probability (Target Option)",
             ylim=(0, 1.05), legend=:bottomright)

    x_pos = 1:length(condition_labels)
    scatter!(p, x_pos, observed_acc, label="Observed", color=:blue, markersize=10, alpha=0.8)
    scatter!(p, x_pos, predicted_acc, label="Predicted", color=:red, markersize=10, alpha=0.8, marker=:x)

    # Add lines connecting points
    plot!(p, x_pos, observed_acc, linestyle=:dash, color=:blue, alpha=0.3, label="")
    plot!(p, x_pos, predicted_acc, linestyle=:dash, color=:red, alpha=0.3, label="")

    # Add perfect accuracy line
    plot!(p, [0, length(condition_labels)+1], [1, 1], linestyle=:dot, color=:gray, alpha=0.5, label="Perfect Accuracy", linewidth=1)

    # Set x-axis labels
    plot!(p, xticks=(x_pos, condition_labels), xrotation=45)

    # Add trial count annotations
    for (i, n) in enumerate(n_trials_per_cond)
        annotate!(p, i, 0.05, text("n=$n", :gray, :center, 8))
    end

    savefig(p, output_plot)
    println("Saved overall accuracy plot to $output_plot")

    if length(observed_acc) > 0
        mean_obs = mean(observed_acc)
        mean_pred = mean(predicted_acc)
        rmse = sqrt(mean((observed_acc .- predicted_acc).^2))
        println("  Mean observed accuracy: $(round(mean_obs, digits=3))")
        println("  Mean predicted accuracy: $(round(mean_pred, digits=3))")
        println("  RMSE: $(round(rmse, digits=3))")
    end
end

end # module
