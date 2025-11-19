# ==========================================================================
# Fitting Utilities Module
# Optimization and visualization functions
# ==========================================================================

module FittingUtils

using DataFrames
using Distributions
using SequentialSamplingModels
using Optim
using Statistics
using Random
using Plots
using CSV

export fit_model, save_results, generate_plot, save_results_dual, generate_plot_dual

"""
    fit_model(data::DataFrame, objective_func;
              lower, upper, x0, time_limit=600.0)

    Fits the model using optimization.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - objective_func: Function to minimize (typically a log-likelihood function)
    - lower: Vector of lower bounds for parameters
    - upper: Vector of upper bounds for parameters
    - x0: Vector of initial parameter values
    - time_limit: Maximum time in seconds for optimization

    Returns:
    - result: Optim optimization result object
"""
function fit_model(data::DataFrame, objective_func;
                   lower, upper, x0, time_limit=600.0)
    println("Fitting model (this may take a minute)...")

    func = x -> objective_func(x, data)

    opt_options = Optim.Options(time_limit = time_limit, show_trace = true, show_every=5)

    res = optimize(func, lower, upper, x0, Fminbox(BFGS()), opt_options;
                   autodiff=:forward)

    best = Optim.minimizer(res)
    println("\n--- Optimization Complete ---")
    println("Best Params: $best")
    println("Min LogLikelihood: $(Optim.minimum(res))")

    return res
end

"""
    save_results(result, output_csv="model_fit_results.csv"; cue_condition=nothing)

    Saves the optimization results to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename
    - cue_condition: Optional cue condition identifier (for multi-condition fits)

    Returns parameter names and values as a DataFrame.
"""
function save_results(result, output_csv="model_fit_results.csv"; cue_condition=nothing)
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)",
                     "ThreshGap(k)", "NonDec(t0)", "ProbExp", "MuExp", "SigExp"],
        Value = best
    )
    
    # Add cue condition column if provided
    if !isnothing(cue_condition)
        results_df.CueCondition = fill(cue_condition, nrow(results_df))
    end

    CSV.write(output_csv, results_df)
    println("Saved parameters to $output_csv")

    return results_df
end

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

    # Histogram of Observed RT
    p = histogram(data.CleanRT, normalize=true, label="Observed", alpha=0.5, bins=60,
                  xlabel="Reaction Time (s)", ylabel="Density", title=title_str,
                  color=:blue, legend=:topright, size=(800, 600))

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
    save_results_dual(result, output_csv="model_fit_results.csv"; cue_condition=nothing)

    Saves the optimization results for dual-LBA model to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename
    - cue_condition: Optional cue condition identifier

    Returns parameter names and values as a DataFrame.
"""
function save_results_dual(result, output_csv="model_fit_results.csv"; cue_condition=nothing)
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar1(A1)", "ThreshGap1(k1)",
                     "NonDec1(t0_1)", "StartVar2(A2)", "ThreshGap2(k2)", "NonDec2(t0_2)", "ProbMix(p_mix)"],
        Value = best
    )
    
    if !isnothing(cue_condition)
        results_df.CueCondition = fill(cue_condition, nrow(results_df))
    end

    CSV.write(output_csv, results_df)
    println("Saved parameters to $output_csv")

    return results_df
end

"""
    generate_plot_dual(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing)

    Generates a plot for dual-LBA mixture model showing both LBA components separately.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - params: Vector of model parameters [C, w, A1, k1, t0_1, A2, k2, t0_2, p_mix]
    - output_plot: Output filename for plot
    - cue_condition: Optional cue condition identifier for plot title
"""
function generate_plot_dual(data::DataFrame, params, output_plot="model_fit_plot.png"; cue_condition=nothing)
    println("Generating plot for dual-LBA model...")

    # Unpack parameters
    C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix = params

    # Create title
    title_str = "Dual-LBA Mixture Fit"
    if !isnothing(cue_condition)
        title_str = "Dual-LBA Mixture Fit - Cue Condition: $cue_condition"
    end
    title_str *= "\n(p_mix=$(round(p_mix, digits=3)), t0_1=$(round(t0_1, digits=3))s, t0_2=$(round(t0_2, digits=3))s)"

    # Histogram of Observed RT
    p = histogram(data.CleanRT, normalize=true, label="Observed", alpha=0.5, bins=60,
                  xlabel="Reaction Time (s)", ylabel="Density", title=title_str,
                  color=:blue, legend=:topright, size=(800, 600))

    # Compute unconditional PDF
    t_grid = range(0.05, 1.5, length=300)
    y_pred_total = zeros(length(t_grid))
    y_pred_lba1 = zeros(length(t_grid))
    y_pred_lba2 = zeros(length(t_grid))

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

        # Reconstruct drift rates
        ws = 1.0 .+ (w_slope .* rewards)
        vs = C .* (ws ./ sum(ws))

        # LBA components
        lba1 = LBA(ν=vs, A=A1, k=k1, τ=t0_1)
        lba2 = LBA(ν=vs, A=A2, k=k2, τ=t0_2)

        for (j, t) in enumerate(t_grid)
            # LBA component 1
            lba1_dens = 0.0
            if t > t0_1
                try
                    lba1_dens = sum([pdf(lba1, (choice=c, rt=t)) for c in 1:length(vs)])
                    if isnan(lba1_dens) || isinf(lba1_dens) lba1_dens = 0.0 end
                catch
                    lba1_dens = 0.0
                end
            end

            # LBA component 2
            lba2_dens = 0.0
            if t > t0_2
                try
                    lba2_dens = sum([pdf(lba2, (choice=c, rt=t)) for c in 1:length(vs)])
                    if isnan(lba2_dens) || isinf(lba2_dens) lba2_dens = 0.0 end
                catch
                    lba2_dens = 0.0
                end
            end

            # Weighted mixture
            y_pred_lba1[j] += weight * p_mix * lba1_dens
            y_pred_lba2[j] += weight * (1-p_mix) * lba2_dens
            y_pred_total[j] += weight * ((p_mix * lba1_dens) + ((1-p_mix) * lba2_dens))
        end
    end

    # Normalize
    if total_weight > 0
        y_pred_total ./= total_weight
        y_pred_lba1 ./= total_weight
        y_pred_lba2 ./= total_weight
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

    # Plot components
    plot!(p, t_grid, y_pred_lba1, label="LBA Component 1 (Fast, p=$(round(p_mix, digits=3)))", 
          linewidth=2, color=:orange, linestyle=:dash, alpha=0.7)
    plot!(p, t_grid, y_pred_lba2, label="LBA Component 2 (Slow, p=$(round(1-p_mix, digits=3)))", 
          linewidth=2, color=:green, linestyle=:dash, alpha=0.7)
    plot!(p, t_grid, y_pred_total, label="Total Mixture", linewidth=3, color=:red)

    # Add vertical lines at peaks
    vline!(p, [max_lba1_rt], color=:orange, linestyle=:dot, linewidth=1, alpha=0.5, label="")
    vline!(p, [max_lba2_rt], color=:green, linestyle=:dot, linewidth=1, alpha=0.5, label="")
    vline!(p, [max_total_rt], color=:red, linestyle=:dot, linewidth=1, alpha=0.5, label="")

    savefig(p, output_plot)
    println("Saved plot to $output_plot")
    println("  Mixing probability: $(round(p_mix, digits=4))")
    println("  LBA1 (fast) - t0: $(round(t0_1, digits=3))s, peak at: $(round(max_lba1_rt, digits=3))s")
    println("  LBA2 (slow) - t0: $(round(t0_2, digits=3))s, peak at: $(round(max_lba2_rt, digits=3))s")
    println("  Component separation: $(round(abs(max_lba1_rt - max_lba2_rt)*1000, digits=0))ms")
end

end # module
