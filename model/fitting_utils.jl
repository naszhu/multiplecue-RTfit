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

export fit_model, save_results, generate_plot

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
    save_results(result, output_csv="model_fit_results.csv")

    Saves the optimization results to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename

    Returns parameter names and values as a DataFrame.
"""
function save_results(result, output_csv="model_fit_results.csv")
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)",
                     "ThreshGap(k)", "NonDec(t0)", "ProbExp", "MuExp", "SigExp"],
        Value = best
    )

    CSV.write(output_csv, results_df)
    println("Saved parameters to $output_csv")

    return results_df
end

"""
    generate_plot(data::DataFrame, params, output_plot="model_fit_plot.png")

    Generates a plot comparing observed RT distribution with model predictions.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - params: Vector of model parameters [C, w, A, k, t0, p_exp, mu_exp, sig_exp]
    - output_plot: Output filename for plot
"""
function generate_plot(data::DataFrame, params, output_plot="model_fit_plot.png")
    println("Generating plot...")

    # Unpack parameters
    C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp = params

    # Histogram of Observed RT
    histogram(data.CleanRT, normalize=true, label="Observed", alpha=0.5, bins=60,
              xlabel="Reaction Time (s)", ylabel="Density", title="MIS-LBA Mixture Fit",
              color=:blue, legend=:topright)

    # Simulate Model Curve
    # We calculate the 'average' predicted PDF across all trials
    t_grid = range(0.05, 1.5, length=200)
    y_pred = zeros(length(t_grid))

    # Use a random subset of trials to approximate the curve
    n_samples = min(200, nrow(data))
    subset_indices = rand(1:nrow(data), n_samples)

    for (j, t) in enumerate(t_grid)
        avg_pdf = 0.0
        for i in subset_indices
            rewards = data.ParsedRewards[i]

            # Reconstruct parameters
            ws = 1.0 .+ (w_slope .* rewards)
            vs = C .* (ws ./ sum(ws))

            # LBA PDF (summed over all choices)
            lba = LBA(ν=vs, A=A, k=k, τ=t0)

            # Check if t > t0 for LBA
            lba_dens = 0.0
            if t > t0
                # Sum pdf of all possible choices
                lba_dens = sum([pdf(lba, (choice=c, rt=t)) for c in 1:length(vs)])
            end

            # Express PDF
            exp_dens = pdf(Normal(mu_exp, sig_exp), t)

            # Mixture
            avg_pdf += (p_exp * exp_dens) + ((1-p_exp) * lba_dens)
        end
        y_pred[j] = avg_pdf / n_samples
    end

    plot!(t_grid, y_pred, label="Model Prediction", linewidth=3, color=:red)

    savefig(output_plot)
    println("Saved plot to $output_plot")
end

end # module
