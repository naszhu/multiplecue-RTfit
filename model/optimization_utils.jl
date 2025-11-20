# ==========================================================================
# Optimization Utilities Module
# Model fitting and optimization functions
# ==========================================================================

module OptimizationUtils

using DataFrames
using Optim

export fit_model

"""
    fit_model(data::DataFrame, objective_func;
              lower, upper, x0, time_limit=600.0, r_max=nothing)

    Fits the model using optimization.

    Arguments:
    - data: DataFrame with CleanRT, Choice, and ParsedRewards columns
    - objective_func: Function to minimize (typically a log-likelihood function)
    - lower: Vector of lower bounds for parameters
    - upper: Vector of upper bounds for parameters
    - x0: Vector of initial parameter values
    - time_limit: Maximum time in seconds for optimization
    - r_max: Optional maximum reward value to pass to objective_func

    Returns:
    - result: Optim optimization result object
"""
function fit_model(data::DataFrame, objective_func;
                   lower, upper, x0, time_limit=600.0, r_max=nothing)
    println("Fitting model (this may take a minute)...")

    # Create wrapper function that passes r_max if provided
    if isnothing(r_max)
        func = x -> objective_func(x, data)
    else
        func = x -> objective_func(x, data; r_max=r_max)
    end

    opt_options = Optim.Options(time_limit = time_limit, show_trace = false)

    res = optimize(func, lower, upper, x0, Fminbox(BFGS()), opt_options;
                   autodiff=:forward)

    best = Optim.minimizer(res)
    println("\n--- Optimization Complete ---")
    println("Best Params: $best")
    println("Min LogLikelihood: $(Optim.minimum(res))")

    return res
end

end # module
