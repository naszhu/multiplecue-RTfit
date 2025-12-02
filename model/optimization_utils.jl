# ==========================================================================
# Optimization Utilities
# Model fitting and optimization functions
# ==========================================================================

using DataFrames
using Optim

# Note: config.jl must be included before this file
# Note: model_utils.jl must be included before this file (for PreprocessedData)

"""
    fit_model(data::Union{DataFrame,PreprocessedData}, objective_func::Function;
              lower::Vector{Float64}, upper::Vector{Float64}, x0::Vector{Float64},
              time_limit::Union{Nothing,Float64}=nothing, r_max::Union{Nothing,Float64}=nothing)::Optim.MultivariateOptimizationResults

    Fits the model using optimization.

    Arguments:
    - data: DataFrame or PreprocessedData structure
    - objective_func: Function to minimize (typically a log-likelihood function)
    - lower: Vector of lower bounds for parameters
    - upper: Vector of upper bounds for parameters
    - x0: Vector of initial parameter values
    - time_limit: Maximum time in seconds for optimization
    - r_max: Optional maximum reward value to pass to objective_func

    Returns:
    - result: Optim optimization result object
"""
function fit_model(data::Union{DataFrame,PreprocessedData}, objective_func::Function;
                   lower::Vector{Float64}, upper::Vector{Float64}, x0::Vector{Float64},
                   time_limit::Union{Nothing,Float64}=nothing, r_max::Union{Nothing,Float64}=nothing)::Optim.MultivariateOptimizationResults
    println("Fitting model (this may take a minute)...")

    # Get optimization configuration from config file
    opt_config = get_optimization_config()

    # Use provided time_limit or default from config
    actual_time_limit = isnothing(time_limit) ? opt_config.time_limit : time_limit

    # Create wrapper function that passes r_max if provided
    if isnothing(r_max)
        func = x -> objective_func(x, data)
    else
        func = x -> objective_func(x, data; r_max=r_max)
    end

    # Optimization settings from config file
    # These can be adjusted in config.jl to change behavior globally
    opt_options = Optim.Options(
        time_limit = actual_time_limit,
        show_trace = false,
        g_tol = opt_config.g_tol,           # From config: gradient tolerance
        f_reltol = opt_config.f_reltol,     # From config: relative function tolerance
        x_reltol = opt_config.x_reltol,     # From config: relative parameter tolerance
        iterations = opt_config.max_iterations  # From config: max iterations
    )

    println("Optimization settings:")
    println("  g_tol: $(opt_config.g_tol)")
    println("  f_reltol: $(opt_config.f_reltol)")
    println("  max_iterations: $(opt_config.max_iterations)")
    println("  time_limit: $(actual_time_limit)s")
    println("  (Adjust these in config.jl if needed)")

    # L-BFGS uses less memory and often converges faster than BFGS
    # Testing showed 3x speedup with equivalent results
    # LBA/pdf are not Dual-number friendly, so use finite-difference gradients
    res = optimize(func, lower, upper, x0, Fminbox(LBFGS()), opt_options;
                   autodiff=:finite)

    best = Optim.minimizer(res)
    println("\n--- Optimization Complete ---")
    println("Best Params: $best")
    println("Min LogLikelihood: $(Optim.minimum(res))")
    println("Converged: $(Optim.converged(res))")
    println("Iterations: $(Optim.iterations(res))")
    println("Function evaluations: $(Optim.f_calls(res))")
    println("Gradient evaluations: $(Optim.g_calls(res))")

    return res
end
