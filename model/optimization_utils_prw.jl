# ==========================================================================
# Optimization Utilities for PRW (independent of main Config)
# ==========================================================================

using DataFrames
using Optim
using ADTypes

# Note: config_prw.jl must be included before this file

"""
    fit_model_prw(data, objective_func; lower, upper, x0, time_limit=nothing, r_max=nothing)

Thin wrapper around Optim with PRW-specific config to avoid dependency on main Config.
"""
function fit_model_prw(data::Union{DataFrame,Any}, objective_func::Function;
                   lower::Vector{Float64}, upper::Vector{Float64}, x0::Vector{Float64},
                   time_limit::Union{Nothing,Float64}=nothing, r_max::Union{Nothing,Float64}=nothing)::Optim.MultivariateOptimizationResults
    println("Fitting PRW model (this may take a minute)...")

    opt_config = get_optimization_config_prw()
    actual_time_limit = isnothing(time_limit) ? opt_config.time_limit : time_limit

    func = isnothing(r_max) ? x -> objective_func(x, data) : x -> objective_func(x, data; r_max=r_max)

    opt_options = Optim.Options(
        time_limit = actual_time_limit,
        show_trace = false,
        g_tol = opt_config.g_tol,
        f_reltol = opt_config.f_reltol,
        x_reltol = opt_config.x_reltol,
        iterations = opt_config.max_iterations
    )

    println("Optimization settings:")
    println("  g_tol: $(opt_config.g_tol)")
    println("  f_reltol: $(opt_config.f_reltol)")
    println("  max_iterations: $(opt_config.max_iterations)")
    println("  time_limit: $(actual_time_limit)s")

    res = optimize(func, lower, upper, x0, Fminbox(LBFGS()), opt_options; autodiff=ADTypes.AutoFiniteDiff())

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
