# Test advanced optimization strategies

using DataFrames

include("data_utils.jl")
include("model_utils.jl")
include("config.jl")

using .DataUtils
using .ModelUtils
using .Config
using Optim

# Load data
data_config = get_data_config(1)
println("Loading data...")
data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)

global r_max = 4.0
params_config = get_default_single_params()
lower = params_config.lower
upper = params_config.upper
x0 = params_config.x0

func = x -> mis_lba_allconditions_loglike(x, data; r_max=r_max)

println("\nTesting advanced strategies...")
println("="^70)

# Strategy 1: Multi-start with quick runs
println("\n1. Multi-start approach (5 random starts, quick L-BFGS)")
println("   Better for avoiding local minima")
global best_result = nothing
global best_ll = Inf

@time begin
    for i in 1:5
        # Random perturbation of initial point
        x_start = x0 .+ (rand(length(x0)) .- 0.5) .* (upper .- lower) .* 0.2
        x_start = max.(lower, min.(upper, x_start))  # Keep in bounds

        opt_options = Optim.Options(
            time_limit = 10.0,  # Only 10 seconds per start
            g_tol = 1e-3,
            iterations = 200
        )

        res = optimize(func, lower, upper, x_start, Fminbox(LBFGS()), opt_options; autodiff=:forward)

        if Optim.minimum(res) < best_ll
            global best_ll = Optim.minimum(res)
            global best_result = res
        end
        println("  Start $i: LogLik = $(round(Optim.minimum(res), digits=2))")
    end
end
println("  Best LogLik: $(round(best_ll, digits=2))")

# Strategy 2: Two-phase optimization (coarse then fine)
println("\n2. Two-phase optimization (coarse NelderMead → fine L-BFGS)")
@time begin
    # Phase 1: Coarse search with NelderMead (no gradients needed)
    opt1 = Optim.Options(iterations = 100, show_trace = false)
    res1 = optimize(func, x0, NelderMead(), opt1)
    x_coarse = Optim.minimizer(res1)
    println("  Phase 1 (coarse): LogLik = $(round(Optim.minimum(res1), digits=2))")

    # Phase 2: Fine-tune with L-BFGS
    opt2 = Optim.Options(g_tol = 1e-4, iterations = 200)
    res2 = optimize(func, lower, upper, x_coarse, Fminbox(LBFGS()), opt2; autodiff=:forward)
    println("  Phase 2 (fine):   LogLik = $(round(Optim.minimum(res2), digits=2))")
end

# Strategy 3: L-BFGS with more aggressive settings
println("\n3. L-BFGS with aggressive early stopping")
opt_options = Optim.Options(
    g_tol = 1e-3,        # Very relaxed gradient tolerance
    f_reltol = 1e-4,     # Stop when improvement < 0.01%
    iterations = 300,
    show_trace = false
)
@time res3 = optimize(func, lower, upper, x0, Fminbox(LBFGS()), opt_options; autodiff=:forward)
println("  Converged: $(Optim.converged(res3))")
println("  Iterations: $(Optim.iterations(res3))")
println("  F-calls: $(Optim.f_calls(res3))")
println("  LogLik: $(round(Optim.minimum(res3), digits=2))")

# Strategy 4: Check if we can reduce data for initial search
println("\n4. Bootstrap approach (fit on subset, refine on full)")
@time begin
    # Use 50% of data for initial fit
    subset_size = div(nrow(data), 2)
    subset_indices = sort(rand(1:nrow(data), subset_size))
    data_subset = data[subset_indices, :]

    func_subset = x -> mis_lba_allconditions_loglike(x, data_subset; r_max=r_max)

    # Quick fit on subset
    opt_subset = Optim.Options(g_tol = 1e-3, iterations = 200)
    res_subset = optimize(func_subset, lower, upper, x0, Fminbox(LBFGS()), opt_subset; autodiff=:forward)
    x_subset = Optim.minimizer(res_subset)
    println("  Subset fit: LogLik = $(round(Optim.minimum(res_subset), digits=2))")

    # Refine on full data
    opt_full = Optim.Options(g_tol = 1e-4, iterations = 100)
    res_full = optimize(func, lower, upper, x_subset, Fminbox(LBFGS()), opt_full; autodiff=:forward)
    println("  Full fit:   LogLik = $(round(Optim.minimum(res_full), digits=2))")
end

println("\n" * "="^70)
println("SUMMARY:")
println("Strategy 1 (Multi-start): Best = $(round(best_ll, digits=2))")
println("Strategy 2 (Two-phase):   Final = $(round(Optim.minimum(res2), digits=2))")
println("Strategy 3 (Aggressive):  Final = $(round(Optim.minimum(res3), digits=2))")
println("Strategy 4 (Bootstrap):   Final = $(round(Optim.minimum(res_full), digits=2))")
