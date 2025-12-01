# Quick test of optimization speed with different settings

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

# Get r_max
global r_max = 4.0

# Get initial parameters
params_config = get_default_single_params()
lower = params_config.lower
upper = params_config.upper
x0 = params_config.x0

println("\nTesting different optimization strategies...")
println("="^70)

# Test 1: Current settings (relaxed tolerance)
println("\n1. BFGS with relaxed tolerance (current)")
func = x -> mis_lba_allconditions_loglike(x, data; r_max=r_max)
opt_options = Optim.Options(
    time_limit = 60.0,
    show_trace = false,
    g_tol = 1e-4,
    f_tol = 1e-6,
    x_tol = 1e-6,
    iterations = 1000
)
@time res1 = optimize(func, lower, upper, x0, Fminbox(BFGS()), opt_options; autodiff=:forward)
println("  Converged: $(Optim.converged(res1))")
println("  Iterations: $(Optim.iterations(res1))")
println("  F-calls: $(Optim.f_calls(res1))")
println("  LogLik: $(Optim.minimum(res1))")

# Test 2: Even more relaxed
println("\n2. BFGS with very relaxed tolerance")
opt_options2 = Optim.Options(
    time_limit = 60.0,
    show_trace = false,
    g_tol = 1e-3,      # Even more relaxed
    f_tol = 1e-5,
    x_tol = 1e-5,
    iterations = 500    # Fewer iterations
)
@time res2 = optimize(func, lower, upper, x0, Fminbox(BFGS()), opt_options2; autodiff=:forward)
println("  Converged: $(Optim.converged(res2))")
println("  Iterations: $(Optim.iterations(res2))")
println("  F-calls: $(Optim.f_calls(res2))")
println("  LogLik: $(Optim.minimum(res2))")

# Test 3: NelderMead (derivative-free, might be faster)
println("\n3. NelderMead (derivative-free)")
opt_options3 = Optim.Options(
    time_limit = 60.0,
    show_trace = false,
    iterations = 500
)
@time res3 = optimize(func, x0, NelderMead(), opt_options3)
println("  Converged: $(Optim.converged(res3))")
println("  Iterations: $(Optim.iterations(res3))")
println("  F-calls: $(Optim.f_calls(res3))")
println("  LogLik: $(Optim.minimum(res3))")

# Test 4: L-BFGS (uses less memory, can be faster)
println("\n4. L-BFGS with relaxed tolerance")
opt_options4 = Optim.Options(
    time_limit = 60.0,
    show_trace = false,
    g_tol = 1e-4,
    f_tol = 1e-6,
    iterations = 500
)
@time res4 = optimize(func, lower, upper, x0, Fminbox(LBFGS()), opt_options4; autodiff=:forward)
println("  Converged: $(Optim.converged(res4))")
println("  Iterations: $(Optim.iterations(res4))")
println("  F-calls: $(Optim.f_calls(res4))")
println("  LogLik: $(Optim.minimum(res4))")

println("\n" * "="^70)
println("COMPARISON:")
println("Method 1 (BFGS relaxed):      LogLik=$(round(Optim.minimum(res1), digits=2)), Time shown above")
println("Method 2 (BFGS very relaxed): LogLik=$(round(Optim.minimum(res2), digits=2)), Time shown above")
println("Method 3 (NelderMead):        LogLik=$(round(Optim.minimum(res3), digits=2)), Time shown above")
println("Method 4 (L-BFGS):            LogLik=$(round(Optim.minimum(res4), digits=2)), Time shown above")
