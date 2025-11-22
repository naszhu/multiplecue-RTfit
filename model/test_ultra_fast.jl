# Test ultra-fast preprocessing approach

using DataFrames

include("data_utils.jl")
include("model_utils.jl")
include("model_utils_fast.jl")
include("config.jl")

using .DataUtils
using .ModelUtils
using .ModelUtilsFast
using .Config
using Optim

# Load data
data_config = get_data_config(1)
println("Loading data...")
data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)
println("Loaded $(nrow(data)) trials")

global r_max = 4.0
params_config = get_default_single_params()
test_params = params_config.x0

println("\n" * "="^70)
println("SPEED COMPARISON")
println("="^70)

# Test original version
println("\n1. Original (with caching)")
@time begin
    for i in 1:10
        mis_lba_allconditions_loglike(test_params, data; r_max=r_max)
    end
end

# Test ultra-fast version
println("\n2. Ultra-fast (preprocessed)")
println("   Preprocessing data...")
@time preprocessed = preprocess_data_for_fast_fitting(data)
println("   Number of unique reward configs: $(length(preprocessed.unique_rewards))")
println("   Running likelihood (10x)...")
@time begin
    for i in 1:10
        mis_lba_allconditions_loglike_fast(test_params, preprocessed; r_max=r_max)
    end
end

# Verify they give same result
ll_original = mis_lba_allconditions_loglike(test_params, data; r_max=r_max)
ll_fast = mis_lba_allconditions_loglike_fast(test_params, preprocessed; r_max=r_max)

println("\n" * "="^70)
println("VERIFICATION")
println("="^70)
println("Original LogLik: $ll_original")
println("Fast LogLik:     $ll_fast")
println("Difference:      $(abs(ll_original - ll_fast))")

# Test full optimization with ultra-fast version
println("\n" * "="^70)
println("FULL OPTIMIZATION TEST (Ultra-Fast)")
println("="^70)

lower = params_config.lower
upper = params_config.upper
x0 = params_config.x0

func_fast = x -> mis_lba_allconditions_loglike_fast(x, preprocessed; r_max=r_max)

opt_config = get_optimization_config()
opt_options = Optim.Options(
    time_limit = 60.0,
    g_tol = opt_config.g_tol,
    f_reltol = opt_config.f_reltol,
    x_reltol = opt_config.x_reltol,
    iterations = opt_config.max_iterations
)

@time res = optimize(func_fast, lower, upper, x0, Fminbox(LBFGS()), opt_options; autodiff=:forward)

println("\nResults:")
println("  Converged: $(Optim.converged(res))")
println("  Iterations: $(Optim.iterations(res))")
println("  F-calls: $(Optim.f_calls(res))")
println("  LogLik: $(round(Optim.minimum(res), digits=2))")
println("  Params: $(Optim.minimizer(res))")
