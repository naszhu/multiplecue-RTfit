# Final speed test with all optimizations

using DataFrames

include("data_utils.jl")
include("model_utils.jl")
include("fitting_utils.jl")
include("config.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils
using .Config
using Optim

# Load data
data_config = get_data_config(1)
println("Loading data...")
data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)

global r_max = 4.0
params_config = get_default_single_params()

println("\n" * "="^70)
println("FINAL OPTIMIZATIONS TEST")
println("="^70)

# Preprocess
println("\nPreprocessing...")
@time preprocessed = preprocess_data_for_fitting(data)

# Run optimization
println("\nRunning full optimization...")
@time result = fit_model(preprocessed, mis_lba_allconditions_loglike;
                   lower=params_config.lower, upper=params_config.upper,
                   x0=params_config.x0, r_max=r_max)

println("\n" * "="^70)
println("RESULTS:")
println("  Converged: $(Optim.converged(result))")
println("  Iterations: $(Optim.iterations(result))")
println("  F-calls: $(Optim.f_calls(result))")
println("  LogLik: $(round(Optim.minimum(result), digits=2))")
