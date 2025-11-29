# Time the full optimization process

include("data_utils.jl")
include("model_utils.jl")
include("fitting_utils.jl")
include("config.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils
using .Config
using Optim
using DataFrames

# Get data configuration
data_config = get_data_config(1)
println("Loading data...")
@time data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)

# Get r_max
global r_max = 0.0
for rewards in data.ParsedRewards
    if !isempty(rewards)
        global r_max = max(r_max, maximum(rewards))
    end
end
println("r_max: $r_max")

# Get parameter bounds
params_config = get_default_single_params()
lower = params_config.lower
upper = params_config.upper
x0 = params_config.x0

println("\n" * "="^70)
println("TIMING FULL OPTIMIZATION")
println("="^70)

@time result = fit_model(data, mis_lba_allconditions_loglike;
                   lower=lower, upper=upper, x0=x0, time_limit=600.0, r_max=r_max)

println("\n" * "="^70)
println("FINAL RESULTS")
println("="^70)
println("Converged: $(Optim.converged(result))")
println("Iterations: $(Optim.iterations(result))")
println("Function calls: $(Optim.f_calls(result))")
println("Best LogLik: $(Optim.minimum(result))")
println("Best params: $(Optim.minimizer(result))")
