# ==========================================================================
# Profile Model Performance
# ==========================================================================

using Pkg
using Profile
using ProfileView
using DataFrames

# Load utility modules
include("data_utils.jl")
include("model_utils.jl")
include("fitting_utils.jl")
include("config.jl")

using .DataUtils
using .ModelUtils
using .FittingUtils
using .Config

# Load data
data_config = get_data_config(1)
println("Loading data...")
data = load_and_process_data(data_config.data_base_path, data_config.file_pattern)
println("Loaded $(nrow(data)) trials")

# Get r_max
global r_max = 0.0
for rewards in data.ParsedRewards
    if !isempty(rewards)
        global r_max = max(r_max, maximum(rewards))
    end
end
println("r_max: $r_max")

# Test parameters
params_config = get_default_single_params()
test_params = params_config.x0
println("Test parameters: $test_params")

# Warm up
println("\nWarming up...")
for i in 1:3
    mis_lba_allconditions_loglike(test_params, data; r_max=r_max)
end

# Profile
println("\nProfiling likelihood function (10 iterations)...")
Profile.clear()
@profile begin
    for i in 1:10
        mis_lba_allconditions_loglike(test_params, data; r_max=r_max)
    end
end

# Print profile results
println("\nProfile Results:")
Profile.print(format=:flat, sortedby=:count, maxdepth=20)

println("\n" * "="^70)
println("Checking unique reward configurations...")
unique_rewards = unique(data.ParsedRewards)
println("Number of unique reward configurations: $(length(unique_rewards))")
println("Total trials: $(nrow(data))")
println("Cache hit ratio potential: $(round((1 - length(unique_rewards)/nrow(data)) * 100, digits=1))%")

# Time one call
println("\n" * "="^70)
println("Timing single likelihood evaluation...")
@time result = mis_lba_allconditions_loglike(test_params, data; r_max=r_max)
println("Negative log-likelihood: $result")

# Time 10 calls
println("\n" * "="^70)
println("Timing 10 likelihood evaluations...")
@time begin
    for i in 1:10
        mis_lba_allconditions_loglike(test_params, data; r_max=r_max)
    end
end
