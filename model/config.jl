# ==========================================================================
# Configuration Module
# Settings and flags for model fitting and plotting
# ==========================================================================

module Config

export ModelConfig, SingleLBAParams, DualLBAParams, DataConfig, OptimizationConfig
export get_default_single_params, get_default_dual_params, get_data_config, get_optimization_config, get_weighting_mode

"""
    ModelConfig

Configuration settings for model fitting and plotting.

Fields:
- show_target_choice::Bool - Show target choice (highest reward) line in plots
- show_distractor_choice::Bool - Show distractor choice lines in plots
"""
struct ModelConfig
    show_target_choice::Bool
    show_distractor_choice::Bool
end

# Default constructor with all flags enabled
ModelConfig() = ModelConfig(true, true)

"""
    SingleLBAParams

Parameter bounds and initial values for single LBA model.

Fields:
- lower::Vector{Float64} - Lower bounds [C, w_slope, A, k, t0]
- upper::Vector{Float64} - Upper bounds [C, w_slope, A, k, t0]
- x0::Vector{Float64} - Initial values [C, w_slope, A, k, t0]
"""
struct SingleLBAParams
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

"""
    DualLBAParams

Parameter bounds and initial values for dual LBA mixture model.

Fields:
- lower::Vector{Float64} - Lower bounds [C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix]
- upper::Vector{Float64} - Upper bounds [C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix]
- x0::Vector{Float64} - Initial values [C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix]
"""
struct DualLBAParams
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

# Default weighting mode for reward transforms (either :exponential or :free)
const DEFAULT_WEIGHTING_MODE = :free

"""
    get_weighting_mode()

Returns the default weighting mode for reward-to-weight transform.
Change `DEFAULT_WEIGHTING_MODE` to `:free` to estimate separate weights per reward value.
"""
get_weighting_mode() = DEFAULT_WEIGHTING_MODE

"""
    get_default_single_params()

Returns default parameter bounds and initial values for single LBA model.
Parameters: [C, w_slope, A, k, t0]
"""
function get_default_single_params(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE)
    if weighting_mode == :exponential
        lower = [1.0,  0.0,   0.01, 0.05, 0.05]   # C, w_slope, A, k, t0
        upper = [30.0, 10.0,  1.0,  1.0,  0.6]
        x0    = [10.0, 1.0,   0.2,  0.2,  0.25]
    elseif weighting_mode == :free
        # Free weights: [C, w2, w3, w4, A, k, t0] with w1 fixed at 1.0 baseline
        lower = [1.0,  1.0, 1.0, 1.0,   0.01, 0.05, 0.05]
        upper = [30.0, 50.0, 50.0, 50.0, 1.0,  1.0,  0.6]
        x0    = [10.0, 2.0,  5.0, 10.0, 0.2,  0.2,  0.25]
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free")
    end

    return SingleLBAParams(lower, upper, x0)
end

"""
    get_default_dual_params()

Returns default parameter bounds and initial values for dual LBA mixture model.
Parameters: [C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix]
Component 1 (fast): lower thresholds, faster t0
Component 2 (slow): higher thresholds, slower t0
"""
function get_default_dual_params()
    lower = [1.0,  0.0,   0.01, 0.05, 0.05,  0.01, 0.05, 0.15,  0.0]
    upper = [30.0, 10.0,  1.0,  1.0,  0.4,   1.0,  1.0,  0.6,   0.99]
    x0    = [10.0, 1.0,   0.2,  0.2,  0.2,   0.3,  0.3,  0.35,  0.4]
    return DualLBAParams(lower, upper, x0)
end

"""
    DataConfig

Data loading configuration for participant selection.

Fields:
- participant_id::Int - Participant ID (1, 2, or 3)
- data_base_path::String - Base path to data directory
- file_pattern::String - Pattern for matching data files (default: "*.dat")
"""
struct DataConfig
    participant_id::Int
    data_base_path::String
    file_pattern::String
end

# Default constructor with validation
function DataConfig(participant_id::Int;
                   data_base_path::String=joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003"),
                   file_pattern::String="*.dat")
    if !(participant_id in [1, 2, 3])
        error("Invalid participant_id: $participant_id. Must be 1, 2, or 3.")
    end
    return new(participant_id, data_base_path, file_pattern)
end

"""
    get_data_config(participant_id::Int)

Returns data configuration for the specified participant.

Arguments:
- participant_id::Int - Participant ID (1, 2, or 3)

Returns:
- DataConfig with paths and settings for the specified participant
"""
function get_data_config(participant_id::Int)
    if !(participant_id in [1, 2, 3])
        error("Invalid participant_id: $participant_id. Must be 1, 2, or 3.")
    end

    # Construct data path based on participant ID
    # Folder structure: ../data/ParticipantCPP002-00X/ParticipantCPP002-00X
    # where X is the participant number (1, 2, or 3)
    participant_folder_name = "ParticipantCPP002-00$(participant_id)"
    data_path = joinpath("..", "data", participant_folder_name, participant_folder_name)

    return DataConfig(participant_id, data_path, "*.dat")
end

"""
    OptimizationConfig

Configuration settings for optimization tolerances and limits.

Fields:
- g_tol::Float64 - Gradient tolerance (smaller = more precise but slower)
- f_reltol::Float64 - Relative function tolerance (stop when improvement < this %)
- x_reltol::Float64 - Relative parameter tolerance
- max_iterations::Int - Maximum number of iterations
- time_limit::Float64 - Maximum time in seconds

Adjust these values to trade off between speed and precision:
- For faster fitting: increase tolerances (e.g., g_tol=1e-2, f_reltol=1e-3)
- For more precise fitting: decrease tolerances (e.g., g_tol=1e-4, f_reltol=1e-5)
"""
struct OptimizationConfig
    g_tol::Float64
    f_reltol::Float64
    x_reltol::Float64
    max_iterations::Int
    time_limit::Float64
end

"""
    get_optimization_config()

Returns default optimization configuration.

Current settings are optimized for SPEED with acceptable precision.
You can modify these values here to change optimization behavior globally.
"""
function get_optimization_config()
    return OptimizationConfig(
        1e-2,      # g_tol: Very relaxed gradient tolerance for speed
        1e-3,      # f_reltol: Stop when improvement < 0.1%
        1e-3,      # x_reltol: Relaxed parameter tolerance
        300,       # max_iterations: Lower cap for faster termination
        600.0      # time_limit: Maximum time in seconds
    )
end

end # module
