# ==========================================================================
# Configuration Module
# Settings and flags for model fitting and plotting
# ==========================================================================

module Config

export ModelConfig, SingleLBAParams, DualLBAParams, DataConfig, OptimizationConfig
export get_default_single_params, get_default_dual_params, get_data_config, get_optimization_config, get_weighting_mode, get_plot_config
export ACCURACY_YLIM, SAVE_INDIVIDUAL_CONDITION_PLOTS, SHOW_TARGET_CHOICE_IN_PLOTS, SHOW_DISTRACTOR_CHOICE_IN_PLOTS
export RT_ALLCONDITIONS_YLIM, AXIS_FONT_SIZE
export DATA_BASE_DIR, DATA_PATH, FILE_PATTERN
export OUTPUTDATA_DIR, IMAGES_DIR
export OUTPUT_CSV_MIXTURE, OUTPUT_PLOT_MIXTURE
export PARTICIPANT_ID_SINGLE, OUTPUT_CSV_SINGLE, OUTPUT_PLOT_SINGLE
export PARTICIPANT_ID_DUAL, OUTPUT_CSV_DUAL, OUTPUT_PLOT_DUAL
export PARTICIPANT_ID_ALLCONDITIONS, WEIGHTING_MODE_OVERRIDE_ALLCONDITIONS, OUTPUT_CSV_ALLCONDITIONS, OUTPUT_PLOT_ALLCONDITIONS

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

# ==========================================================================
# Centralized boolean flags for model/plot behavior
# Toggle these values to control global behaviors in one place.
# ==========================================================================
const SHOW_TARGET_CHOICE_IN_PLOTS = false
const SHOW_DISTRACTOR_CHOICE_IN_PLOTS = false
const SAVE_INDIVIDUAL_CONDITION_PLOTS = false

# Default constructor pulls values from the centralized flags above
ModelConfig() = ModelConfig(SHOW_TARGET_CHOICE_IN_PLOTS, SHOW_DISTRACTOR_CHOICE_IN_PLOTS)
get_plot_config()::ModelConfig = ModelConfig()

# ==========================================================================
# Plot appearance settings
# Centralize y-limits and font sizing for all generated plots.
# ==========================================================================
const RT_ALLCONDITIONS_YLIM = (0.0, 10.5)
const AXIS_FONT_SIZE = 12

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

# Y-limits for accuracy plots (observed vs predicted)
# Adjust here to change the vertical range of all accuracy figures
const ACCURACY_YLIM = (0.5, 1.02)

# ==========================================================================
# Data and I/O settings for the main scripts
# Update participant IDs, paths, and filenames here to change runs globally.
# ==========================================================================
const DATA_BASE_DIR = joinpath(@__DIR__, "..", "data")
const DATA_PATH = joinpath(DATA_BASE_DIR, "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"
const OUTPUTDATA_DIR = joinpath(@__DIR__, "outputdata")
const IMAGES_DIR = joinpath(@__DIR__, "images")

# Mixture (fit-mis-model.jl)
const OUTPUT_CSV_MIXTURE = joinpath(@__DIR__, "outputdata", "model_fit_results.csv")
const OUTPUT_PLOT_MIXTURE = "model_fit_plot.png"

# Single (fit-mis-model-single.jl)
const PARTICIPANT_ID_SINGLE = 3  # Options: 1, 2, or 3
const OUTPUT_CSV_SINGLE = joinpath(@__DIR__, "outputdata", "model_fit_results_single_P$(PARTICIPANT_ID_SINGLE).csv")
const OUTPUT_PLOT_SINGLE = "model_fit_plot_single.png"

# Dual (fit-mis-model-dual.jl)
const PARTICIPANT_ID_DUAL = 2  # Options: 1, 2, or 3
const OUTPUT_CSV_DUAL = joinpath(@__DIR__, "outputdata", "model_fit_results_dual_P$(PARTICIPANT_ID_DUAL).csv")
const OUTPUT_PLOT_DUAL = "model_fit_plot_dual.png"

# All-conditions (fit-mis-model-allconditions.jl)
const PARTICIPANT_ID_ALLCONDITIONS = 1  # Options: 1, 2, or 3
const WEIGHTING_MODE_OVERRIDE_ALLCONDITIONS = nothing  # leave as `nothing` to use DEFAULT_WEIGHTING_MODE
const OUTPUT_CSV_ALLCONDITIONS = "model_fit_results_allconditions.csv"
const OUTPUT_PLOT_ALLCONDITIONS = "model_fit_plot_allconditions.png"

"""
    get_weighting_mode()::Symbol

Returns the default weighting mode for reward-to-weight transform.
Change `DEFAULT_WEIGHTING_MODE` to `:free` to estimate separate weights per reward value.
"""
get_weighting_mode()::Symbol = DEFAULT_WEIGHTING_MODE

"""
    get_default_single_params(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE)::SingleLBAParams

Returns default parameter bounds and initial values for single LBA model.
Parameters: [C, w_slope, A, k, t0]
"""
function get_default_single_params(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE)::SingleLBAParams
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
    get_default_dual_params()::DualLBAParams

Returns default parameter bounds and initial values for dual LBA mixture model.
Parameters: [C, w_slope, A1, k1, t0_1, A2, k2, t0_2, p_mix]
Component 1 (fast): lower thresholds, faster t0
Component 2 (slow): higher thresholds, slower t0
"""
function get_default_dual_params()::DualLBAParams
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
                   data_base_path::String=joinpath(DATA_BASE_DIR, "ParticipantCPP002-003", "ParticipantCPP002-003"),
                   file_pattern::String="*.dat")
    if !(participant_id in [1, 2, 3])
        error("Invalid participant_id: $participant_id. Must be 1, 2, or 3.")
    end
    return new(participant_id, data_base_path, file_pattern)
end

"""
    get_data_config(participant_id::Int)::DataConfig

Returns data configuration for the specified participant.

Arguments:
- participant_id::Int - Participant ID (1, 2, or 3)

Returns:
- DataConfig with paths and settings for the specified participant
"""
function get_data_config(participant_id::Int)::DataConfig
    if !(participant_id in [1, 2, 3])
        error("Invalid participant_id: $participant_id. Must be 1, 2, or 3.")
    end

    # Construct data path based on participant ID
    # Folder structure: ../data/ParticipantCPP002-00X/ParticipantCPP002-00X
    # where X is the participant number (1, 2, or 3)
    participant_folder_name = "ParticipantCPP002-00$(participant_id)"
    data_path = joinpath(DATA_BASE_DIR, participant_folder_name, participant_folder_name)

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
    get_optimization_config()::OptimizationConfig

Returns default optimization configuration.

Current settings are optimized for SPEED with acceptable precision.
You can modify these values here to change optimization behavior globally.
"""
function get_optimization_config()::OptimizationConfig
    return OptimizationConfig(
        1e-2,      # g_tol: Very relaxed gradient tolerance for speed
        1e-3,      # f_reltol: Stop when improvement < 0.1%
        1e-3,      # x_reltol: Relaxed parameter tolerance
        300,       # max_iterations: Lower cap for faster termination
        600.0      # time_limit: Maximum time in seconds
    )
end

end # module
