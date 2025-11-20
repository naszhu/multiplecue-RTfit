# ==========================================================================
# Configuration Module
# Settings and flags for model fitting and plotting
# ==========================================================================

module Config

export ModelConfig, SingleLBAParams, DualLBAParams, DataConfig
export get_default_single_params, get_default_dual_params, get_data_config

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

"""
    get_default_single_params()

Returns default parameter bounds and initial values for single LBA model.
Parameters: [C, w_slope, A, k, t0]
"""
function get_default_single_params()
    lower = [1.0,  0.0,   0.01, 0.05, 0.05]
    upper = [30.0, 10.0,  1.0,  1.0,  0.6]
    x0    = [10.0, 1.0,   0.2,  0.2,  0.25]
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

end # module
