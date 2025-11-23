# ==========================================================================
# Configuration for Dual-Mode LBA (Mixture of Two LBAs with Shared Weights)
# Keeps defaults separate from the main config.jl to avoid cross-contamination.
# ==========================================================================

module ConfigDualModes

export ModelConfig, DualModesParams, DataConfig
export get_default_dualmodes_params, get_data_config, get_weighting_mode, get_plot_config, cue_condition_type
export PARTICIPANT_ID_DUALMODES, OUTPUT_CSV_DUALMODES, OUTPUT_PLOT_DUALMODES
export WEIGHTING_MODE_DUALMODES

import Base.Filesystem: joinpath
using Base: @assert

struct ModelConfig
    show_target_choice::Bool
    show_distractor_choice::Bool
end

const SHOW_TARGET_CHOICE_IN_PLOTS = false
const SHOW_DISTRACTOR_CHOICE_IN_PLOTS = false
ModelConfig() = ModelConfig(SHOW_TARGET_CHOICE_IN_PLOTS, SHOW_DISTRACTOR_CHOICE_IN_PLOTS)
get_plot_config()::ModelConfig = ModelConfig()

# Participant / IO
const PARTICIPANT_ID_DUALMODES = 3
const OUTPUT_CSV_DUALMODES = joinpath(@__DIR__, "outputdata", "model_fit_results_dualmodes_P$(PARTICIPANT_ID_DUALMODES).csv")
const OUTPUT_PLOT_DUALMODES = "model_fit_plot_dualmodes.png"

# Weighting mode (support :free or :exponential)
const WEIGHTING_MODE_DUALMODES = :free
get_weighting_mode() = WEIGHTING_MODE_DUALMODES

# Parameter container
struct DualModesParams
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

# Minimal DataConfig copy (decoupled from main config)
struct DataConfig
    participant_id::Int
    data_base_path::String
    file_pattern::String
end

const DATA_BASE_DIR = joinpath(@__DIR__, "..", "data")

function cue_condition_type(cc)::Symbol
    single_set = Set([1,2,3,4])
    double_set = Set([5,6,7,8,9,10])
    if cc in single_set
        return :single
    elseif cc in double_set
        return :double
    else
        error("Unexpected CueCondition $cc")
    end
end

# Returns parameter bounds/starts for [C_fast, C_slow, w2, w3, w4, A, k_fast, k_slow, t0, pi_single, pi_double] (free mode)
# or [C_fast, C_slow, w_slope, A, k_fast, k_slow, t0, pi_single, pi_double] (exponential mode)
function get_default_dualmodes_params(weighting_mode::Symbol=:free)::DualModesParams
    if weighting_mode == :free
        lower = [1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.05, 0.05, 0.05, 0.01, 0.01]
        upper = [30.0, 30.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 0.6, 0.99, 0.99]
        x0    = [10.0, 8.0, 2.0, 5.0, 10.0, 0.2, 0.2, 0.3, 0.25, 0.9, 0.2]
    elseif weighting_mode == :exponential
        lower = [1.0, 1.0, 0.0, 0.01, 0.05, 0.05, 0.05, 0.01, 0.01]
        upper = [30.0, 30.0, 10.0, 1.0, 1.0, 1.0, 0.6, 0.99, 0.99]
        x0    = [10.0, 8.0, 1.0, 0.2, 0.2, 0.3, 0.25, 0.9, 0.2]
    else
        error("Unknown weighting_mode: $weighting_mode")
    end
    return DualModesParams(lower, upper, x0)
end

function get_data_config(participant_id::Int)::DataConfig
    @assert participant_id in (1,2,3)
    folder = "ParticipantCPP002-00$(participant_id)"
    data_path = joinpath(DATA_BASE_DIR, folder, folder)
    return DataConfig(participant_id, data_path, "*.dat")
end

end # module
