# ==========================================================================
# Configuration for Dual-Mode LBA (Mixture of Two LBAs with Shared Weights)
# Keeps defaults separate from the main config.jl to avoid cross-contamination.
# ==========================================================================

module ConfigDualModes

export ModelConfig, DualModesParams, DataConfig
export get_default_dualmodes_params, get_data_config, get_weighting_mode, get_plot_config
export PARTICIPANT_ID_DUALMODES, OUTPUT_CSV_DUALMODES, OUTPUT_PLOT_DUALMODES
export WEIGHTING_MODE_DUALMODES

import Main.Config
using Main.Config: DataConfig, DATA_BASE_DIR, cue_condition_type  # reuse DataConfig and base path

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

get_data_config(participant_id::Int)::DataConfig = Config.get_data_config(participant_id)

end # module
