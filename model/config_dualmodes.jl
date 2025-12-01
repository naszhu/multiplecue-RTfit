# ==========================================================================
# Configuration for Dual-Mode LBA (Mixture of Two LBAs with Shared Weights)
# Fully self-contained (does not depend on main config)
# ==========================================================================

module ConfigDualModes

export ModelConfig, DualModesParams, DualModesLayout, DataConfig
export build_dualmodes_params, get_data_config, get_weighting_mode, get_plot_config, cue_condition_type
export PARTICIPANT_ID_DUALMODES, OUTPUT_CSV_DUALMODES, OUTPUT_PLOT_DUALMODES
export WEIGHTING_MODE_DUALMODES

using Base: @assert
import Base: joinpath

struct ModelConfig
    show_target_choice::Bool
    show_distractor_choice::Bool
end

const SHOW_TARGET_CHOICE_IN_PLOTS = false
const SHOW_DISTRACTOR_CHOICE_IN_PLOTS = false
ModelConfig() = ModelConfig(SHOW_TARGET_CHOICE_IN_PLOTS, SHOW_DISTRACTOR_CHOICE_IN_PLOTS)
get_plot_config()::ModelConfig = ModelConfig()

# Participant / IO
const PARTICIPANT_ID_DUALMODES = 2
const OUTPUT_CSV_DUALMODES = joinpath(@__DIR__, "outputdata", "model_fit_results_dualmodes_P$(PARTICIPANT_ID_DUALMODES).csv")
const OUTPUT_PLOT_DUALMODES = "model_fit_plot_dualmodes.png"

# Weighting mode (support :free or :exponential)
const WEIGHTING_MODE_DUALMODES = :free
get_weighting_mode() = WEIGHTING_MODE_DUALMODES

# Variation flags (mode vs cue)
const VARY_C_BY_MODE = true
const VARY_K_BY_MODE = true
const VARY_T0_BY_MODE = false
const VARY_A_BY_MODE = false

const VARY_C_BY_CUE = false
const VARY_K_BY_CUE = false
const VARY_T0_BY_CUE = false
const VARY_A_BY_CUE = false

const VARY_PI_BY_CUE = false  # mixture between LBAs varies by cue condition

# Contaminant flags, if use contamination or not
const USE_CONTAMINANT = false
const ESTIMATE_CONTAMINANT = false
const VARY_CONTAM_ALPHA_BY_MODE = false
const VARY_CONTAM_ALPHA_BY_CUE = false
const VARY_CONTAM_RT_BY_MODE = false
const VARY_CONTAM_RT_BY_CUE = false
const CONTAM_ALPHA_BOUNDS = (0.0, 0.2, 0.02)    # (lower, upper, x0)
const CONTAM_RT_BOUNDS    = (1.0, 4.0, 3.0)     # (lower, upper, x0) seconds

struct DualModesParams
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

struct DualModesLayout
    weighting_mode::Symbol
    vary_C_by_mode::Bool
    vary_C_by_cue::Bool
    vary_k_by_mode::Bool
    vary_k_by_cue::Bool
    vary_t0_by_mode::Bool
    vary_t0_by_cue::Bool
    vary_A_by_mode::Bool
    vary_A_by_cue::Bool
    vary_pi_by_cue::Bool
    use_contaminant::Bool
    estimate_contaminant::Bool
    vary_contam_alpha_by_mode::Bool
    vary_contam_alpha_by_cue::Bool
    vary_contam_rt_by_mode::Bool
    vary_contam_rt_by_cue::Bool
    idx_C::Dict{Symbol,Dict{Symbol,Int}}
    idx_k::Dict{Symbol,Dict{Symbol,Int}}
    idx_t0::Dict{Symbol,Dict{Symbol,Int}}
    idx_pi::Dict{Symbol,Int}
    idx_w::Dict{Symbol,Int}
    idx_A::Dict{Symbol,Dict{Symbol,Int}}
    idx_contam_alpha::Dict{Symbol,Dict{Symbol,Int}}
    idx_contam_rt::Dict{Symbol,Dict{Symbol,Int}}
    contam_alpha_fixed::Float64
    contam_rt_fixed::Float64
end

# Minimal DataConfig (decoupled from main config)
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

# builder
function build_dualmodes_params(weighting_mode::Symbol=get_weighting_mode();
    vary_C_by_mode::Bool=VARY_C_BY_MODE, vary_C_by_cue::Bool=VARY_C_BY_CUE,
    vary_k_by_mode::Bool=VARY_K_BY_MODE, vary_k_by_cue::Bool=VARY_K_BY_CUE,
    vary_t0_by_mode::Bool=VARY_T0_BY_MODE, vary_t0_by_cue::Bool=VARY_T0_BY_CUE,
    vary_A_by_mode::Bool=VARY_A_BY_MODE, vary_A_by_cue::Bool=VARY_A_BY_CUE,
    vary_pi_by_cue::Bool=VARY_PI_BY_CUE,
    use_contaminant::Bool=USE_CONTAMINANT, estimate_contaminant::Bool=ESTIMATE_CONTAMINANT,
    vary_contam_alpha_by_mode::Bool=VARY_CONTAM_ALPHA_BY_MODE, vary_contam_alpha_by_cue::Bool=VARY_CONTAM_ALPHA_BY_CUE,
    vary_contam_rt_by_mode::Bool=VARY_CONTAM_RT_BY_MODE, vary_contam_rt_by_cue::Bool=VARY_CONTAM_RT_BY_CUE)

    names = String[]
    lower = Float64[]
    upper = Float64[]
    x0    = Float64[]

    idx_C = Dict{Symbol,Dict{Symbol,Int}}()
    idx_k = Dict{Symbol,Dict{Symbol,Int}}()
    idx_t0= Dict{Symbol,Dict{Symbol,Int}}()
    idx_pi= Dict{Symbol,Int}()
    idx_w = Dict{Symbol,Int}()
    idx_A = Dict{Symbol,Dict{Symbol,Int}}()
    idx_contam_alpha = Dict{Symbol,Dict{Symbol,Int}}()
    idx_contam_rt    = Dict{Symbol,Dict{Symbol,Int}}()

    cue_keys(flag) = flag ? (:single, :double) : (:all,)
    mode_keys(flag)= flag ? (:fast, :slow) : (:shared,)
    function push_param!(n, lo, up, start)
        push!(names, n); push!(lower, lo); push!(upper, up); push!(x0, start); return length(names)
    end

    # C
    C_bounds = (1.0, 30.0, 10.0)
    for mode in mode_keys(vary_C_by_mode)
        idx_C[mode] = Dict{Symbol,Int}()
        for cue in cue_keys(vary_C_by_cue)
            suffix = cue==:all ? "" : "_$(cue)"
            m_suffix = mode==:shared ? "" : "_$(mode)"
            idx = push_param!("C$(m_suffix)$(suffix)", C_bounds...)
            idx_C[mode][cue] = idx
        end
    end

    # weights
    if weighting_mode == :free
        idx_w[:w2] = push_param!("w2", 1.0, 50.0, 2.0)
        idx_w[:w3] = push_param!("w3", 1.0, 50.0, 5.0)
        idx_w[:w4] = push_param!("w4", 1.0, 50.0, 10.0)
    elseif weighting_mode == :exponential
        idx_w[:w_slope] = push_param!("w_slope", 0.0, 10.0, 1.0)
    else
        error("Unknown weighting_mode: $weighting_mode")
    end

    # A
    A_bounds = (0.01, 1.0, 0.2)
    for mode in mode_keys(vary_A_by_mode)
        idx_A[mode] = Dict{Symbol,Int}()
        for cue in cue_keys(vary_A_by_cue)
            suffix = cue==:all ? "" : "_$(cue)"
            m_suffix = mode==:shared ? "" : "_$(mode)"
            idx = push_param!("A$(m_suffix)$(suffix)", A_bounds...)
            idx_A[mode][cue] = idx
        end
    end

    # k
    k_bounds = (0.05, 1.0, 0.2)
    for mode in mode_keys(vary_k_by_mode)
        idx_k[mode] = Dict{Symbol,Int}()
        for cue in cue_keys(vary_k_by_cue)
            suffix = cue==:all ? "" : "_$(cue)"
            m_suffix = mode==:shared ? "" : "_$(mode)"
            idx = push_param!("k$(m_suffix)$(suffix)", k_bounds...)
            idx_k[mode][cue] = idx
        end
    end

    # t0
    t0_bounds = (0.05, 0.6, 0.25)
    for mode in mode_keys(vary_t0_by_mode)
        idx_t0[mode] = Dict{Symbol,Int}()
        for cue in cue_keys(vary_t0_by_cue)
            suffix = cue==:all ? "" : "_$(cue)"
            m_suffix = mode==:shared ? "" : "_$(mode)"
            idx = push_param!("t0$(m_suffix)$(suffix)", t0_bounds...)
            idx_t0[mode][cue] = idx
        end
    end

    # pi
    pi_bounds = (0.01, 0.99, 0.5)
    if vary_pi_by_cue
        idx_pi[:single] = push_param!("pi_single", pi_bounds[1], pi_bounds[2], 0.9)
        idx_pi[:double] = push_param!("pi_double", pi_bounds[1], pi_bounds[2], 0.2)
    else
        idx_pi[:all] = push_param!("pi_all", pi_bounds...)
    end

    # contaminant
    if use_contaminant && estimate_contaminant
        for mode in mode_keys(vary_contam_alpha_by_mode)
            idx_contam_alpha[mode] = Dict{Symbol,Int}()
            for cue in cue_keys(vary_contam_alpha_by_cue)
                suffix = cue==:all ? "" : "_$(cue)"
                m_suffix = mode==:shared ? "" : "_$(mode)"
                idx = push_param!("alpha_contam$(m_suffix)$(suffix)", CONTAM_ALPHA_BOUNDS...)
                idx_contam_alpha[mode][cue] = idx
            end
        end
        for mode in mode_keys(vary_contam_rt_by_mode)
            idx_contam_rt[mode] = Dict{Symbol,Int}()
            for cue in cue_keys(vary_contam_rt_by_cue)
                suffix = cue==:all ? "" : "_$(cue)"
                m_suffix = mode==:shared ? "" : "_$(mode)"
                idx = push_param!("rtmax_contam$(m_suffix)$(suffix)", CONTAM_RT_BOUNDS...)
                idx_contam_rt[mode][cue] = idx
            end
        end
    end

    layout = DualModesLayout(weighting_mode, vary_C_by_mode, vary_C_by_cue, vary_k_by_mode, vary_k_by_cue, vary_t0_by_mode, vary_t0_by_cue, vary_A_by_mode, vary_A_by_cue, vary_pi_by_cue, use_contaminant, estimate_contaminant, vary_contam_alpha_by_mode, vary_contam_alpha_by_cue, vary_contam_rt_by_mode, vary_contam_rt_by_cue, idx_C, idx_k, idx_t0, idx_pi, idx_w, idx_A, idx_contam_alpha, idx_contam_rt, CONTAM_ALPHA_BOUNDS[3], CONTAM_RT_BOUNDS[3])
    return DualModesParams(lower, upper, x0), layout, names
end

function get_data_config(participant_id::Int)::DataConfig
    @assert participant_id in (1,2,3)
    folder = "ParticipantCPP002-00$(participant_id)"
    data_path = joinpath(DATA_BASE_DIR, folder, folder)
    return DataConfig(participant_id, data_path, "*.dat")
end

end # module
