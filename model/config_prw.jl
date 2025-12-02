# ==========================================================================
# PRW Configuration
# Independent settings for MIS Poisson Random Walk fitting
# ==========================================================================

# --------------------------------------------------------------------------
# Basic plot and data settings
# --------------------------------------------------------------------------
const ACCURACY_YLIM_PRW = (0.5, 1.02)
const RT_ALLCONDITIONS_YLIM_PRW = (0.0, 10.5)
const AXIS_FONT_SIZE_PRW = 12
const SAVE_INDIVIDUAL_CONDITION_PLOTS_PRW = true

# Output directories
const OUTPUTDATA_DIR_PRW = joinpath(@__DIR__, "outputdata")
const IMAGES_DIR_PRW = joinpath(@__DIR__, "images")

# --------------------------------------------------------------------------
# Data config
# --------------------------------------------------------------------------
struct DataConfig
    participant_id::Int
    data_base_path::String
    file_pattern::String
end

const DATA_BASE_DIR_PRW = joinpath(@__DIR__, "..", "data")
const FILE_PATTERN_PRW = "*.dat"
const PARTICIPANT_ID_PRW = 1
const PARTICIPANT_FOLDER_PRW = "ParticipantCPP002-00$(PARTICIPANT_ID_PRW)"
const DATA_PATH_PRW = joinpath(DATA_BASE_DIR_PRW, PARTICIPANT_FOLDER_PRW, PARTICIPANT_FOLDER_PRW)

function get_data_config_prw(participant_id::Int=PARTICIPANT_ID_PRW)::DataConfig
    return DataConfig(participant_id, DATA_PATH_PRW, FILE_PATTERN_PRW)
end

# --------------------------------------------------------------------------
# Cue-condition setup
# --------------------------------------------------------------------------
const CUE_CONDITION_SETUP_PRW = Dict(
    1 => [1],
    2 => [2],
    3 => [3],
    4 => [4],
    5 => [2, 1],
    6 => [3, 1],
    7 => [4, 1],
    8 => [3, 2],
    9 => [4, 2],
    10 => [4, 3],
)
const SINGLE_CUE_CONDITIONS_PRW = Set([1, 2, 3, 4])
const DOUBLE_CUE_CONDITIONS_PRW = Set([5, 6, 7, 8, 9, 10])

cue_condition_type_prw(cc) = cc in SINGLE_CUE_CONDITIONS_PRW ? :single : :double

# --------------------------------------------------------------------------
# Parameter bounds
# --------------------------------------------------------------------------
const DEFAULT_WEIGHTING_MODE_PRW = :exponential
const WEIGHTING_MODE_OVERRIDE_PRW = nothing  # set to :free or :exponential to force

const C_BOUNDS_PRW = (1.0, 30.0, 10.0)
const W_SLOPE_BOUNDS_PRW = (0.0, 10.0, 1.0)
const W_FREE_BOUNDS_PRW = Dict(
    :w2 => (1.0, 50.0, 2.0),
    :w3 => (1.0, 50.0, 5.0),
    :w4 => (1.0, 50.0, 10.0),
)
const T0_BOUNDS_PRW = (0.05, 0.6, 0.25)
const K_BOUNDS_PRW = (1.0, 12.0, 4.0)

const VARY_C_BY_CUECOUNT_PRW = true
const VARY_T0_BY_CUECOUNT_PRW = true
const VARY_K_BY_CUECOUNT_PRW = true

# Optional contaminant floor (uniform RT density) to handle fat RT tails.
# Mixture probability is fixed manually; not estimated in the PRW fit.
const USE_CONTAMINANT_FLOOR_PRW = true
const CONTAMINANT_ALPHA_PRW = 0.05
const CONTAMINANT_RT_MAX_PRW = 3.0  # seconds

# Optimization defaults (lightweight copy to avoid depending on main Config)
struct OptimizationConfigPRW
    g_tol::Float64
    f_reltol::Float64
    x_reltol::Float64
    max_iterations::Int
    time_limit::Float64
end

function get_optimization_config_prw()::OptimizationConfigPRW
    return OptimizationConfigPRW(
        1e-2,
        1e-3,
        1e-3,
        200,
        600.0,
    )
end

# --------------------------------------------------------------------------
# Layout containers (reused from main code for compatibility)
# --------------------------------------------------------------------------
struct AllConditionsParams
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
end

struct AllConditionsLayout
    weighting_mode::Symbol
    vary_C_by_cue::Bool
    vary_k_by_cue::Bool
    vary_t0_by_cue::Bool
    idx_C::Dict{Symbol,Int}
    idx_k::Dict{Symbol,Int}
    idx_t0::Dict{Symbol,Int}
    idx_w::Dict{Symbol,Int}
end

"""
    build_allconditions_params_prw(weighting_mode; vary_C_by_cue, vary_t0_by_cue, vary_k_by_cue, c_start_override)

Construct parameter arrays and layout for PRW fits independent of the LBA config.
"""
function build_allconditions_params_prw(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE_PRW;
    vary_C_by_cue::Bool=VARY_C_BY_CUECOUNT_PRW,
    vary_t0_by_cue::Bool=VARY_T0_BY_CUECOUNT_PRW,
    vary_k_by_cue::Bool=VARY_K_BY_CUECOUNT_PRW,
    c_start_override::Union{Nothing,Real,Tuple}=nothing)

    names = String[]
    lower = Float64[]
    upper = Float64[]
    x0 = Float64[]

    idx_C = Dict{Symbol,Int}()
    idx_k = Dict{Symbol,Int}()
    idx_t0 = Dict{Symbol,Int}()
    idx_w = Dict{Symbol,Int}()

    pushp!(n, lo, hi, start) = (push!(names, n); push!(lower, lo); push!(upper, hi); push!(x0, start); length(names))

    c_lo, c_hi, c_start_default = C_BOUNDS_PRW
    c_single_start = isnothing(c_start_override) ? c_start_default : (isa(c_start_override, Tuple) ? c_start_override[1] : c_start_override)
    c_double_start = isnothing(c_start_override) ? c_start_default : (isa(c_start_override, Tuple) ? c_start_override[end] : c_start_override)
    if vary_C_by_cue
        idx_C[:single] = pushp!("C_single", c_lo, c_hi, c_single_start)
        idx_C[:double] = pushp!("C_double", c_lo, c_hi, c_double_start)
    else
        idx_C[:all] = pushp!("C_all", c_lo, c_hi, c_single_start)
    end

    if weighting_mode == :exponential
        ws_lo, ws_hi, ws_start = W_SLOPE_BOUNDS_PRW
        idx_w[:w_slope] = pushp!("w_slope", ws_lo, ws_hi, ws_start)
    elseif weighting_mode == :free
        for sym in (:w2, :w3, :w4)
            bnds = W_FREE_BOUNDS_PRW[sym]
            idx_w[sym] = pushp!(String(sym), bnds...)
        end
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free.")
    end

    k_lo, k_hi, k_start = K_BOUNDS_PRW
    if vary_k_by_cue
        idx_k[:single] = pushp!("k_single", k_lo, k_hi, k_start)
        idx_k[:double] = pushp!("k_double", k_lo, k_hi, k_start)
    else
        idx_k[:all] = pushp!("k_all", k_lo, k_hi, k_start)
    end

    t0_lo, t0_hi, t0_start = T0_BOUNDS_PRW
    if vary_t0_by_cue
        idx_t0[:single] = pushp!("t0_single", t0_lo, t0_hi, t0_start)
        idx_t0[:double] = pushp!("t0_double", t0_lo, t0_hi, t0_start)
    else
        idx_t0[:all] = pushp!("t0_all", t0_lo, t0_hi, t0_start)
    end

    layout = AllConditionsLayout(weighting_mode, vary_C_by_cue, vary_k_by_cue, vary_t0_by_cue, idx_C, idx_k, idx_t0, idx_w)
    return AllConditionsParams(lower, upper, x0), layout, names
end
