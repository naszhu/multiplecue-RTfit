# ==========================================================================
# Configuration
# Settings and flags for model fitting and plotting
# ==========================================================================

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

const AllConditionsParams = SingleLBAParams

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
    AllConditionsLayout

Captures how parameters are expanded when they vary by cue type and/or include contaminants.
Index maps correspond to the order produced by `build_allconditions_params`.
"""
struct AllConditionsLayout
    weighting_mode::Symbol
    vary_C_by_cue::Bool
    vary_k_by_cue::Bool
    vary_t0_by_cue::Bool
    use_contaminant::Bool
    estimate_contaminant::Bool
    vary_contam_by_cue::Bool
    idx_C::Dict{Symbol,Int}
    idx_k::Dict{Symbol,Int}
    idx_t0::Dict{Symbol,Int}
    idx_w::Dict{Symbol,Int}
    idx_A::Int
    idx_contam_alpha::Dict{Symbol,Int}
    idx_contam_rt::Dict{Symbol,Int}
    contam_alpha_fixed::Float64
    contam_rt_fixed::Float64
end

# Default weighting mode for reward transforms (either :exponential or :free)
const DEFAULT_WEIGHTING_MODE = :free

# Parameter bounds (lower, upper, x0) centralized here
const C_BOUNDS_ALLCONDITIONS = (1.0, 30.0, 10.0)
const W_SLOPE_BOUNDS_ALLCONDITIONS = (0.0, 10.0, 1.0)
const W_FREE_BOUNDS_ALLCONDITIONS = Dict(
    :w2 => (1.0, 50.0, 2.0),
    :w3 => (1.0, 50.0, 5.0),
    :w4 => (1.0, 50.0, 10.0),
)
const A_BOUNDS_ALLCONDITIONS = (0.01, 1.0, 0.2)
const K_BOUNDS_ALLCONDITIONS = (0.05, 1.0, 0.2)
const T0_BOUNDS_ALLCONDITIONS = (0.05, 0.6, 0.25)

# Allow C/t0/k to vary by cue-count (single vs double cue) in all-conditions run
const VARY_C_BY_CUECOUNT_ALLCONDITIONS = false
const VARY_T0_BY_CUECOUNT_ALLCONDITIONS = false
const VARY_K_BY_CUECOUNT_ALLCONDITIONS = false

# Optional contaminant (uniform) floor to reduce catastrophic penalties from long tails
const USE_CONTAMINANT_FLOOR_ALLCONDITIONS = false
const ESTIMATE_CONTAMINANT_ALLCONDITIONS = false  # when true, alpha/rt_max are fitted parameters
const VARY_CONTAM_BY_CUE_ALLCONDITIONS = false
const CONTAM_ALPHA_BOUNDS_ALLCONDITIONS = (0.0, 0.1, 0.02)
const CONTAM_RT_MAX_BOUNDS_ALLCONDITIONS = (1.5, 4.0, 3.0)  # seconds
const CONTAMINANT_ALPHA_ALLCONDITIONS = CONTAM_ALPHA_BOUNDS_ALLCONDITIONS[3]
const CONTAMINANT_RT_MAX_ALLCONDITIONS = CONTAM_RT_MAX_BOUNDS_ALLCONDITIONS[3]

# Optional starting-value override for C (single or tuple for single/double)
const C_START_OVERRIDE_ALLCONDITIONS = nothing  # e.g., 30.0 or (30.0, 30.0)

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
const DATASET_VERSION_ALLCONDITIONS = 1  # 1 = CPP002 (ParticipantCPP002-00X), 2 = CPP001 (CPP001 - subj X)
const WEIGHTING_MODE_OVERRIDE_ALLCONDITIONS = nothing  # leave as `nothing` to use DEFAULT_WEIGHTING_MODE
const OUTPUT_CSV_ALLCONDITIONS = "model_fit_results_allconditions.csv"
const OUTPUT_PLOT_ALLCONDITIONS = "model_fit_plot_allconditions.png"

# Cue condition setup for experiment (1..10):
# (1), (2), (3), (4), (2,1), (3,1), (4,1), (3,2), (4,2), (4,3)
const CUE_CONDITION_SETUP = Dict(
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

const SINGLE_CUE_CONDITIONS = Set([1, 2, 3, 4])
const DOUBLE_CUE_CONDITIONS = Set([5, 6, 7, 8, 9, 10])

"""
    cue_condition_type(cue_condition)::Symbol

Returns :single or :double based on the configured cue condition setup.
"""
function cue_condition_type(cue_condition)::Symbol
    if !haskey(CUE_CONDITION_SETUP, cue_condition)
        error("Unknown cue condition: $cue_condition (expected keys: $(collect(keys(CUE_CONDITION_SETUP))))")
    end
    n_cues = length(CUE_CONDITION_SETUP[cue_condition])
    @assert n_cues in (1, 2) "CueCondition $cue_condition mapped to $n_cues cues, expected 1 or 2."
    return n_cues == 1 ? :single : :double
end

"""
    get_weighting_mode()::Symbol

Returns the default weighting mode for reward-to-weight transform.
Change `DEFAULT_WEIGHTING_MODE` to `:free` to estimate separate weights per reward value.
"""
get_weighting_mode()::Symbol = DEFAULT_WEIGHTING_MODE

"""
    build_allconditions_params(weighting_mode; vary_C_by_cue, vary_t0_by_cue, vary_k_by_cue, use_contaminant, estimate_contaminant, vary_contam_by_cue)

Constructs parameter arrays, names, and an index layout for the all-conditions single LBA with optional cue-specific C/t0/k and optional contaminant estimation.
"""
function build_allconditions_params(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE;
    vary_C_by_cue::Bool=VARY_C_BY_CUECOUNT_ALLCONDITIONS,
    vary_t0_by_cue::Bool=VARY_T0_BY_CUECOUNT_ALLCONDITIONS,
    vary_k_by_cue::Bool=VARY_K_BY_CUECOUNT_ALLCONDITIONS,
    use_contaminant::Bool=USE_CONTAMINANT_FLOOR_ALLCONDITIONS,
    estimate_contaminant::Bool=ESTIMATE_CONTAMINANT_ALLCONDITIONS,
    vary_contam_by_cue::Bool=VARY_CONTAM_BY_CUE_ALLCONDITIONS,
    c_start_override::Union{Nothing,Real,Tuple}=C_START_OVERRIDE_ALLCONDITIONS)

    names = String[]
    lower = Float64[]
    upper = Float64[]
    x0 = Float64[]

    idx_C = Dict{Symbol,Int}()
    idx_k = Dict{Symbol,Int}()
    idx_t0 = Dict{Symbol,Int}()
    idx_w = Dict{Symbol,Int}()
    idx_contam_alpha = Dict{Symbol,Int}()
    idx_contam_rt = Dict{Symbol,Int}()
    idx_A = 0

    function pushp!(n, lo, up, start)
        push!(names, n); push!(lower, lo); push!(upper, up); push!(x0, start); return length(names)
    end

    # C with optional override
    c_lo, c_hi, c_start_default = C_BOUNDS_ALLCONDITIONS
    c_single_start = isnothing(c_start_override) ? c_start_default : (isa(c_start_override, Tuple) ? c_start_override[1] : c_start_override)
    c_double_start = isnothing(c_start_override) ? c_start_default : (isa(c_start_override, Tuple) ? c_start_override[end] : c_start_override)
    if vary_C_by_cue
        idx_C[:single] = pushp!("C_single", c_lo, c_hi, c_single_start)
        idx_C[:double] = pushp!("C_double", c_lo, c_hi, c_double_start)
    else
        idx_C[:all] = pushp!("C_all", c_lo, c_hi, c_single_start)
    end

    # weights
    if weighting_mode == :exponential
        ws_lo, ws_hi, ws_start = W_SLOPE_BOUNDS_ALLCONDITIONS
        idx_w[:w_slope] = pushp!("w_slope", ws_lo, ws_hi, ws_start)
    elseif weighting_mode == :free
        for sym in (:w2, :w3, :w4)
            bnds = W_FREE_BOUNDS_ALLCONDITIONS[sym]
            idx_w[sym] = pushp!(String(sym), bnds...)
        end
    else
        error("Unknown weighting_mode: $weighting_mode. Use :exponential or :free")
    end

    # A (shared)
    idx_A = pushp!("A", A_BOUNDS_ALLCONDITIONS...)

    # k
    if vary_k_by_cue
        idx_k[:single] = pushp!("k_single", K_BOUNDS_ALLCONDITIONS...)
        idx_k[:double] = pushp!("k_double", K_BOUNDS_ALLCONDITIONS...)
    else
        idx_k[:all] = pushp!("k_all", K_BOUNDS_ALLCONDITIONS...)
    end

    # t0
    if vary_t0_by_cue
        idx_t0[:single] = pushp!("t0_single", T0_BOUNDS_ALLCONDITIONS...)
        idx_t0[:double] = pushp!("t0_double", T0_BOUNDS_ALLCONDITIONS...)
    else
        idx_t0[:all] = pushp!("t0_all", T0_BOUNDS_ALLCONDITIONS...)
    end

    # contaminant (alpha, rt_max) at end if estimated
    if use_contaminant && estimate_contaminant
        for cue in (vary_contam_by_cue ? (:single, :double) : (:all,))
            suffix = cue == :all ? "" : "_$(cue)"
            idx_contam_alpha[cue] = pushp!("alpha_contam$(suffix)", CONTAM_ALPHA_BOUNDS_ALLCONDITIONS...)
            idx_contam_rt[cue] = pushp!("rtmax_contam$(suffix)", CONTAM_RT_MAX_BOUNDS_ALLCONDITIONS...)
        end
    end

    layout = AllConditionsLayout(
        weighting_mode,
        vary_C_by_cue,
        vary_k_by_cue,
        vary_t0_by_cue,
        use_contaminant,
        estimate_contaminant,
        vary_contam_by_cue,
        idx_C,
        idx_k,
        idx_t0,
        idx_w,
        idx_A,
        idx_contam_alpha,
        idx_contam_rt,
        CONTAM_ALPHA_BOUNDS_ALLCONDITIONS[3],
        CONTAM_RT_MAX_BOUNDS_ALLCONDITIONS[3],
    )

    return AllConditionsParams(lower, upper, x0), layout, names
end

"""
    get_default_single_params(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE)::SingleLBAParams

Compatibility helper that builds the non-varying all-conditions parameter set.
"""
function get_default_single_params(weighting_mode::Symbol=DEFAULT_WEIGHTING_MODE)::SingleLBAParams
    params, _, _ = build_allconditions_params(weighting_mode;
        vary_C_by_cue=false,
        vary_t0_by_cue=false,
        vary_k_by_cue=false,
        use_contaminant=false,
        estimate_contaminant=false,
        vary_contam_by_cue=false)
    return params
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
    get_data_config(participant_id::Int; dataset_version::Int=1)::DataConfig

Returns data configuration for the specified participant.

Arguments:
- participant_id::Int - Participant ID (1, 2, or 3)
- dataset_version::Int - Dataset version (1 = CPP002, 2 = CPP001). Defaults to 1.

Returns:
- DataConfig with paths and settings for the specified participant
"""
function get_data_config(participant_id::Int; dataset_version::Int=1)::DataConfig
    if !(participant_id in [1, 2, 3])
        error("Invalid participant_id: $participant_id. Must be 1, 2, or 3.")
    end
    @assert dataset_version in (1,2) "dataset_version must be 1 (CPP002) or 2 (CPP001)"

    # Construct data path based on participant ID and dataset version
    if dataset_version == 1
        # CPP002: nested structure ParticipantCPP002-00X/ParticipantCPP002-00X/
        participant_folder_name = "ParticipantCPP002-00$(participant_id)"
        data_path = joinpath(DATA_BASE_DIR, participant_folder_name, participant_folder_name)
    else
        # CPP001: flat structure CPP001 - subj X/
        folder = "CPP001 - subj $(participant_id)"
        data_path = joinpath(DATA_BASE_DIR, folder)
    end

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
