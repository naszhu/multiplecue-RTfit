# Plot dual-cue vs all-cues bootstrap distributions (100% resample only).
# Two KDE curves per parameter panel.
#
# Run (uses defaults below):
#   julia plot_ccp001_cue_scope_bootstrap_density.jl
#
# Run with CLI override (subject, optional run tag):
#   julia plot_ccp001_cue_scope_bootstrap_density.jl S3
#   julia plot_ccp001_cue_scope_bootstrap_density.jl S3 full
#   julia plot_ccp001_cue_scope_bootstrap_density.jl S3 check
#
# When included from mis_lba_cue_scope_bootstrap.jl, pass keyword args:
#   plot_ccp001_cue_scope_bootstrap_density(; subject_id="S5", run_tag="full")
#
# Priority: keyword args > CLI ARGS > DEFAULT_* below.

using CSV, DataFrames, Statistics, Plots, KernelDensity, Printf

# --- Config (used when no CLI args / keyword overrides) ---
const DEFAULT_SUBJECT_ID = "S4"
const DEFAULT_RUN_TAG = "full"   # "full" or "check" (matches bootstrap CSV suffix)

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16", "bootstrap")

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]

const DATA_SCOPES = ["dual_cue_only", "all_cues"]
const SCOPE_DISPLAY = Dict(
    "dual_cue_only" => "dual-cue only",
    "all_cues" => "all cues",
)
const SCOPE_COLORS = Dict(
    "dual_cue_only" => "#D94801",
    "all_cues" => "#08519C",
)
const SCOPE_LINE_STYLES = Dict(
    "dual_cue_only" => :dash,
    "all_cues" => :solid,
)

# X-axis limits per parameter. Use `nothing` for automatic subject-specific limits.
const X_LIMS = Dict{Symbol, Union{Nothing, Tuple{Float64, Float64}}}(
    :theta  => nothing,
    :omega0 => nothing,
    :CI     => nothing,
    :A      => nothing,
    :k      => nothing,
    :t0     => nothing,
)

const AUTO_XLIM_Q_LO = 0.05
const AUTO_XLIM_Q_HI = 0.95
const AUTO_XLIM_PAD_FRAC = 0.08
const AUTO_XLIM_PAD_IQR = 0.20
const AUTO_XLIM_PRINT = true
const SHOW_MODE_LINES = true
const MODE_LINE_STYLE = :dot
const MODE_LINE_LW = 1.5
const KDE_LINE_LW = 2.5
const KDE_ALPHA = 0.80

function normalize_subject_id(raw::AbstractString)::String
    s = uppercase(strip(raw))
    startswith(s, "S") || (s = "S" * s)
    s
end

function normalize_run_tag(raw::AbstractString)::String
    tag = lowercase(strip(raw))
    tag in ("full", "check") || error("RUN_TAG must be \"full\" or \"check\"; got \"$(raw)\"")
    tag
end

function resolve_subject_and_run_tag(;
    subject_id::Union{Nothing, AbstractString}=nothing,
    run_tag::Union{Nothing, AbstractString}=nothing,
)
    subj = if subject_id !== nothing
        normalize_subject_id(String(subject_id))
    elseif !isempty(ARGS)
        normalize_subject_id(ARGS[1])
    else
        DEFAULT_SUBJECT_ID
    end

    tag = if run_tag !== nothing
        normalize_run_tag(String(run_tag))
    elseif length(ARGS) >= 2
        normalize_run_tag(ARGS[2])
    else
        DEFAULT_RUN_TAG
    end

    subj, tag
end

function bootstrap_paths(subject_id::AbstractString, run_tag::AbstractString)
    subj = normalize_subject_id(subject_id)
    tag = normalize_run_tag(run_tag)
    results_csv = joinpath(FIG_DIR, "ccp001_$(lowercase(subj))_cue_compare_bootstrap_params_$(tag).csv")
    out_fig = joinpath(FIG_DIR, "ccp001_$(lowercase(subj))_cue_compare_bootstrap_density_$(tag).png")
    subj, tag, results_csv, out_fig
end

function safe_kde(values::Vector{Float64})
    v = filter(isfinite, values)
    length(v) < 2 && return nothing
    length(unique(v)) < 2 && return nothing
    try
        kde(v)
    catch
        nothing
    end
end

function kde_mode(ys::Vector{Float64})::Float64
    dens = safe_kde(ys)
    if dens !== nothing
        return dens.x[argmax(dens.density)]
    end
    u = unique(filter(isfinite, ys))
    length(u) == 1 && return u[1]
    median(ys)
end

function auto_xlims(
    col::Symbol,
    ys::Vector{Float64},
    results::DataFrame,
    scopes_present::Vector{String},
)::Tuple{Float64, Float64}
    v = filter(isfinite, ys)
    isempty(v) && return (0.0, 1.0)

    qs = quantile(v, [AUTO_XLIM_Q_LO, 0.25, 0.75, AUTO_XLIM_Q_HI])
    q_lo, q25, q75, q_hi = qs
    iqr = max(q75 - q25, 1e-6)

    modes = Float64[
        kde_mode(Vector{Float64}(results[results.data_scope .== scope, col]))
        for scope in scopes_present
        if !isempty(results[results.data_scope .== scope, :])
    ]
    modes = filter(isfinite, modes)

    core_lo = isempty(modes) ? q_lo : min(q_lo, minimum(modes))
    core_hi = isempty(modes) ? q_hi : max(q_hi, maximum(modes))
    core_span = max(core_hi - core_lo, iqr)

    pad = max(core_span * AUTO_XLIM_PAD_FRAC, iqr * AUTO_XLIM_PAD_IQR)
    lo = core_lo - pad
    hi = core_hi + pad

    if col in (:omega0, :A, :k, :t0)
        lo = max(lo, 0.0)
    end

    lo, hi
end

function resolve_xlims(
    col::Symbol,
    ys::Vector{Float64},
    results::DataFrame,
    scopes_present::Vector{String},
)
    cfg = get(X_LIMS, col, nothing)
    if cfg !== nothing
        return cfg
    end
    auto_xlims(col, ys, results, scopes_present)
end

function load_bootstrap_results(path::String)::DataFrame
    isfile(path) || error("Results CSV not found: $(path)")
    df = CSV.read(path, DataFrame)
    col_names = Symbol.(names(df))
    :data_scope in col_names || error("CSV missing data_scope column")
    for col in PARAM_COLS
        col in col_names || error("CSV missing $(col) column")
    end
    df
end

function plot_cue_scope_bootstrap_density(
    results::DataFrame,
    fig_path::String,
    subject_id::AbstractString,
    run_tag::AbstractString,
)
    scopes_present = [s for s in DATA_SCOPES if any(results.data_scope .== s)]
    isempty(scopes_present) && error("No recognized data_scope values in results CSV")

    plots = Vector{Any}(undef, length(PARAM_COLS))

    for (i, col) in enumerate(PARAM_COLS)
        all_ys = Float64[]
        for scope in scopes_present
            sub = results[results.data_scope .== scope, :]
            append!(all_ys, Vector{Float64}(sub[!, col]))
        end
        xlims = resolve_xlims(col, all_ys, results, scopes_present)
        if AUTO_XLIM_PRINT && get(X_LIMS, col, nothing) === nothing
            @printf("  xlim %-6s = (%.4f, %.4f)\n", PARAM_LABELS[i], xlims[1], xlims[2])
        end

        p = plot(
            legend=(i == 1 ? :topright : false),
            title=PARAM_LABELS[i],
            xlabel="fitted",
            ylabel="density",
            xlims=xlims,
            grid=false,
        )

        for scope in scopes_present
            sub = results[results.data_scope .== scope, :]
            ys = Vector{Float64}(sub[!, col])
            scope_name = get(SCOPE_DISPLAY, scope, scope)
            color = get(SCOPE_COLORS, scope, "#333333")
            ls = get(SCOPE_LINE_STYLES, scope, :solid)
            n_trials = nrow(sub) > 0 ? sub.n_trials[1] : 0
            label = "$(scope_name) (n=$(n_trials))"

            dens = safe_kde(ys)
            if dens !== nothing
                plot!(p, dens.x, dens.density; label=label, color=color, lw=KDE_LINE_LW, ls=ls, alpha=KDE_ALPHA)
            else
                scatter!(p, ys, zeros(length(ys)); label=label, color=color, ms=4, alpha=KDE_ALPHA)
            end

            if SHOW_MODE_LINES
                mode_x = kde_mode(ys)
                vline!(
                    p, [mode_x];
                    label=false,
                    color=color,
                    lw=MODE_LINE_LW,
                    ls=MODE_LINE_STYLE,
                    alpha=0.85,
                )
            end
        end

        plots[i] = p
    end

    fig = plot(
        plots...;
        layout=(length(PARAM_COLS), 1),
        size=(520, 220 * length(PARAM_COLS)),
        plot_title="CCP001 dual-cue vs all cues ($(subject_id), $(run_tag), 100% bootstrap)",
        bottom_margin=3Plots.mm,
    )
    savefig(fig, fig_path)
end

function plot_ccp001_cue_scope_bootstrap_density(;
    subject_id::Union{Nothing, AbstractString}=nothing,
    run_tag::Union{Nothing, AbstractString}=nothing,
)
    subj, tag = resolve_subject_and_run_tag(; subject_id, run_tag)
    _, _, results_csv, out_fig = bootstrap_paths(subj, tag)

    isfile(results_csv) || error(
        "No results at $(results_csv). Run mis_lba_cue_scope_bootstrap.jl first."
    )

    results = load_bootstrap_results(results_csv)
    println("Subject: ", subj, "  run_tag: ", tag)
    println("Loaded ", nrow(results), " rows from ", results_csv)
    println("Data scopes: ", sort(unique(results.data_scope)))
    if AUTO_XLIM_PRINT && all(get(X_LIMS, col, nothing) === nothing for col in PARAM_COLS)
        println("Auto x-limits:")
    end
    println("Writing figure: ", out_fig)

    plot_cue_scope_bootstrap_density(results, out_fig, subj, tag)
    println("Done.")
    out_fig
end

# --- Main (direct script run only; skipped when included from bootstrap) ---
if abspath(PROGRAM_FILE) == @__FILE__
    plot_ccp001_cue_scope_bootstrap_density()
end
