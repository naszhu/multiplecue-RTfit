# Plot cue-scope bootstrap parameter distributions (dual-cue vs all cues).
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

using CSV, DataFrames, Statistics, Plots, KernelDensity

# --- Config (used when no CLI args / keyword overrides) ---
const DEFAULT_SUBJECT_ID = "S5"
const DEFAULT_RUN_TAG = "full"   # "full" or "check" (matches bootstrap CSV suffix)

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16", "bootstrap")

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]

const DATA_SCOPES = ["dual_cue_only", "all_cues"]
const SCOPE_DISPLAY = Dict(
    "dual_cue_only" => "dual-cue only",
    "all_cues" => "all cues",
)
const SCOPE_LINE_STYLES = Dict(
    "dual_cue_only" => :dash,
    "all_cues" => :solid,
)

# Blue gradient for all-cues; orange gradient for dual-cue only.
const FULL_FRAC_COLOR_GRADIENT = ["#B3D9F2", "#6BAED6", "#3182BD", "#08519C", "#08306B"]
const DUAL_FRAC_COLOR_GRADIENT = ["#FDD0A2", "#FDAE6B", "#F16913", "#D94801", "#8C2D04"]

# X-axis limits per parameter (edit freely). Use `nothing` to auto-fit from data + padding.
const X_LIMS = Dict{Symbol, Union{Nothing, Tuple{Float64, Float64}}}(
    :theta  => nothing,
    :omega0 => nothing,
    :CI     => nothing,
    :A      => nothing,
    :k      => nothing,
    :t0     => nothing,
)

const AUTO_XLIM_PAD = 0.04
const SHOW_MODE_LINES = true
const MODE_LINE_STYLE = :dot
const MODE_LINE_LW = 1.5
const KDE_LINE_LW = 2.25
const KDE_ALPHA = 0.75

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
    results_csv = joinpath(FIG_DIR, "ccp001_$(lowercase(subj))_cue_scope_bootstrap_params_$(tag).csv")
    out_fig = joinpath(FIG_DIR, "ccp001_$(lowercase(subj))_cue_scope_bootstrap_density_$(tag)_v2.png")
    subj, tag, results_csv, out_fig
end

function scope_color_gradient(scope::AbstractString, n::Int)::Vector{String}
    base = scope == "dual_cue_only" ? DUAL_FRAC_COLOR_GRADIENT : FULL_FRAC_COLOR_GRADIENT
    n <= 0 && return String[]
    n == 1 && return [base[end]]
    if n == length(base)
        return collect(base)
    end
    idx = round.(Int, range(1, length(base); length=n))
    base[idx]
end

function fraction_label(fraction::Float64)::String
    "$(round(Int, 100 * fraction))pct"
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

function resolve_xlims(col::Symbol, ys::Vector{Float64})
    cfg = get(X_LIMS, col, nothing)
    if cfg !== nothing
        return cfg
    end
    v = filter(isfinite, ys)
    isempty(v) && return (0.0, 1.0)
    lo, hi = extrema(v)
    span = max(hi - lo, 1e-6)
    pad = span * AUTO_XLIM_PAD
    (lo - pad, hi + pad)
end

function load_bootstrap_results(path::String)::DataFrame
    isfile(path) || error("Results CSV not found: $(path)")
    df = CSV.read(path, DataFrame)
    col_names = Symbol.(names(df))
    :data_scope in col_names || error("CSV missing data_scope column")
    :trial_fraction in col_names || error("CSV missing trial_fraction column")
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
    fractions = sort(unique(results.trial_fraction))
    scopes_present = [s for s in DATA_SCOPES if any(results.data_scope .== s)]
    isempty(scopes_present) && error("No recognized data_scope values in results CSV")

    scope_colors = Dict(
        scope => scope_color_gradient(scope, length(fractions))
        for scope in scopes_present
    )

    plots = Vector{Any}(undef, length(PARAM_COLS))
    legend_added = false

    for (i, col) in enumerate(PARAM_COLS)
        all_ys = Float64[]
        for scope in scopes_present
            sub_scope = results[results.data_scope .== scope, :]
            for frac in fractions
                sub = sub_scope[sub_scope.trial_fraction .≈ frac, :]
                append!(all_ys, Vector{Float64}(sub[!, col]))
            end
        end
        xlims = resolve_xlims(col, all_ys)

        p = plot(
            legend=false,
            title=PARAM_LABELS[i],
            xlabel="fitted",
            ylabel="density",
            xlims=xlims,
            grid=false,
        )

        for scope in scopes_present
            sub_scope = results[results.data_scope .== scope, :]
            scope_name = get(SCOPE_DISPLAY, scope, scope)
            ls = get(SCOPE_LINE_STYLES, scope, :solid)
            colors = scope_colors[scope]

            for (j, frac) in enumerate(fractions)
                sub = sub_scope[sub_scope.trial_fraction .≈ frac, :]
                ys = Vector{Float64}(sub[!, col])
                color = colors[j]
                n_trials = nrow(sub) > 0 ? sub.n_trials[1] : 0
                label = "$(scope_name), $(fraction_label(frac)) (n=$(n_trials))"

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

                if !legend_added
                    plot!(p; legend=:topright)
                    legend_added = true
                end
            end
        end

        plots[i] = p
    end

    fig = plot(
        plots...;
        layout=(length(PARAM_COLS), 1),
        size=(560, 220 * length(PARAM_COLS)),
        plot_title="CCP001 cue-scope bootstrap ($(subject_id), $(run_tag))",
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
    println("Trial fractions: ", sort(unique(results.trial_fraction)))
    println("Writing figure: ", out_fig)

    plot_cue_scope_bootstrap_density(results, out_fig, subj, tag)
    println("Done.")
    out_fig
end

# --- Main (direct script run only; skipped when included from bootstrap) ---
if abspath(PROGRAM_FILE) == @__FILE__
    plot_ccp001_cue_scope_bootstrap_density()
end
