# Plot bootstrap parameter distributions from saved CSV (no refitting).
#
# Run (uses defaults below):
#   julia plot_ccp001_bootstrap_density.jl
#
# Run with CLI override (subject, optional run tag):
#   julia plot_ccp001_bootstrap_density.jl S3
#   julia plot_ccp001_bootstrap_density.jl S3 full
#   julia plot_ccp001_bootstrap_density.jl S3 check
#
# CLI args take priority over DEFAULT_* below. Edit X_LIMS in config as needed.

using CSV, DataFrames, Statistics, Plots, KernelDensity, Printf

# --- Config (used when no CLI args) ---
const DEFAULT_SUBJECT_ID = "S2"
const DEFAULT_RUN_TAG = "full"   # "full" or "check" (matches bootstrap CSV suffix)

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16", "bootstrap")

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]

# X-axis limits per parameter. Use `nothing` for automatic subject-specific limits.
# Set a tuple to override manually, e.g. :theta => (7.0, 11.0)
const X_LIMS = Dict{Symbol, Union{Nothing, Tuple{Float64, Float64}}}(
    :theta  => nothing,
    :omega0 => nothing,
    :CI     => nothing,
    :A      => nothing,
    :k      => nothing,
    :t0     => nothing,
)

# Auto xlim tuning (used when X_LIMS[param] == nothing)
const AUTO_XLIM_Q_LO = 0.05      # robust lower quantile of all bootstrap reps
const AUTO_XLIM_Q_HI = 0.95      # robust upper quantile
const AUTO_XLIM_PAD_FRAC = 0.08  # padding as fraction of core span
const AUTO_XLIM_PAD_IQR = 0.20   # minimum padding as fraction of IQR
const AUTO_XLIM_PRINT = true     # print computed limits when plotting

# Light -> dark blue gradient (10% -> 100% trial fraction). Edit hex stops if needed.
const FRAC_COLOR_GRADIENT = ["#B3D9F2", "#6BAED6", "#3182BD", "#08519C", "#08306B"]

const SHOW_MODE_LINES = true
const MODE_LINE_STYLE = :dash
const MODE_LINE_LW = 1.75
const KDE_LINE_LW = 2.5
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
    results_csv = joinpath(FIG_DIR, "ccp001_$(lowercase(subj))_bootstrap_params_$(tag).csv")
    out_fig = joinpath(FIG_DIR, "ccp001_$(lowercase(subj))_bootstrap_density_$(tag)_v2.png")
    subj, tag, results_csv, out_fig
end

function fraction_colors(n::Int)::Vector{String}
    n <= 0 && return String[]
    n == 1 && return [FRAC_COLOR_GRADIENT[end]]
    if n == length(FRAC_COLOR_GRADIENT)
        return collect(FRAC_COLOR_GRADIENT)
    end
    idx = round.(Int, range(1, length(FRAC_COLOR_GRADIENT); length=n))
    FRAC_COLOR_GRADIENT[idx]
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

function auto_xlims(
    col::Symbol,
    ys::Vector{Float64},
    results::DataFrame,
    fractions::Vector{Float64},
)::Tuple{Float64, Float64}
    v = filter(isfinite, ys)
    isempty(v) && return (0.0, 1.0)

    qs = quantile(v, [AUTO_XLIM_Q_LO, 0.25, 0.75, AUTO_XLIM_Q_HI])
    q_lo, q25, q75, q_hi = qs
    iqr = max(q75 - q25, 1e-6)

    modes = Float64[]
    for frac in fractions
        sub = results[results.trial_fraction .≈ frac, :]
        push!(modes, kde_mode(Vector{Float64}(sub[!, col])))
    end
    modes = filter(isfinite, modes)

    core_lo = isempty(modes) ? q_lo : min(q_lo, minimum(modes))
    core_hi = isempty(modes) ? q_hi : max(q_hi, maximum(modes))
    core_span = max(core_hi - core_lo, iqr)

    pad = max(core_span * AUTO_XLIM_PAD_FRAC, iqr * AUTO_XLIM_PAD_IQR)
    lo = core_lo - pad
    hi = core_hi + pad

    # keep non-negative parameters readable (omega0, A, k, t0)
    if col in (:omega0, :A, :k, :t0)
        lo = max(lo, 0.0)
    end

    lo, hi
end

function resolve_xlims(
    col::Symbol,
    ys::Vector{Float64},
    results::DataFrame,
    fractions::Vector{Float64},
)
    cfg = get(X_LIMS, col, nothing)
    if cfg !== nothing
        return cfg
    end
    auto_xlims(col, ys, results, fractions)
end

function load_bootstrap_results(path::String)::DataFrame
    isfile(path) || error("Results CSV not found: $(path)")
    df = CSV.read(path, DataFrame)
    col_names = Symbol.(names(df))
    :trial_fraction in col_names || error("CSV missing trial_fraction column")
    for col in PARAM_COLS
        col in col_names || error("CSV missing $(col) column")
    end
    df
end

function plot_bootstrap_density_v2(
    results::DataFrame,
    fig_path::String,
    subject_id::AbstractString,
    run_tag::AbstractString,
)
    fractions = sort(unique(results.trial_fraction))
    frac_colors = fraction_colors(length(fractions))
    plots = Vector{Any}(undef, length(PARAM_COLS))

    for (i, col) in enumerate(PARAM_COLS)
        all_ys = Float64[]
        for frac in fractions
            sub = results[results.trial_fraction .≈ frac, :]
            append!(all_ys, Vector{Float64}(sub[!, col]))
        end
        xlims = resolve_xlims(col, all_ys, results, fractions)
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

        for (j, frac) in enumerate(fractions)
            sub = results[results.trial_fraction .≈ frac, :]
            ys = Vector{Float64}(sub[!, col])
            color = frac_colors[j]
            n_trials = nrow(sub) > 0 ? sub.n_trials[1] : 0
            label = "$(fraction_label(frac)) (n=$(n_trials))"

            dens = safe_kde(ys)
            if dens !== nothing
                plot!(p, dens.x, dens.density; label=label, color=color, lw=KDE_LINE_LW, alpha=KDE_ALPHA)
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
                    alpha=0.9,
                )
            end
        end

        plots[i] = p
    end

    fig = plot(
        plots...;
        layout=(length(PARAM_COLS), 1),
        size=(520, 220 * length(PARAM_COLS)),
        plot_title="CCP001 MIS+LBA bootstrap ($(subject_id), $(run_tag))",
        bottom_margin=3Plots.mm,
    )
    savefig(fig, fig_path)
end

function plot_ccp001_bootstrap_density(;
    subject_id::Union{Nothing, AbstractString}=nothing,
    run_tag::Union{Nothing, AbstractString}=nothing,
)
    subj, tag = resolve_subject_and_run_tag(; subject_id, run_tag)
    _, _, results_csv, out_fig = bootstrap_paths(subj, tag)

    isfile(results_csv) || error("No results at $(results_csv). Run mis_lba_basic_bootstrap.jl first.")

    results = load_bootstrap_results(results_csv)
    println("Subject: ", subj, "  run_tag: ", tag)
    println("Loaded ", nrow(results), " rows from ", results_csv)
    println("Trial fractions: ", sort(unique(results.trial_fraction)))
    if AUTO_XLIM_PRINT && all(get(X_LIMS, col, nothing) === nothing for col in PARAM_COLS)
        println("Auto x-limits:")
    end
    println("Writing figure: ", out_fig)

    plot_bootstrap_density_v2(results, out_fig, subj, tag)
    println("Done.")
    out_fig
end

# --- Main (direct script run only; skipped when included from bootstrap) ---
if abspath(PROGRAM_FILE) == @__FILE__
    plot_ccp001_bootstrap_density()
end
