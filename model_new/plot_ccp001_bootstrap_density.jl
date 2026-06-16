# Plot bootstrap parameter distributions from saved CSV (no refitting).
# Run:  julia plot_ccp001_bootstrap_density.jl
#
# Edit SUBJECT_ID, RUN_TAG, and X_LIMS below. Set any xlim to nothing for auto range.

using CSV, DataFrames, Statistics, Plots, KernelDensity

# --- Config ---
const SUBJECT_ID = "S2"
const RUN_TAG = "full"   # "full" or "check" (matches bootstrap CSV suffix)

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16", "bootstrap")
const RESULTS_CSV = joinpath(FIG_DIR, "ccp001_$(lowercase(SUBJECT_ID))_bootstrap_params_$(RUN_TAG).csv")
const OUT_FIG = joinpath(FIG_DIR, "ccp001_$(lowercase(SUBJECT_ID))_bootstrap_density_$(RUN_TAG)_v2.png")

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]

# X-axis limits per parameter (edit freely). Use `nothing` to auto-fit from data + padding.
const X_LIMS = Dict{Symbol, Union{Nothing, Tuple{Float64, Float64}}}(
    :theta  => (7.0, 11.0),
    :omega0 => (0.05, 0.45),
    :CI     => (3.7, 4.6),
    :A      => (0.9, 1.35),
    :k      => (0.6, 1.1),
    :t0     => (0.32, 0.40),
)

# Light -> dark blue gradient (10% -> 100% trial fraction). Edit hex stops if needed.
const FRAC_COLOR_GRADIENT = ["#B3D9F2", "#6BAED6", "#3182BD", "#08519C", "#08306B"]

const AUTO_XLIM_PAD = 0.04   # fraction of data range added on each side when auto
const SHOW_MODE_LINES = true
const MODE_LINE_STYLE = :dash
const MODE_LINE_LW = 1.75
const KDE_LINE_LW = 2.5
const KDE_ALPHA = 0.75

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
    :trial_fraction in col_names || error("CSV missing trial_fraction column")
    for col in PARAM_COLS
        col in col_names || error("CSV missing $(col) column")
    end
    df
end

function plot_bootstrap_density_v2(results::DataFrame, fig_path::String)
    fractions = sort(unique(results.trial_fraction))
    frac_colors = fraction_colors(length(fractions))
    plots = Vector{Any}(undef, length(PARAM_COLS))

    for (i, col) in enumerate(PARAM_COLS)
        all_ys = Float64[]
        for frac in fractions
            sub = results[results.trial_fraction .≈ frac, :]
            append!(all_ys, Vector{Float64}(sub[!, col]))
        end
        xlims = resolve_xlims(col, all_ys)

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
        plot_title="CCP001 MIS+LBA bootstrap ($(SUBJECT_ID), $(RUN_TAG))",
        bottom_margin=3Plots.mm,
    )
    savefig(fig, fig_path)
end

# --- Main ---
@assert isfile(RESULTS_CSV) "No results at $(RESULTS_CSV). Run mis_lba_basic_bootstrap.jl first."

results = load_bootstrap_results(RESULTS_CSV)
println("Loaded ", nrow(results), " rows from ", RESULTS_CSV)
println("Trial fractions: ", sort(unique(results.trial_fraction)))
println("Writing figure: ", OUT_FIG)

plot_bootstrap_density_v2(results, OUT_FIG)
println("Done.")
