# Plot bootstrap parameter densities from saved CSV.
# Run:  julia plot_ccp001_bootstrap_densities.jl
#
# Re-plot after fitting without re-running mis_lba_basic_bootstrap.jl.

using CSV, DataFrames, Statistics, Plots, KernelDensity

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ═══════════════════════════════════════════════════════════════════════════════

const SUBJECT_ID = "S3"
const RUN_TAG = "full"          # "full" or "check" (matches bootstrap output suffix)
const PROB_MODEL_LABEL = "noise eps=0.05"

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16", "bootstrap")
const RESULTS_CSV = joinpath(FIG_DIR, "ccp001_$(lowercase(SUBJECT_ID))_bootstrap_params_$(RUN_TAG).csv")
const DENSITY_FIG = joinpath(FIG_DIR, "ccp001_$(lowercase(SUBJECT_ID))_bootstrap_density_$(RUN_TAG).png")
const DENSITY_FIG_LOG_OMEGA = joinpath(
    FIG_DIR, "ccp001_$(lowercase(SUBJECT_ID))_bootstrap_density_omega0_log_$(RUN_TAG).png"
)

# Fixed x-axis limits per parameter (narrower than auto-scale).
# Set to nothing for automatic limits from data.
const THETA_XLIM = (7.5, 10.5)
const OMEGA0_XLIM = (0.05, 0.45)
const OMEGA0_LOG_XLIM = (-1.0, -0.25)   # log10(omega0); used in log-omega figure only
const CI_XLIM = (3.8, 4.6)
const A_XLIM = (0.95, 1.25)
const K_XLIM = (0.55, 1.05)
const T0_XLIM = (0.31, 0.40)

const PARAM_XLIMS = Dict(
    :theta => THETA_XLIM,
    :omega0 => OMEGA0_XLIM,
    :CI => CI_XLIM,
    :A => A_XLIM,
    :k => K_XLIM,
    :t0 => T0_XLIM
)

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]
const FRAC_COLORS = [:steelblue, :darkorange, :seagreen, :purple, :crimson]
const FIG_WIDTH = 900
const FIG_ROW_HEIGHT = 220
const DRAW_MODE_LINES = true   # vertical dashed line at KDE mode per fraction

# ═══════════════════════════════════════════════════════════════════════════════

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

function kde_mode(dens, xlim)
    if xlim !== nothing
        mask = (dens.x .>= xlim[1]) .& (dens.x .<= xlim[2])
        xs = dens.x[mask]
        ds = dens.density[mask]
        isempty(xs) && return nothing
        return xs[argmax(ds)]
    end
    return dens.x[argmax(dens.density)]
end

function plot_mode_line!(p, mode_x, color)
    DRAW_MODE_LINES || return
    mode_x === nothing && return
    vline!(p, [mode_x]; label=false, color=color, lw=1.5, ls=:dash, alpha=0.95)
end

function resolve_xlim(col::Symbol, log_omega0::Bool)
    if col == :omega0 && log_omega0
        return OMEGA0_LOG_XLIM
    end
    get(PARAM_XLIMS, col, nothing)
end

function plot_fraction_density!(p, ys::Vector{Float64}, label::String, color, xlim)
    u = length(unique(ys))
    if u == 1
        val = ys[1]
        xlo, xhi = xlim === nothing ? (val - 0.05, val + 0.05) : xlim
        val = clamp(val, xlo, xhi)
        plot!(p, [val, val], [0.0, 1.05]; label=label, color=color, lw=3)
        plot_mode_line!(p, val, color)
        return
    end
    dens = safe_kde(ys)
    if dens !== nothing
        if xlim !== nothing
            mask = (dens.x .>= xlim[1]) .& (dens.x .<= xlim[2])
            xs = dens.x[mask]
            ds = dens.density[mask]
            isempty(xs) && return
            plot!(p, xs, ds; label=label, color=color, lw=2.5, alpha=0.75)
            plot_mode_line!(p, kde_mode(dens, xlim), color)
        else
            plot!(p, dens.x, dens.density; label=label, color=color, lw=2.5, alpha=0.75)
            plot_mode_line!(p, kde_mode(dens, nothing), color)
        end
    else
        if xlim !== nothing
            ys = filter(v -> xlim[1] <= v <= xlim[2], ys)
            isempty(ys) && return
        end
        scatter!(
            p, ys, fill(0.0, length(ys));
            label=label, color=color, marker=:vline, markersize=4, alpha=0.6
        )
        plot_mode_line!(p, median(ys), color)
    end
end

function plot_bootstrap_densities(
    results::DataFrame,
    fig_path::String;
    subject_id::String = SUBJECT_ID,
    prob_label::String = PROB_MODEL_LABEL,
    log_omega0::Bool = false
)
    fractions = sort(unique(results.trial_fraction))
    n_fracs = length(fractions)
    plots = Vector{Any}(undef, length(PARAM_COLS))

    for (i, col) in enumerate(PARAM_COLS)
        xlim = resolve_xlim(col, log_omega0)
        xlab = col == :omega0 && log_omega0 ? "log10(omega0)" : "fitted"
        p = plot(
            legend=(i == 1 ? :topright : false),
            title=PARAM_LABELS[i],
            xlabel=xlab,
            ylabel="density",
        )
        for (j, frac) in enumerate(fractions)
            sub = results[results.trial_fraction .≈ frac, :]
            ys = Vector{Float64}(sub[!, col])
            if col == :omega0 && log_omega0
                ys = log10.(max.(ys, 1e-6))
            end
            if xlim !== nothing
                ys = filter(v -> xlim[1] <= v <= xlim[2], ys)
            end
            label = fraction_label(frac)
            plot_fraction_density!(p, ys, label, FRAC_COLORS[mod1(j, length(FRAC_COLORS))], xlim)
        end
        plot!(p; legend=(i == 1), xlim=xlim)
        plots[i] = p
    end

    fig = plot(
        plots...;
        layout=(length(PARAM_COLS), 1),
        size=(min(FIG_WIDTH, 200 * n_fracs + 120), FIG_ROW_HEIGHT * length(PARAM_COLS)),
        plot_title="CCP001 MIS+LBA bootstrap ($(subject_id), $(prob_label))",
        bottom_margin=3Plots.mm,
    )
    savefig(fig, fig_path)
end

function plot_bootstrap_from_csv(;
    results_csv::String = RESULTS_CSV,
    density_fig::String = DENSITY_FIG,
    density_fig_log_omega::String = DENSITY_FIG_LOG_OMEGA,
    subject_id::String = SUBJECT_ID,
    prob_label::String = PROB_MODEL_LABEL
)
    @assert isfile(results_csv) "Results CSV not found: $(results_csv)"
    results = CSV.read(results_csv, DataFrame)
    println("Loaded ", nrow(results), " rows from ", results_csv)
    println("Plotting with fixed xlims:")
    for col in PARAM_COLS
        xl = PARAM_XLIMS[col]
        println("  ", col, ": ", xl === nothing ? "auto" : xl)
    end
    println("  omega0 (log fig): ", OMEGA0_LOG_XLIM)
    plot_bootstrap_densities(results, density_fig; subject_id, prob_label, log_omega0=false)
    println("Saved: ", density_fig)
    plot_bootstrap_densities(results, density_fig_log_omega; subject_id, prob_label, log_omega0=true)
    println("Saved: ", density_fig_log_omega)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    plot_bootstrap_from_csv()
    println("Done.")
end
