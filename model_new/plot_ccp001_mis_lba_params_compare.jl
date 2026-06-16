using CSV, DataFrames, Plots

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16")
const LINE_LW = 2.5
const MARKER_SIZE = 6
const MARKER_STROKE = 1.25
const X_DODGE = 0.14
const OMEGA0_FLOOR = 5e-7

# Each entry: csv path, canonical subset id, plot label, color, marker, x-dodge sign
const SUBSET_SOURCES = [
    (joinpath(FIG_DIR, "ccp001_mis_lba_params_s6_17_all_cues.csv"), "s6_17_all_cues", "sessions 6–17, all cues", :steelblue, :circle, -1),
    (joinpath(FIG_DIR, "ccp001_mis_lba_params_s6_17_dual_cue.csv"), "s6_17_dual_cue", "sessions 6–17, dual-cue only", :darkorange, :rect, 0),
    (joinpath(FIG_DIR, "ccp001_mis_lba_params_s6_11_all_cues.csv"), "s6_11_all_cues", "sessions 6–11, all cues", :seagreen, :diamond, 1),
]

# Legacy filenames from earlier runs (migrated on load if new names are absent).
const LEGACY_ALIASES = Dict(
    joinpath(FIG_DIR, "ccp001_mis_lba_params_full.csv") => ("s6_17_all_cues", "sessions 6–17, all cues"),
    joinpath(FIG_DIR, "ccp001_mis_lba_params.csv") => ("s6_17_all_cues", "sessions 6–17, all cues"),
    joinpath(FIG_DIR, "ccp001_mis_lba_params_dual_cue.csv") => ("s6_17_dual_cue", "sessions 6–17, dual-cue only"),
)

function normalize_results(df::DataFrame, subset_id::String, subset_label::String)
    if "data_subset" in names(df) && "subset_label" in names(df)
        return df
    end
    out = copy(df)
    if "data_chunk" in names(out)
        chunk = string(out.data_chunk[1])
        if chunk == "full"
            subset_id = "s6_17_all_cues"
            subset_label = "sessions 6–17, all cues"
        elseif chunk == "dual_cue"
            subset_id = "s6_17_dual_cue"
            subset_label = "sessions 6–17, dual-cue only"
        end
        select!(out, Not(:data_chunk))
    end
    out.data_subset = fill(subset_id, nrow(out))
    out.subset_label = fill(subset_label, nrow(out))
    select!(out, [:subject_id, :data_subset, :subset_label, :n_trials, :theta, :omega0, :CI, :A, :k, :t0, :nll])
    out
end

function load_subset_csv(path::String, subset_id::String, subset_label::String)
    actual_path = isfile(path) ? path : begin
        for (legacy_path, (lid, llab)) in LEGACY_ALIASES
            if legacy_path == path || endswith(legacy_path, basename(path))
                if isfile(legacy_path)
                    subset_id, subset_label = lid, llab
                    return legacy_path
                end
            end
        end
        error("Missing parameter CSV: $(path)")
    end
    normalize_results(CSV.read(actual_path, DataFrame), subset_id, subset_label)
end

function resolve_subset_source(entry)
    path, subset_id, subset_label, color, marker, dodge_sign = entry
    if !isfile(path)
        for (legacy_path, (lid, llab)) in LEGACY_ALIASES
            basename(legacy_path) == basename(path) && continue
            if lid == subset_id && isfile(legacy_path)
                path = legacy_path
                break
            end
        end
        if subset_id == "s6_17_all_cues"
            for candidate in (
                joinpath(FIG_DIR, "ccp001_mis_lba_params_full.csv"),
                joinpath(FIG_DIR, "ccp001_mis_lba_params.csv"),
            )
                isfile(candidate) && (path = candidate; break)
            end
        elseif subset_id == "s6_17_dual_cue"
            legacy = joinpath(FIG_DIR, "ccp001_mis_lba_params_dual_cue.csv")
            isfile(legacy) && (path = legacy)
        end
    end
    df = load_subset_csv(path, subset_id, subset_label)
    (df=df, subset_id=subset_id, subset_label=subset_label, color=color, marker=marker, dodge_sign=dodge_sign)
end

function plot_y_values(col::Symbol, ys::Vector{Float64})
    col == :omega0 ? max.(ys, OMEGA0_FLOOR) : ys
end

function plot_params_compare(combined::DataFrame, subset_meta, fig_path::String)
    subjects = sort(unique(combined.subject_id))
    x = 1:length(subjects)
    param_cols = [:theta, :omega0, :CI, :A, :k, :t0]
    param_labels = ["theta", "omega0", "CI", "A", "k", "t0"]

    plots = Vector{Any}(undef, length(param_cols))
    for (i, col) in enumerate(param_cols)
        p = plot(legend=(i == 1 ? :topright : false))
        for meta in subset_meta
            sub = filter(row -> row.data_subset == meta.subset_id, combined)
            isempty(sub) && continue
            ys = [sub[sub.subject_id .== sid, col][1] for sid in subjects]
            ys_plot = plot_y_values(col, ys)
            x_plot = x .+ meta.dodge_sign * X_DODGE
            plot!(
                p, x_plot, ys_plot;
                label=meta.subset_label,
                color=meta.color,
                lw=LINE_LW,
                alpha=0.9,
                marker=meta.marker,
                markersize=MARKER_SIZE,
                markerstrokecolor=:black,
                markerstrokewidth=MARKER_STROKE
            )
        end
        if col == :omega0
            plot!(
                p;
                xticks=(x, subjects),
                xrotation=45,
                ylabel="omega0 (log10)",
                yguidefontsize=9,
                yscale=:log10,
                legend=(i == 1)
            )
        else
            plot!(
                p;
                xticks=(x, subjects),
                xrotation=45,
                ylabel=param_labels[i],
                legend=(i == 1)
            )
        end
        plots[i] = p
    end

    plot(
        plots...;
        layout=(length(param_cols), 1),
        size=(800, 940),
        plot_title="CCP001 MIS+LBA parameters by data subset",
        bottom_margin=5Plots.mm
    )
    savefig(fig_path)
end

loaded = [resolve_subset_source(entry) for entry in SUBSET_SOURCES]
subset_meta = [
    (; subset_id=l.subset_id, subset_label=l.subset_label, color=l.color, marker=l.marker, dodge_sign=l.dodge_sign)
    for l in loaded
]
combined = vcat((l.df for l in loaded)...; cols=:union)
sort!(combined, [:subject_id, :data_subset])

combined_path = joinpath(FIG_DIR, "ccp001_mis_lba_params_compare.csv")
CSV.write(combined_path, combined)
println("Saved combined table: ", combined_path)

fig_path = joinpath(FIG_DIR, "ccp001_mis_lba_param_lines_compare_noise_eps0p05.png")
plot_params_compare(combined, subset_meta, fig_path)
println("Saved plot: ", fig_path)
