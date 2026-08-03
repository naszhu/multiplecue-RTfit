# Compare two MIS+LBA prediction modes on the same observed data.
#
# Run:  julia compare_two_modes.jl
#
# For each subject / mode:
#   1) Look for prediction CSV in data_prediction/
#   2) If missing, call mis_lba_basic_multisub.jl with CLI args to fit + save
#   3) Overlay both predictions on observed data (2x2 panels)
#
# =============================================================================
# CLI ARG MAP (same as mis_lba_basic_multisub.jl)
# =============================================================================
#   subs=12:20          subject range (here set below as subject_range)
#   omega=freeW|expW    free omega1..4  vs  exponential MIS (theta)
#   k=shared|cueN|competitor|delta_r|conflict
#                       shared k
#                       cueN       = separate k for 1-cue vs 2-cue
#                       competitor = k = k0 + beta_k * C
#                       delta_r    = k = k0 + beta_k * (H-L)
#                       conflict   = k = k0 + beta_k * (3-(H-L))
#   CI=shared|cueN      shared CI vs split by cue number
#   t0=shared|cueN      shared t0 vs split by cue number
#   nsims=100           predictive simulation count
#
# File-name tags produced by multisub (for CSV lookup):
#   freeW / expW
#   + _varyK | _kComp | _kDeltaR | _kConflict   (from k=)
#   + _varyCI / _varyT0                         (if CI/t0 = cueN)
#   Example: freeW_varyK , freeW_kComp
# =============================================================================

ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Plots
default(show = false)

# =============================================================================
# CONFIG (change these)
# =============================================================================

# Subjects to compare
subject_range = 16:28

# Two modes to overlay (CLI args passed to mis_lba_basic_multisub.jl)
# Options:
#   omega: freeW | expW
#   k:     shared | cueN | competitor | delta_r | conflict
#   CI:    shared | cueN
#   t0:    shared | cueN
#
# Default request: freeW + k by cue number  vs  freeW + k by competitor
mode_a_args = ["omega=freeW", "k=competitor", "CI=shared", "t0=shared"]
mode_b_args = ["omega=freeW", "k=delta_r", "CI=shared", "t0=shared"]

# Colors: black = data, two colors = the two predictions
color_data = :black
color_a = :steelblue
color_b = :darkorange

# If true, re-run fits even when prediction CSV already exists
force_refit = false

# Extra CLI passed when fitting (e.g. fewer sims for a quick test)
extra_fit_args = String[]   # e.g. ["nsims=50"]

# Paths
model_dir = @__DIR__
fit_script = joinpath(model_dir, "mis_lba_basic_multisub.jl")
pred_dir = joinpath(model_dir, "data_prediction")
fig_dir = joinpath(model_dir, "figs", "fit_aug1")

# =============================================================================
# Helpers: turn CLI args into the same mode tag multisub uses in filenames
# =============================================================================
function get_arg(args::Vector{String}, key::String, default::String)
    key_l = lowercase(key)
    for a in args
        occursin("=", a) || continue
        k, v = split(a, "="; limit=2)
        if lowercase(strip(k)) == key_l
            return lowercase(strip(v))
        end
    end
    return lowercase(default)
end

# Build a short legend label from the CLI args (always matches the chosen mode)
function mode_label_from_args(args::Vector{String})
    omega = get_arg(args, "omega", "freeW")
    k = get_arg(args, "k", "shared")
    ci = get_arg(args, "CI", "shared")
    t0 = get_arg(args, "t0", "shared")

    omega_lab = startswith(omega, "free") ? "freeW" : "expW"
    if k in ("cuen", "cue_number", "varyk")
        k_lab = "k by cueN"
    elseif k in ("competitor", "comp", "kcomp")
        k_lab = "k by competitor"
    elseif k in ("delta_r", "deltar", "kdelta")
        k_lab = "k by delta_r"
    elseif k in ("conflict", "kconflict")
        k_lab = "k by conflict"
    else
        k_lab = "k shared"
    end

    bits = ["$(omega_lab)", k_lab]
    ci in ("cuen", "cue_number") && push!(bits, "CI by cueN")
    t0 in ("cuen", "cue_number") && push!(bits, "t0 by cueN")
    return join(bits, " + ")
end

function mode_tag_from_args(args::Vector{String})
    omega = get_arg(args, "omega", "freeW")
    k = get_arg(args, "k", "shared")
    ci = get_arg(args, "CI", "shared")
    t0 = get_arg(args, "t0", "shared")

    tag = startswith(omega, "free") ? "freeW" : "expW"
    extras = String[]
    if ci in ("cuen", "cue_number")
        push!(extras, "varyCI")
    end
    if k in ("cuen", "cue_number", "varyk")
        push!(extras, "varyK")
    elseif k in ("competitor", "comp", "kcomp")
        push!(extras, "kComp")
    elseif k in ("delta_r", "deltar", "kdelta")
        push!(extras, "kDeltaR")
    elseif k in ("conflict", "kconflict")
        push!(extras, "kConflict")
    end
    if t0 in ("cuen", "cue_number")
        push!(extras, "varyT0")
    end
    isempty(extras) || (tag = tag * "_" * join(extras, "_"))
    return tag
end

# Labels generated from the args above (do not edit by hand)
mode_a_label = mode_label_from_args(mode_a_args)
mode_b_label = mode_label_from_args(mode_b_args)

# Same data / session rules as mis_lba_basic_multisub.jl
lab_data_root = joinpath(model_dir, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab")
MIN_N_SESSIONS = 10

function list_trial_csvs(subj_dir)
    files = readdir(subj_dir; join=true)
    out = String[]
    for f in files
        name = basename(f)
        if endswith(name, "_trials.csv")
            push!(out, f)
        elseif endswith(name, ".csv") && !occursin("metadata", name)
            push!(out, f)
        end
    end
    return out
end

# true if this subject folder exists and has enough session files
function subject_is_usable(sub_num::Int)
    subj_dir = joinpath(lab_data_root, "sub$(sub_num)")
    if !isdir(subj_dir)
        println("  skip sub$(sub_num): folder not found")
        return false
    end
    n_sessions = length(list_trial_csvs(subj_dir))
    if n_sessions < MIN_N_SESSIONS
        println("  skip sub$(sub_num): only $(n_sessions) session files (< $(MIN_N_SESSIONS))")
        return false
    end
    return true
end

function pred_csv_path(sub_num::Int, mode_tag::String)
    return joinpath(pred_dir, "sub$(sub_num)_mis_lba_$(mode_tag)_pred.csv")
end

# Returns CSV path, or nothing if the subject/mode cannot be fit
function ensure_prediction!(sub_num::Int, mode_args::Vector{String}, mode_tag::String)
    path = pred_csv_path(sub_num, mode_tag)
    if isfile(path) && !force_refit
        println("  cache hit: ", basename(path))
        return path
    end

    println("  fitting (no cache): sub$(sub_num)  tag=$(mode_tag)")
    cmd_args = String["subs=$(sub_num)", mode_args..., extra_fit_args...]
    try
        run(`julia $(fit_script) $(cmd_args)`)
    catch err
        println("  skip sub$(sub_num) tag=$(mode_tag): fit failed ($(typeof(err)))")
        return nothing
    end
    if !isfile(path)
        println("  skip sub$(sub_num) tag=$(mode_tag): prediction CSV was not created")
        return nothing
    end
    return path
end

function load_pred(path::String)
    return CSV.read(path, DataFrame)
end

# =============================================================================
# Overlay plotting
# =============================================================================
function overlay_accuracy_panel(df_a, df_b, title_str, label_a, label_b)
    conditions = String.(df_a.condition)
    x = collect(1:length(conditions))

    data_y = Float64.(df_a.data_acc)
    a_y = Float64.(df_a.pred_acc)
    a_lo = Float64.(df_a.pred_acc_lo)
    a_hi = Float64.(df_a.pred_acc_hi)
    b_y = Float64.(df_b.pred_acc)
    b_lo = Float64.(df_b.pred_acc_lo)
    b_hi = Float64.(df_b.pred_acc_hi)

    p = plot(
        title = title_str,
        ylabel = "Accuracy",
        xlabel = "",
        xticks = (x, conditions),
        xrotation = 30,
        legend = :bottomright,
        size = (500, 350)
    )
    # Data (black)
    plot!(p, x, data_y; color = color_data, lw = 1, linestyle = :dash, label = false)
    scatter!(p, x, data_y;
        marker = :circle, markersize = 6,
        markercolor = :white, markerstrokecolor = color_data, markerstrokewidth = 1.5,
        label = "Data"
    )
    # Mode A
    plot!(p, x, a_y; color = color_a, lw = 1.2, label = false)
    scatter!(p, x, a_y;
        yerror = (a_y .- a_lo, a_hi .- a_y),
        marker = :circle, markersize = 5,
        markercolor = color_a, markerstrokecolor = color_a,
        label = label_a
    )
    # Mode B
    plot!(p, x, b_y; color = color_b, lw = 1.2, label = false)
    scatter!(p, x, b_y;
        yerror = (b_y .- b_lo, b_hi .- b_y),
        marker = :diamond, markersize = 5,
        markercolor = color_b, markerstrokecolor = color_b,
        label = label_b
    )
    return p
end

function overlay_rt_panel(df_a, df_b, title_str)
    conditions = String.(df_a.condition)
    x = collect(1:length(conditions))

    p = plot(
        title = title_str,
        ylabel = "RT (seconds)",
        xlabel = "",
        xticks = (x, conditions),
        xrotation = 30,
        legend = false,
        size = (500, 350)
    )

    # Three quantile bands: 0.1, 0.5, 0.9
    data_cols = [:data_rt10, :data_rt50, :data_rt90]
    a_cols = [:pred_rt10, :pred_rt50, :pred_rt90]
    a_lo_cols = [:pred_rt10_lo, :pred_rt50_lo, :pred_rt90_lo]
    a_hi_cols = [:pred_rt10_hi, :pred_rt50_hi, :pred_rt90_hi]
    b_cols = [:pred_rt10, :pred_rt50, :pred_rt90]
    b_lo_cols = [:pred_rt10_lo, :pred_rt50_lo, :pred_rt90_lo]
    b_hi_cols = [:pred_rt10_hi, :pred_rt50_hi, :pred_rt90_hi]

    for q in 1:3
        data_y = Float64.(df_a[!, data_cols[q]])
        a_y = Float64.(df_a[!, a_cols[q]])
        a_lo = Float64.(df_a[!, a_lo_cols[q]])
        a_hi = Float64.(df_a[!, a_hi_cols[q]])
        b_y = Float64.(df_b[!, b_cols[q]])
        b_lo = Float64.(df_b[!, b_lo_cols[q]])
        b_hi = Float64.(df_b[!, b_hi_cols[q]])

        plot!(p, x, data_y; color = color_data, lw = 1, linestyle = :dash, label = false)
        scatter!(p, x, data_y;
            marker = :circle, markersize = 4,
            markercolor = :white, markerstrokecolor = color_data, markerstrokewidth = 1.2,
            label = false
        )
        plot!(p, x, a_y; color = color_a, lw = 1, label = false)
        scatter!(p, x, a_y;
            yerror = (a_y .- a_lo, a_hi .- a_y),
            marker = :circle, markersize = 3.5,
            markercolor = color_a, markerstrokecolor = color_a,
            label = false
        )
        plot!(p, x, b_y; color = color_b, lw = 1, label = false)
        scatter!(p, x, b_y;
            yerror = (b_y .- b_lo, b_hi .- b_y),
            marker = :diamond, markersize = 3.5,
            markercolor = color_b, markerstrokecolor = color_b,
            label = false
        )
    end
    return p
end

function save_compare_figure(sub_num, df_a, df_b, tag_a, tag_b)
    df_a_s = filter(r -> r.cue_group == "single", df_a)
    df_a_d = filter(r -> r.cue_group == "dual", df_a)
    df_b_s = filter(r -> r.cue_group == "single", df_b)
    df_b_d = filter(r -> r.cue_group == "dual", df_b)

    # Keep condition order aligned
    @assert String.(df_a_s.condition) == String.(df_b_s.condition) "Single-cue conditions differ between modes."
    @assert String.(df_a_d.condition) == String.(df_b_d.condition) "Dual-cue conditions differ between modes."

    p1 = overlay_accuracy_panel(df_a_s, df_b_s, "Single-cue accuracy", mode_a_label, mode_b_label)
    p2 = overlay_accuracy_panel(df_a_d, df_b_d, "Dual-cue accuracy", mode_a_label, mode_b_label)
    p3 = overlay_rt_panel(df_a_s, df_b_s, "Single-cue RT quantiles (0.1/0.5/0.9)")
    p4 = overlay_rt_panel(df_a_d, df_b_d, "Dual-cue RT quantiles (0.1/0.5/0.9)")

    fig = plot(
        p1, p2, p3, p4;
        layout = (2, 2),
        size = (1200, 850),
        plot_title = "sub$(sub_num)  |  black=data  |  blue=$(tag_a)  |  orange=$(tag_b)"
    )

    isdir(fig_dir) || mkpath(fig_dir)
    out = joinpath(fig_dir, "sub$(sub_num)_compare_$(tag_a)_vs_$(tag_b).png")
    savefig(fig, out)
    println("Saved compare figure: ", out)
    return out
end

# =============================================================================
# MAIN
# =============================================================================
@assert isfile(fit_script) "Missing fit script: $(fit_script)"
isdir(pred_dir) || mkpath(pred_dir)
isdir(fig_dir) || mkpath(fig_dir)

tag_a = mode_tag_from_args(mode_a_args)
tag_b = mode_tag_from_args(mode_b_args)

println("Compare modes:")
println("  A: ", join(mode_a_args, " "), "  -> tag ", tag_a)
println("  B: ", join(mode_b_args, " "), "  -> tag ", tag_b)
println("Subjects: ", subject_range)

for sub_num in collect(subject_range)
    println("="^60)
    println("Subject sub$(sub_num)")

    if !subject_is_usable(sub_num)
        continue
    end

    path_a = ensure_prediction!(sub_num, mode_a_args, tag_a)
    path_b = ensure_prediction!(sub_num, mode_b_args, tag_b)
    if path_a === nothing || path_b === nothing
        println("  skip sub$(sub_num): missing prediction for one or both modes")
        continue
    end

    df_a = load_pred(path_a)
    df_b = load_pred(path_b)
    save_compare_figure(sub_num, df_a, df_b, tag_a, tag_b)
end

println("="^60)
println("Done. Compare figures in: ", fig_dir)
println("Prediction CSVs in: ", pred_dir)
