# Stratified bootstrap for MIS+LBA: dual-cue-only vs full (all cue) datasets.
# Copied from mis_lba_basic_bootstrap.jl; adds a data_scope dimension.
#
# Run:  julia mis_lba_cue_scope_bootstrap.jl
#
# Switch mode on ONE line below:
#   RUN_MODE = :check  ->  smoke test (~8 fits, seconds)
#   RUN_MODE = :full   ->  production (~1500 fits, ~2 h for one subject)
#
# Re-run safely: results CSV is appended/resumed; set FORCE_REFIT=true to start over.

using CSV, DataFrames, Statistics, Optim, SequentialSamplingModels
using Random, Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE THIS ONE LINE to switch check vs full run:
const RUN_MODE = :check   # :check  or  :full
# ═══════════════════════════════════════════════════════════════════════════════

const SUBJECT_ID = "S2"
const RANDOM_SEED = 20260616
const FORCE_REFIT = true
const RESUME_PARTIAL = true
const SAVE_EVERY = 10
const OPT_OPTIONS = Optim.Options(g_tol=1e-4, x_abstol=1e-6, f_reltol=1e-8, iterations=10_000)

const RUN_TAG = RUN_MODE == :check ? "check" : "full"
const TRIAL_FRACTIONS = RUN_MODE == :check ? [0.25, 1.00] : [0.10, 0.25, 0.50, 0.75, 1.00]
const N_BOOTSTRAP_REPS = RUN_MODE == :check ? 2 : 150
const PROGRESS_PCT_STEP = RUN_MODE == :check ? 25.0 : 1.0

# Fit on each scope; compare dual-cue trials only vs all single+dual trials.
const DATA_SCOPES = [:dual_cue_only, :full]

@assert RUN_MODE in (:check, :full) "RUN_MODE must be :check or :full"

prob_model = :noise_model
epsilon = 0.05
@assert prob_model in (:original, :noise_model)
@assert 0.0 <= epsilon <= 1.0

vary_CI_by_cue_number = false
vary_k_by_cue_number = false
vary_t0_by_cue_number = false

const CCP001_BASE = joinpath(@__DIR__, "..", "data", "CCP001")
const CCP001_PROCESSED_FALLBACK = joinpath(
    @__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab", "extracted_data_processed"
)
const MIN_SESSION = 6
const MIN_RT_MS = 100.0
const SINGLE_CUE_CONDITIONS = Set(["(1)", "(2)", "(3)", "(4)"])
const DUAL_CUE_CONDITIONS = Set(["(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"])
const ALLOWED_CONDITIONS = union(SINGLE_CUE_CONDITIONS, DUAL_CUE_CONDITIONS)

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]

const OUT_DIR = joinpath(@__DIR__, "figs", "figs_jun16", "bootstrap")
const RESULTS_CSV = joinpath(OUT_DIR, "ccp001_$(lowercase(SUBJECT_ID))_cue_scope_bootstrap_params_$(RUN_TAG).csv")
const PLOT_SCRIPT = joinpath(@__DIR__, "plot_ccp001_cue_scope_bootstrap_density.jl")

const DATA_SCOPE_LABELS = Dict(
    :dual_cue_only => "dual_cue_only",
    :full => "all_cues",
)

# --- CCP001 data loading ---

function cuevalues_to_rewards(v)::Vector{Int}
    s = lpad(string(v), 4, '0')
    @assert ncodeunits(s) == 4 "CueValues must map to 4 digits, got: $v"
    parse.(Int, collect(s))
end

function derive_condition_label(v)::Union{String, Nothing}
    rewards = cuevalues_to_rewards(v)
    non_zero = filter(>(0), rewards)
    isempty(non_zero) && return nothing
    cond = join(sort(unique(non_zero); rev=true), ",")
    label = "($cond)"
    label in ALLOWED_CONDITIONS ? label : nothing
end

function parse_choice_value(candidate)::Union{Int, Nothing}
    if candidate isa Integer
        val = Int(candidate)
    else
        val = try
            parse(Int, string(candidate))
        catch
            return nothing
        end
    end
    val in 1:4 ? val : nothing
end

function resolve_choice(resp_loc, point_target_response)::Union{Int, Nothing}
    for candidate in (resp_loc, point_target_response)
        val = parse_choice_value(candidate)
        val !== nothing && return val
    end
    return nothing
end

function resolve_csv_path(data_dir::String, file_path::String)
    if endswith(file_path, "_trials.csv")
        return file_path
    end
    processed = joinpath(CCP001_PROCESSED_FALLBACK, basename(file_path))
    isfile(processed) ? processed : file_path
end

function list_trial_csv_files(data_dir::String)
    filter(f -> endswith(f, ".csv") && !occursin("metadata", basename(f)), readdir(data_dir; join=true))
end

function discover_ccp001_subject_dir(base_dir::String, subject_id::String)
    m = match(r"^S(\d+)$", uppercase(strip(subject_id)))
    m === nothing && error("SUBJECT_ID must look like S1, S2, ...; got: $(subject_id)")
    sub_num = m.captures[1]
    subj_dir = joinpath(base_dir, "CCP001 - sub $(sub_num)")
    isdir(subj_dir) || error("Subject folder not found: $(subj_dir)")
    subj_dir
end

function load_ccp001_subject(data_dir::String)::DataFrame
    files = list_trial_csv_files(data_dir)
    @assert !isempty(files) "No trial CSV files found in: $(data_dir)"

    col_types = Dict(:Cues => String, :CueRanks => String)
    frames = DataFrame[]
    for f in files
        src = resolve_csv_path(data_dir, f)
        hdr = names(CSV.read(src, DataFrame; limit=0))
        has_warmup = "WarmUpTrial" in string.(hdr)
        has_rt = "RT" in string.(hdr)
        @assert has_rt "RT column missing in: $(src)"
        if has_warmup
            push!(frames, CSV.read(src, DataFrame;
                types=col_types,
                select=[:Session, :WarmUpTrial, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition, :RT]
            ))
        else
            df = CSV.read(src, DataFrame;
                types=col_types,
                select=[:Session, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition, :RT]
            )
            df.WarmUpTrial = ones(Int, nrow(df))
            select!(df, [:Session, :WarmUpTrial, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition, :RT])
            push!(frames, df)
        end
    end

    data = reduce(vcat, frames; cols=:setequal)
    rows = NamedTuple[]
    for row in eachrow(data)
        Int(row.Session) < MIN_SESSION && continue
        Int(row.WarmUpTrial) != 0 && continue
        choice = resolve_choice(row.RespLoc, row.PointTargetResponse)
        choice === nothing && continue
        rt_ms = Float64(row.RT)
        rt_ms < MIN_RT_MS && continue
        cond = string(row.CueCondition)
        if !(startswith(cond, "(") && cond in ALLOWED_CONDITIONS)
            cond = derive_condition_label(row.CueValues)
            cond === nothing && continue
        end
        push!(rows, (
            trial_id = length(rows) + 1,
            Session = Int(row.Session),
            CueValues = row.CueValues,
            Choice = choice,
            CueCondition = cond,
            RT_ms = rt_ms
        ))
    end
    DataFrame(rows)
end

function filter_by_data_scope(data::DataFrame, scope::Symbol)::DataFrame
    if scope == :full
        return data
    elseif scope == :dual_cue_only
        return filter(row -> string(row.CueCondition) in DUAL_CUE_CONDITIONS, data)
    else
        error("Unknown data_scope: $(scope)")
    end
end

function data_scope_label(scope::Symbol)::String
    get(DATA_SCOPE_LABELS, scope) do
        error("Unknown data_scope: $(scope)")
    end
end

# --- Stratified subsample / bootstrap ---

function stratified_sample!(
    out_idx::Vector{Int},
    data::DataFrame,
    fraction::Float64,
    rep::Int,
    scope::Symbol,
    rng::AbstractRNG
)
    empty!(out_idx)
    frac_label = round(Int, 100 * fraction)
    scope_tag = data_scope_label(scope)
    for cond in sort(unique(data.CueCondition))
        cond_idx = findall(data.CueCondition .== cond)
        n_c = length(cond_idx)
        n_c == 0 && continue
        cond_seed = hash((SUBJECT_ID, scope_tag, frac_label, rep, cond, RANDOM_SEED))
        cond_rng = MersenneTwister(cond_seed)
        if fraction >= 1.0 - 1e-9
            draw = rand(cond_rng, cond_idx, n_c)
        else
            n_take = max(1, floor(Int, n_c * fraction))
            n_take = min(n_take, n_c)
            draw = cond_idx[randperm(cond_rng, n_c)[1:n_take]]
        end
        append!(out_idx, draw)
    end
    sort!(out_idx)
    out_idx
end

function fraction_label(fraction::Float64)::String
    "$(round(Int, 100 * fraction))pct"
end

# --- Model building / fitting ---

function t0_upper_bound(rt_sec::Vector{Float64})
    upper = minimum(rt_sec) - 0.001
    upper > 0.001 || (upper = quantile(rt_sec, 0.05) - 0.001)
    @assert upper > 0.001 "RTs too fast to estimate positive t0 (min=$(minimum(rt_sec)) s)."
    upper
end

function prepare_lba_trials(data::DataFrame)
    trial_rewards_arrarr = [cuevalues_to_rewards(v) for v in data.CueValues]
    chosen_idx = data.Choice
    rt_sec = Float64.(data.RT_ms) ./ 1000
    r_max = maximum(vcat(trial_rewards_arrarr...))
    (trial_rewards_arrarr, chosen_idx, rt_sec, r_max)
end

function build_lba_indices(t0_upper::Float64)
    param_names = String[]
    lower = Float64[]
    upper = Float64[]
    x0 = Float64[]

    push!(param_names, "theta")
    push!(lower, 0.0); push!(upper, 30.0); push!(x0, 1.0)
    idx_theta = 1

    push!(param_names, "omega0")
    push!(lower, 1e-6); push!(upper, 50.0); push!(x0, 0.1)
    idx_omega0 = 2

    idx_CI = Dict{Symbol, Int}()
    idx_k = Dict{Symbol, Int}()
    idx_t0 = Dict{Symbol, Int}()

    for key in (vary_CI_by_cue_number ? [:single, :multi] : [:all])
        push!(param_names, "CI_$(key)")
        push!(lower, 0.01); push!(upper, 100.0); push!(x0, 5.0)
        idx_CI[key] = length(param_names)
    end

    push!(param_names, "A")
    push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
    idx_A = length(param_names)

    for key in (vary_k_by_cue_number ? [:single, :multi] : [:all])
        push!(param_names, "k_$(key)")
        push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
        idx_k[key] = length(param_names)
    end

    for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
        push!(param_names, "t0_$(key)")
        push!(lower, 0.001); push!(upper, t0_upper); push!(x0, min(0.2, t0_upper / 2))
        idx_t0[key] = length(param_names)
    end

    (
        param_names=param_names, lower=lower, upper=upper, x0=x0,
        idx_theta=idx_theta, idx_omega0=idx_omega0,
        idx_CI=idx_CI, idx_A=idx_A, idx_k=idx_k, idx_t0=idx_t0
    )
end

function mis_lba_neg_loglik(
    x::Vector{Float64},
    trial_rewards_arrarr,
    chosen_idx,
    rt_sec,
    r_max::Int,
    cue_number_type,
    idx_theta::Int,
    idx_omega0::Int,
    idx_CI::Dict{Symbol, Int},
    idx_A::Int,
    idx_k::Dict{Symbol, Int},
    idx_t0::Dict{Symbol, Int};
    prob_model::Symbol,
    epsilon::Float64
)
    theta = x[idx_theta]
    omega0 = x[idx_omega0]
    A = x[idx_A]
    eps_val = 1e-12
    (A <= 0) && return Inf
    nll = 0.0
    for i in eachindex(trial_rewards_arrarr)
        cue_key = cue_number_type[i]
        CI = x[idx_CI[vary_CI_by_cue_number ? cue_key : :all]]
        k = x[idx_k[vary_k_by_cue_number ? cue_key : :all]]
        t0 = x[idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        (CI <= 0 || k <= 0 || t0 <= 0) && return Inf
        rt_sec[i] <= t0 && return Inf
        r = trial_rewards_arrarr[i]
        w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in r]
        p = w ./ sum(w)
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
        end
        drift_rates = CI .* p
        lba = try
            LBA(ν=drift_rates, A=A, k=k, τ=t0)
        catch
            return Inf
        end
        lik = try
            pdf(lba, (choice=chosen_idx[i], rt=rt_sec[i]))
        catch
            eps_val
        end
        nll -= log(max(lik, eps_val))
    end
    nll
end

function fit_subset(data::DataFrame)
    trial_rewards_arrarr, chosen_idx, rt_sec, r_max = prepare_lba_trials(data)
    cue_number_type = [count(!=(0), r) == 1 ? :single : :multi for r in trial_rewards_arrarr]
    t0_upper = t0_upper_bound(rt_sec)
    spec = build_lba_indices(t0_upper)
    x0 = copy(spec.x0)
    obj = x -> mis_lba_neg_loglik(
        x, trial_rewards_arrarr, chosen_idx, rt_sec, r_max, cue_number_type,
        spec.idx_theta, spec.idx_omega0, spec.idx_CI, spec.idx_A, spec.idx_k, spec.idx_t0;
        prob_model, epsilon
    )
    t0 = time()
    fit = optimize(obj, spec.lower, spec.upper, x0, Fminbox(NelderMead()), OPT_OPTIONS)
    elapsed = time() - t0
    best = Optim.minimizer(fit)
    nll = Optim.minimum(fit)
    fit_nt = NamedTuple{Tuple(Symbol.(spec.param_names))}(Tuple(best))
    (
        n_trials = nrow(data),
        theta = fit_nt.theta,
        omega0 = fit_nt.omega0,
        CI = fit_nt.CI_all,
        A = fit_nt.A,
        k = fit_nt.k_all,
        t0 = fit_nt.t0_all,
        nll = nll,
        elapsed_sec = elapsed
    )
end

# --- Progress reporting ---

mutable struct ProgressTracker
    total::Int
    done::Int
    next_milestone::Float64
    step::Float64
    t_start::Float64
end

function ProgressTracker(total::Int, step::Float64)
    ProgressTracker(total, 0, step, step, time())
end

function tick!(tracker::ProgressTracker)
    tracker.done += 1
    pct = 100.0 * tracker.done / tracker.total
    while pct + 1e-9 >= tracker.next_milestone && tracker.next_milestone <= 100.0
        elapsed = time() - tracker.t_start
        @printf(
            "  Progress: %5.1f%%  (%d / %d fits)  elapsed %s\n",
            tracker.next_milestone,
            tracker.done,
            tracker.total,
            format_duration(elapsed)
        )
        tracker.next_milestone += tracker.step
    end
end

function format_duration(sec::Float64)::String
    s = Int(round(sec))
    h, r = divrem(s, 3600)
    m, s = divrem(r, 60)
    h > 0 ? @sprintf("%dh %02dm %02ds", h, m, s) : @sprintf("%dm %02ds", m, s)
end

# --- Results I/O ---

function expected_total_fits()::Int
    length(DATA_SCOPES) * length(TRIAL_FRACTIONS) * N_BOOTSTRAP_REPS
end

function load_existing_results(path::String)::DataFrame
    if !isfile(path)
        return DataFrame(
            subject_id = String[],
            data_scope = String[],
            trial_fraction = Float64[],
            fraction_label = String[],
            rep = Int[],
            n_trials = Int[],
            theta = Float64[],
            omega0 = Float64[],
            CI = Float64[],
            A = Float64[],
            k = Float64[],
            t0 = Float64[],
            nll = Float64[],
            elapsed_sec = Float64[],
            fitted_at = String[]
        )
    end
    df = CSV.read(path, DataFrame)
    col_names = Symbol.(names(df))
    required = [:subject_id, :data_scope, :trial_fraction, :rep, :theta, :omega0, :CI, :A, :k, :t0]
    for col in required
        col in col_names || error("Existing results missing column $(col): $(path)")
    end
    df
end

function result_done(existing::DataFrame, scope::Symbol, fraction::Float64, rep::Int)::Bool
    scope_tag = data_scope_label(scope)
    !isempty(existing) &&
        any(
            (existing.data_scope .== scope_tag) .&
            (existing.trial_fraction .≈ fraction) .&
            (existing.rep .== rep)
        )
end

function append_result_row!(
    rows::Vector{NamedTuple},
    scope::Symbol,
    fraction::Float64,
    rep::Int,
    fit
)
    push!(rows, (
        subject_id = SUBJECT_ID,
        data_scope = data_scope_label(scope),
        trial_fraction = fraction,
        fraction_label = fraction_label(fraction),
        rep = rep,
        n_trials = fit.n_trials,
        theta = fit.theta,
        omega0 = fit.omega0,
        CI = fit.CI,
        A = fit.A,
        k = fit.k,
        t0 = fit.t0,
        nll = fit.nll,
        elapsed_sec = fit.elapsed_sec,
        fitted_at = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
    ))
end

function save_results!(path::String, existing::DataFrame, new_rows::Vector{NamedTuple})
    isempty(new_rows) && return existing
    new_df = DataFrame(new_rows)
    combined = isempty(existing) ? new_df : vcat(existing, new_df; cols=:union)
    sort!(combined, [:data_scope, :trial_fraction, :rep])
    CSV.write(path, combined)
    combined
end

# --- Main ---

function run_bootstrap()
    mkpath(OUT_DIR)
    @assert isdir(CCP001_BASE) "CCP001 data folder not found: $(CCP001_BASE)"

    subj_dir = discover_ccp001_subject_dir(CCP001_BASE, SUBJECT_ID)
    full_data = load_ccp001_subject(subj_dir)
    @assert nrow(full_data) > 0 "No usable trials for $(SUBJECT_ID)"

    scope_data = Dict(scope => filter_by_data_scope(full_data, scope) for scope in DATA_SCOPES)
    for scope in DATA_SCOPES
        @assert nrow(scope_data[scope]) > 0 "No trials for scope $(scope) ($(SUBJECT_ID))"
    end

    total_fits = expected_total_fits()
    println("=" ^ 72)
    println("CCP001 MIS+LBA cue-scope bootstrap (dual-cue vs all cues)")
    println("=" ^ 72)
    println("RUN_MODE:       ", RUN_MODE, "  (tag=", RUN_TAG, ")")
    println("Subject:        ", SUBJECT_ID)
    println("Data dir:       ", subj_dir)
    println("Full trials:    ", nrow(full_data))
    for scope in DATA_SCOPES
        sd = scope_data[scope]
        println(
            "  $(data_scope_label(scope)): ",
            nrow(sd),
            " trials, ",
            length(unique(sd.CueCondition)),
            " conditions"
        )
    end
    println("Data scopes:    ", [data_scope_label(s) for s in DATA_SCOPES])
    println("Fractions:      ", TRIAL_FRACTIONS)
    println("Reps/fraction:  ", N_BOOTSTRAP_REPS)
    println("Total fits:     ", total_fits)
    println("Progress step:  ", PROGRESS_PCT_STEP, "%")
    println("Results CSV:    ", RESULTS_CSV)
    println("prob_model:     ", prob_model, "  epsilon=", epsilon)
    if RUN_MODE == :check
        println("NOTE: check mode — only ", total_fits, " fits to verify CSV + plots.")
    else
        est_min = round(Int, total_fits * 4 / 60)
        println("NOTE: full mode — expect ~", est_min, "–", est_min + 30, " min for this subject.")
    end
    println("=" ^ 72)

    if FORCE_REFIT && isfile(RESULTS_CSV)
        println("FORCE_REFIT=true -> removing existing results: ", RESULTS_CSV)
        rm(RESULTS_CSV)
    end

    existing = load_existing_results(RESULTS_CSV)
    if !FORCE_REFIT && nrow(existing) >= total_fits
        println("Found complete results (", nrow(existing), " rows). Skipping fits.")
        println("Set FORCE_REFIT=true to rerun from scratch.")
        return existing
    end

    if RESUME_PARTIAL && nrow(existing) > 0
        println("Resuming: ", nrow(existing), " fits already saved.")
    end

    rng = MersenneTwister(RANDOM_SEED)
    sample_idx = Int[]
    tracker = ProgressTracker(total_fits, PROGRESS_PCT_STEP)
    tracker.done = nrow(existing)
    if tracker.done > 0
        tracker.next_milestone = PROGRESS_PCT_STEP * ceil(tracker.done / total_fits * 100 / PROGRESS_PCT_STEP)
    end

    new_rows = NamedTuple[]
    batch_rows = NamedTuple[]

    println("Fit start: fresh default x0 for every bootstrap rep")
    println("\nStarting fits for subject ", SUBJECT_ID, " ...\n")

    for scope in DATA_SCOPES
        scope_df = scope_data[scope]
        println("=== Data scope: ", data_scope_label(scope), " (", nrow(scope_df), " trials) ===")
        for fraction in TRIAL_FRACTIONS
            println("--- Trial fraction: ", fraction_label(fraction), " ($(fraction)) ---")
            for rep in 1:N_BOOTSTRAP_REPS
                if RESUME_PARTIAL && result_done(existing, scope, fraction, rep)
                    continue
                end

                stratified_sample!(sample_idx, scope_df, fraction, rep, scope, rng)
                sub_data = scope_df[sample_idx, :]
                fit = fit_subset(sub_data)

                append_result_row!(batch_rows, scope, fraction, rep, fit)
                if length(batch_rows) >= SAVE_EVERY
                    existing = save_results!(RESULTS_CSV, existing, batch_rows)
                    append!(new_rows, batch_rows)
                    empty!(batch_rows)
                end

                tick!(tracker)
            end
        end
    end

    if !isempty(batch_rows)
        existing = save_results!(RESULTS_CSV, existing, batch_rows)
        append!(new_rows, batch_rows)
    end

    println("\nFinished fitting. Saved ", nrow(existing), " rows -> ", RESULTS_CSV)
    summarize_bootstrap_results(existing)
    existing
end

function summarize_bootstrap_results(results::DataFrame)
    println("\n--- Cue-scope bootstrap summary ---")
    med_elapsed = median(results.elapsed_sec)
    println("Median fit time: ", round(med_elapsed, digits=3), " s")
    if med_elapsed < 0.1
        println("WARNING: median fit time < 0.1 s — fits may not be re-optimizing properly.")
    end
    for scope in sort(unique(results.data_scope))
        sub_scope = results[results.data_scope .== scope, :]
        println("Scope ", scope, ":")
        for f in sort(unique(sub_scope.trial_fraction))
            sub = sub_scope[sub_scope.trial_fraction .≈ f, :]
            println(
                "  ",
                fraction_label(f),
                ": theta unique=", length(unique(sub.theta)),
                ", t0 unique=", length(unique(sub.t0)),
                ", mean elapsed=", round(mean(sub.elapsed_sec), digits=3), "s"
            )
        end
    end
end

function plot_from_results(results::DataFrame)
    println("\nBootstrap fits saved (", nrow(results), " rows). Plotting densities ...")
    if !isfile(PLOT_SCRIPT)
        println("WARNING: plot script not found: ", PLOT_SCRIPT)
        return nothing
    end
    include(PLOT_SCRIPT)
    Base.invokelatest() do
        plot_ccp001_cue_scope_bootstrap_density(; subject_id=SUBJECT_ID, run_tag=RUN_TAG)
    end
end

results = run_bootstrap()
plot_from_results(results)
println("\nDone.")
