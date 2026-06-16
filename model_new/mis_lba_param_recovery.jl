# MIS+LBA parameter recovery for CCP001 (sessions 6–17 trial design).
# Run:  julia mis_lba_param_recovery.jl
#
# Switch smoke test vs full run with ONE line:
#   const RUN_MODE = :check   # fast smoke test (~1–2 min)
#   const RUN_MODE = :full    # ~1 hr recovery run

using CSV, DataFrames, Statistics, Optim, Plots, SequentialSamplingModels, KernelDensity
using Random, Printf, Dates

# --- Run mode (change this one line) ---
const RUN_MODE = :check  # :check | :full

# --- Shared model / data config ---
const SUBJECT_ID = "S2"
const DESIGN_SUBJECT_ID = "S1"   # whose session 6–17 trial schedule to mirror
const RANDOM_SEED = 20260616

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
const MAX_SESSION = 17
const MIN_RT_MS = 100.0
const SINGLE_CUE_CONDITIONS = Set(["(1)", "(2)", "(3)", "(4)"])
const DUAL_CUE_CONDITIONS = Set(["(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"])
const ALLOWED_CONDITIONS = union(SINGLE_CUE_CONDITIONS, DUAL_CUE_CONDITIONS)

const PARAM_COLS = [:theta, :omega0, :CI, :A, :k, :t0]
const PARAM_LABELS = ["theta", "omega0", "CI", "A", "k", "t0"]

# Representative ground truth (S2-like; sessions 6–11 fit).
const GROUND_TRUTH = (
    theta = 8.0,
    omega0 = 0.20,
    CI = 4.0,
    A = 1.2,
    k = 0.75,
    t0 = 0.36,
)

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16")
const RESULTS_CSV = joinpath(FIG_DIR, "ccp001_recovery.csv")
const DIST_FIG = joinpath(FIG_DIR, "ccp001_recovery_dist.png")

# --- Mode-specific workload ---
function recovery_settings(mode::Symbol, n_design::Int)
    if mode == :check
        design_cap = min(250, n_design)
        trial_levels = [80, min(150, design_cap)]
        n_reps = 2
    elseif mode == :full
        design_cap = n_design
        trial_levels = unique(sort([
            max(400, round(Int, 0.20 * n_design)),
            max(800, round(Int, 0.40 * n_design)),
            max(1500, round(Int, 0.65 * n_design)),
            n_design,
        ]))
        n_reps = 20
    else
        error("RUN_MODE must be :check or :full; got $(mode)")
    end
    (design_cap=design_cap, trial_levels=trial_levels, n_reps=n_reps)
end

# --- Data loading ---

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
    m === nothing && error("Subject ID must look like S1, S2, ...; got: $(subject_id)")
    subj_dir = joinpath(base_dir, "CCP001 - sub $(m.captures[1])")
    isdir(subj_dir) || error("Subject folder not found: $(subj_dir)")
    subj_dir
end

function load_ccp001_design(data_dir::String)::DataFrame
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
        sess = Int(row.Session)
        sess < MIN_SESSION && continue
        sess > MAX_SESSION && continue
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
            Session = sess,
            CueValues = row.CueValues,
            CueCondition = cond,
        ))
    end
    DataFrame(rows)
end

# --- MIS + LBA helpers ---

function mis_choice_probs(r::Vector{Int}, theta::Float64, omega0::Float64, r_max::Int)
    w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in r]
    p = w ./ sum(w)
    if prob_model == :noise_model
        p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
    end
    p
end

function simulate_dataset(design::DataFrame, truth::NamedTuple, rng::AbstractRNG)::DataFrame
    rewards_list = [cuevalues_to_rewards(v) for v in design.CueValues]
    r_max = maximum(vcat(rewards_list...))
    choices = Int[]
    rts_ms = Float64[]
    for r in rewards_list
        p = mis_choice_probs(r, truth.theta, truth.omega0, r_max)
        ν = truth.CI .* p
        lba = LBA(ν=ν, A=truth.A, k=truth.k, τ=truth.t0)
        sample = rand(rng, lba)
        push!(choices, sample.choice)
        push!(rts_ms, sample.rt * 1000)
    end
    DataFrame(
        trial_id = design.trial_id,
        CueValues = design.CueValues,
        CueCondition = design.CueCondition,
        Choice = choices,
        RT_ms = rts_ms,
    )
end

function stratified_subsample_indices(data::DataFrame, n_take::Int, rep::Int, rng::AbstractRNG)::Vector{Int}
    n_all = nrow(data)
    n_take >= n_all && return collect(1:n_all)
    n_take <= 0 && return Int[]

    out = Int[]
    conditions = sort(unique(data.CueCondition))
    counts = Dict(c => count(==(c), data.CueCondition) for c in conditions)
    remaining = n_take
    for (i, cond) in enumerate(conditions)
        cond_idx = findall(data.CueCondition .== cond)
        if i == length(conditions)
            n_c = min(length(cond_idx), remaining)
        else
            n_c = min(length(cond_idx), max(1, round(Int, n_take * counts[cond] / n_all)))
            remaining -= n_c
        end
        n_c == 0 && continue
        perm = randperm(rng, length(cond_idx))
        append!(out, cond_idx[perm[1:n_c]])
    end
    while length(out) < n_take
        extra = rand(rng, setdiff(1:n_all, out))
        push!(out, extra)
    end
    length(out) > n_take && (out = out[1:n_take])
    sort!(out)
    out
end

function t0_upper_bound(rt_sec::Vector{Float64})
    upper = minimum(rt_sec) - 0.001
    upper > 0.001 || (upper = quantile(rt_sec, 0.05) - 0.001)
    @assert upper > 0.001 "RTs too fast to estimate positive t0."
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
    x, trial_rewards_arrarr, chosen_idx, rt_sec, r_max, cue_number_type,
    idx_theta, idx_omega0, idx_CI, idx_A, idx_k, idx_t0;
    prob_model::Symbol, epsilon::Float64
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
        p = mis_choice_probs(r, theta, omega0, r_max)
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

function fit_simulated(data::DataFrame)
    trial_rewards_arrarr, chosen_idx, rt_sec, r_max = prepare_lba_trials(data)
    cue_number_type = [count(!=(0), r) == 1 ? :single : :multi for r in trial_rewards_arrarr]
    t0_upper = t0_upper_bound(rt_sec)
    spec = build_lba_indices(t0_upper)
    obj = x -> mis_lba_neg_loglik(
        x, trial_rewards_arrarr, chosen_idx, rt_sec, r_max, cue_number_type,
        spec.idx_theta, spec.idx_omega0, spec.idx_CI, spec.idx_A, spec.idx_k, spec.idx_t0;
        prob_model, epsilon
    )
    t0 = time()
    fit = optimize(obj, spec.lower, spec.upper, spec.x0, Fminbox(NelderMead()))
    elapsed = time() - t0
    best = Optim.minimizer(fit)
    (
        theta = best[spec.idx_theta],
        omega0 = best[spec.idx_omega0],
        CI = best[spec.idx_CI[:all]],
        A = best[spec.idx_A],
        k = best[spec.idx_k[:all]],
        t0 = best[spec.idx_t0[:all]],
        nll = Optim.minimum(fit),
        n_trials = nrow(data),
        elapsed_sec = elapsed,
    )
end

# --- Plotting ---

function safe_kde(values::Vector{Float64})
    v = filter(isfinite, values)
    length(v) < 2 && return nothing
    try
        kde(v)
    catch
        nothing
    end
end

function plot_recovery_distributions(results::DataFrame, truth::NamedTuple, fig_path::String)
    trial_levels = sort(unique(results.n_trials_fit))
    n_levels = length(trial_levels)
    plots = Vector{Any}(undef, length(PARAM_COLS))

    for (i, col) in enumerate(PARAM_COLS)
        p = plot(
            legend=(i == 1 ? :topright : false),
            title=PARAM_LABELS[i],
            xlabel="recovered",
            ylabel="density",
        )
        true_val = Float64(truth[col])
        for (j, n_fit) in enumerate(trial_levels)
            sub = results[results.n_trials_fit .== n_fit, :]
            ys = Vector{Float64}(sub[!, col])
            dens = safe_kde(ys)
            label = "n=$(n_fit)"
            if dens !== nothing
                plot!(p, dens.x, dens.density; label=label, color=j, lw=2)
            else
                scatter!(p, ys, zeros(length(ys)); label=label, color=j, ms=4)
            end
            vline!(p, [true_val]; label=(j == 1 ? "true" : false), color=:black, lw=2, ls=:dash)
        end
        plots[i] = p
    end

    fig = plot(
        plots...;
        layout=(length(PARAM_COLS), 1),
        size=(min(900, 200 * n_levels + 120), 220 * length(PARAM_COLS)),
        plot_title="CCP001 MIS+LBA recovery ($(RUN_MODE), design=$(DESIGN_SUBJECT_ID) ses$(MIN_SESSION)-$(MAX_SESSION))",
        bottom_margin=3Plots.mm,
    )
    savefig(fig, fig_path)
end

# --- Main ---

@assert RUN_MODE in (:check, :full)

println("=" ^ 72)
println("CCP001 MIS+LBA parameter recovery  |  RUN_MODE = $(RUN_MODE)")
println("=" ^ 72)

mkpath(FIG_DIR)
rng = MersenneTwister(RANDOM_SEED)

design_dir = discover_ccp001_subject_dir(CCP001_BASE, DESIGN_SUBJECT_ID)
design_full = load_ccp001_design(design_dir)
n_design_full = nrow(design_full)
@assert n_design_full > 0 "No design trials for $(DESIGN_SUBJECT_ID) sessions $(MIN_SESSION)-$(MAX_SESSION)"

cfg = recovery_settings(RUN_MODE, n_design_full)
design = design_full[1:min(cfg.design_cap, n_design_full), :]
n_design = nrow(design)
trial_levels = filter(n -> n <= n_design, cfg.trial_levels)
@assert !isempty(trial_levels) "No trial levels <= design size $(n_design)"

println("Design subject: ", DESIGN_SUBJECT_ID, "  (sessions ", MIN_SESSION, "–", MAX_SESSION, ")")
println("Full design trials: ", n_design_full)
println("Design trials used: ", n_design)
println("Ground truth: ", GROUND_TRUTH)
println("Trial levels: ", trial_levels)
println("Replications per level: ", cfg.n_reps)
println("Total fits: ", length(trial_levels) * cfg.n_reps)

total_fits = length(trial_levels) * cfg.n_reps
fit_done = 0
t_start = time()

rows = NamedTuple[]
for rep in 1:cfg.n_reps
    sim_rng = MersenneTwister(hash((RANDOM_SEED, rep, "simulate")))
    sim_data = simulate_dataset(design, GROUND_TRUTH, sim_rng)

    for n_fit in trial_levels
        sub_rng = MersenneTwister(hash((RANDOM_SEED, rep, n_fit, "subsample")))
        idx = stratified_subsample_indices(sim_data, n_fit, rep, sub_rng)
        fit_data = sim_data[idx, :]
        fit = fit_simulated(fit_data)
        global fit_done += 1

        if RUN_MODE == :full && (fit_done == 1 || fit_done % 5 == 0 || fit_done == total_fits)
            elapsed = time() - t_start
            eta = elapsed / fit_done * (total_fits - fit_done)
            @printf("  fit %d / %d  (n=%d, rep=%d)  elapsed=%.0fs  ETA=%.0fs\n",
                fit_done, total_fits, n_fit, rep, elapsed, eta)
        end

        push!(rows, (
            run_mode = String(RUN_MODE),
            design_subject = DESIGN_SUBJECT_ID,
            n_design = n_design,
            n_trials_fit = n_fit,
            rep = rep,
            true_theta = GROUND_TRUTH.theta,
            true_omega0 = GROUND_TRUTH.omega0,
            true_CI = GROUND_TRUTH.CI,
            true_A = GROUND_TRUTH.A,
            true_k = GROUND_TRUTH.k,
            true_t0 = GROUND_TRUTH.t0,
            theta = fit.theta,
            omega0 = fit.omega0,
            CI = fit.CI,
            A = fit.A,
            k = fit.k,
            t0 = fit.t0,
            nll = fit.nll,
            elapsed_sec = fit.elapsed_sec,
            fitted_at = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"),
        ))
    end
end

results = DataFrame(rows)
CSV.write(RESULTS_CSV, results)
plot_recovery_distributions(results, GROUND_TRUTH, DIST_FIG)

elapsed_total = time() - t_start
println("\nSaved results: ", RESULTS_CSV)
println("Saved figure:  ", DIST_FIG)
@printf("Total runtime: %.1f s (%.1f min)\n", elapsed_total, elapsed_total / 60)

for col in PARAM_COLS
    println("\n", PARAM_LABELS[findfirst(==(col), PARAM_COLS)], " bias / RMSE by n:")
    for n_fit in trial_levels
        sub = results[results.n_trials_fit .== n_fit, :]
        err = Vector{Float64}(sub[!, col]) .- Float64(GROUND_TRUTH[col])
        @printf("  n=%5d  bias=% .4f  rmse=% .4f\n", n_fit, mean(err), sqrt(mean(err .^ 2)))
    end
end

if RUN_MODE == :check
    println("\n[check mode OK] To run full recovery (~1 hr), change line 12 to:")
    println("  const RUN_MODE = :full")
end
