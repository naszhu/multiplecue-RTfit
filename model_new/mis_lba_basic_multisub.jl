# Multi-subject MIS + LBA fit for lab data in:
#   multiplecue-responsebox/exp/data_from_lab/subN
#
# Copied from mis_lba_basic.jl and adapted to fit several subjects in one run.
# Run:  julia mis_lba_basic_multisub.jl

using CSV, DataFrames, Statistics, Optim, Plots, SequentialSamplingModels, Random

# =============================================================================
# CONFIG (change these)
# =============================================================================

# Which subjects to fit. Example: 1:10 or 8:11
# Only folders named exactly "subN" are used (ignore sub1star, sub2*, etc.).
# Subjects with fewer than MIN_N_SESSIONS trial files are skipped (likely testing).
subject_range = 9:11
MIN_N_SESSIONS = 10

# Probability model:
# :original    -> P(i) = w_i / sum_j w_j
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
prob_model = :original
epsilon = 0.05

# How option weights are built from cue values:
# false -> classic MIS: w = exp(theta * r / r_max) for r>0, and omega0 when r==0
# true  -> free weights: omega1..omega4 for cue values 1..4, and omega0 when r==0
#         (no theta / no exponential)
free_omega_by_cue_value = false

# LBA cue-number config (all false = one shared CI, k, t0 for all trials)
vary_CI_by_cue_number = false
vary_k_by_cue_number = false
vary_t0_by_cue_number = false

# How many simulated datasets to build prediction means + 95% intervals
n_pred_sims = 100
RANDOM_SEED = 20260801

# Keep sessions from this number onward (same rule as mis_lba_basic.jl)
MIN_SESSION = 6
# Drop unrealistically fast RTs (milliseconds). Same idea as other lab scripts.
MIN_RT_MS = 100.0

# Figure output folder
fig_dir = joinpath(@__DIR__, "figs", "fit_aug1")

# =============================================================================
# Checks
# =============================================================================
@assert prob_model in (:original, :noise_model)
@assert 0.0 <= epsilon <= 1.0
@assert n_pred_sims >= 20

lab_data_root = joinpath(
    @__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab"
)
@assert isdir(lab_data_root) "Lab data folder not found: $(lab_data_root)"

condition_order = ["(1)", "(2)", "(3)", "(4)", "(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"]
single_cue_conditions = ["(1)", "(2)", "(3)", "(4)"]
dual_cue_conditions = ["(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"]
rt_quantiles = [0.1, 0.5, 0.9]

# =============================================================================
# Small helper functions (kept simple on purpose)
# =============================================================================

# Turn CueValues like "2300" into reward list [2, 3, 0, 0]
function cuevalues_to_rewards(v)
    s = string(v)
    # Some files store CueValues as a number (e.g. 2300). Pad to 4 digits.
    s = lpad(s, 4, '0')
    @assert length(s) == 4 "CueValues must be 4 characters, got: $(v)"
    return parse.(Int, collect(s))
end

# Choice can be stored as 1..4, or as a one-hot string like "0100"
function parse_choice(resp_loc, point_target_response)
    # Prefer RespLoc when it is already 1..4
    for candidate in (resp_loc, point_target_response)
        if candidate isa Integer
            val = Int(candidate)
            if val in 1:4
                return val
            end
        else
            s = strip(string(candidate))
            # Try plain integer text: "2"
            try
                val = parse(Int, s)
                if val in 1:4
                    return val
                end
            catch
            end
            # Try one-hot text: "0100" -> location 2
            if length(s) == 4 && all(c -> c == '0' || c == '1', s) && count(==('1'), s) == 1
                return findfirst(==('1'), s)
            end
        end
    end
    return nothing
end

# List trial CSV files in one subject folder
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

# Find valid subject numbers that:
# 1) have a folder named exactly "subN"
# 2) have at least MIN_N_SESSIONS trial files
# 3) are inside subject_range
function find_subjects_to_fit(lab_root, subject_range, min_n_sessions)
    wanted = Set(collect(subject_range))
    found = Int[]

    for name in readdir(lab_root)
        m = match(r"^sub(\d+)$", name)
        m === nothing && continue
        sub_num = parse(Int, m.captures[1])
        sub_num in wanted || continue

        subj_dir = joinpath(lab_root, name)
        isdir(subj_dir) || continue

        n_sessions = length(list_trial_csvs(subj_dir))
        if n_sessions < min_n_sessions
            println("Skip sub$(sub_num): only $(n_sessions) session files (< $(min_n_sessions))")
            continue
        end
        push!(found, sub_num)
    end
    return sort(found)
end

# Load and clean one subject
function load_subject_data(subj_dir)
    files = list_trial_csvs(subj_dir)
    @assert !isempty(files) "No trial CSV files in: $(subj_dir)"

    col_types = Dict(
        :Cues => String,
        :CueValues => String,
        :CueRanks => String,
        :RespLoc => String,
        :PointTargetResponse => String,
        :CueCondition => String
    )

    frames = DataFrame[]
    for f in files
        # Skip empty / broken files (e.g. incomplete session dumps)
        if filesize(f) == 0
            println("WARNING: empty file skipped: ", basename(f))
            continue
        end

        hdr = string.(names(CSV.read(f, DataFrame; limit=0)))
        if isempty(hdr)
            println("WARNING: no header, skipped: ", basename(f))
            continue
        end
        required = ["Session", "CueValues", "PointTargetResponse", "CueCondition", "RT"]
        if !all(c -> c in hdr, required)
            println("WARNING: missing required columns, skipped: ", basename(f))
            continue
        end

        has_warmup = "WarmUpTrial" in hdr
        has_resploc = "RespLoc" in hdr

        wanted = [:Session, :CueValues, :PointTargetResponse, :CueCondition, :RT]
        has_warmup && push!(wanted, :WarmUpTrial)
        has_resploc && push!(wanted, :RespLoc)

        df = CSV.read(f, DataFrame; types=col_types, select=wanted, validate=false)
        if !has_warmup
            df.WarmUpTrial = ones(Int, nrow(df))
        end
        if !has_resploc
            df.RespLoc = fill("", nrow(df))
        end
        select!(df, [:Session, :WarmUpTrial, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition, :RT])
        push!(frames, df)
    end
    @assert !isempty(frames) "No usable trial CSV files in: $(subj_dir)"

    raw = reduce(vcat, frames; cols=:setequal)

    # Build a clean trial table
    rows = NamedTuple[]
    for row in eachrow(raw)
        session = try
            Int(row.Session)
        catch
            try
                parse(Int, string(row.Session))
            catch
                continue
            end
        end
        session < MIN_SESSION && continue

        warmup = try
            Int(row.WarmUpTrial)
        catch
            try
                parse(Int, string(row.WarmUpTrial))
            catch
                1
            end
        end
        warmup != 0 && continue

        choice = parse_choice(row.RespLoc, row.PointTargetResponse)
        choice === nothing && continue

        rt_ms = try
            Float64(row.RT)
        catch
            try
                parse(Float64, string(row.RT))
            catch
                continue
            end
        end
        rt_ms < MIN_RT_MS && continue

        cond = string(row.CueCondition)
        if !(cond in condition_order)
            # If CueCondition is weird, derive it from CueValues
            rewards = cuevalues_to_rewards(row.CueValues)
            nonzero = sort(unique(filter(>(0), rewards)); rev=true)
            isempty(nonzero) && continue
            cond = "(" * join(nonzero, ",") * ")"
            cond in condition_order || continue
        end

        push!(rows, (
            Session = session,
            CueValues = string(row.CueValues),
            Choice = choice,
            CueCondition = cond,
            RT_ms = rt_ms
        ))
    end

    return DataFrame(rows)
end

# Short tag used in saved figure names
function omega_mode_tag()
    return free_omega_by_cue_value ? "freeW" : "expW"
end

# Softmax-like choice probabilities from rewards
# x = full parameter vector; layout tells where omega / theta live
function choice_probs(rewards, x, layout, r_max)
    omega0 = x[layout.idx_omega0]
    w = Vector{Float64}(undef, length(rewards))
    for (j, rv) in enumerate(rewards)
        if rv == 0
            w[j] = omega0
        elseif free_omega_by_cue_value
            # Free weight for cue values 1, 2, 3, 4
            w[j] = x[layout.idx_omega[rv]]
        else
            # Classic MIS exponential weight
            theta = x[layout.idx_theta]
            w[j] = exp(theta * rv / r_max)
        end
    end
    p = w ./ sum(w)
    if prob_model == :noise_model
        p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
    end
    return p
end

# Build parameter vector layout for this experiment
function build_param_layout(t0_upper)
    names = String[]
    lower = Float64[]
    upper = Float64[]
    x0 = Float64[]

    idx_theta = 0
    idx_omega = Dict{Int, Int}()

    if free_omega_by_cue_value
        # Free omega1..omega4 (no theta)
        for v in 1:4
            push!(names, "omega$(v)")
            push!(lower, 1e-6); push!(upper, 50.0); push!(x0, Float64(v))
            idx_omega[v] = length(names)
        end
    else
        # Classic MIS: one theta
        push!(names, "theta"); push!(lower, 0.0); push!(upper, 30.0); push!(x0, 1.0)
        idx_theta = length(names)
    end

    # Weight for zero-reward options (used in both modes)
    push!(names, "omega0"); push!(lower, 1e-6); push!(upper, 50.0); push!(x0, 0.1)
    idx_omega0 = length(names)

    idx_CI = Dict{Symbol, Int}()
    for key in (vary_CI_by_cue_number ? [:single, :multi] : [:all])
        push!(names, "CI_$(key)")
        push!(lower, 0.01); push!(upper, 100.0); push!(x0, 5.0)
        idx_CI[key] = length(names)
    end

    push!(names, "A"); push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
    idx_A = length(names)

    idx_k = Dict{Symbol, Int}()
    for key in (vary_k_by_cue_number ? [:single, :multi] : [:all])
        push!(names, "k_$(key)")
        push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
        idx_k[key] = length(names)
    end

    idx_t0 = Dict{Symbol, Int}()
    for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
        push!(names, "t0_$(key)")
        push!(lower, 0.001); push!(upper, t0_upper); push!(x0, min(0.2, t0_upper / 2))
        idx_t0[key] = length(names)
    end

    return (
        names = names,
        lower = lower,
        upper = upper,
        x0 = x0,
        idx_theta = idx_theta,
        idx_omega0 = idx_omega0,
        idx_omega = idx_omega,
        idx_CI = idx_CI,
        idx_A = idx_A,
        idx_k = idx_k,
        idx_t0 = idx_t0
    )
end

# Negative log-likelihood for one subject
function neg_loglik(x, trial_rewards, chosen_idx, rt_sec, cue_type, r_max, layout)
    A = x[layout.idx_A]
    nll = 0.0
    eps_val = 1e-12

    for i in eachindex(trial_rewards)
        cue_key = cue_type[i]
        CI = x[layout.idx_CI[vary_CI_by_cue_number ? cue_key : :all]]
        k = x[layout.idx_k[vary_k_by_cue_number ? cue_key : :all]]
        t0 = x[layout.idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        rt_sec[i] <= t0 && return Inf

        p = choice_probs(trial_rewards[i], x, layout, r_max)
        lba = LBA(ν = CI .* p, A = A, k = k, τ = t0)
        lik = try
            pdf(lba, (choice = chosen_idx[i], rt = rt_sec[i]))
        catch
            eps_val
        end
        nll -= log(max(lik, eps_val))
    end
    return nll
end

# Fit one subject with MLE
function fit_one_subject(data)
    trial_rewards = [cuevalues_to_rewards(v) for v in data.CueValues]
    chosen_idx = Int.(data.Choice)
    rt_sec = Float64.(data.RT_ms) ./ 1000
    cue_type = [count(!=(0), r) == 1 ? :single : :multi for r in trial_rewards]
    r_max = maximum(vcat(trial_rewards...))

    # t0 must stay below every RT used in the likelihood
    t0_upper = minimum(rt_sec) - 0.001
    if t0_upper <= 0.001
        t0_upper = quantile(rt_sec, 0.05) - 0.001
    end
    @assert t0_upper > 0.001 "RTs too small to estimate t0"

    layout = build_param_layout(t0_upper)

    # Keep the starting t0 safely below the smallest RT
    for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
        t0_idx = layout.idx_t0[key]
        layout.x0[t0_idx] = min(layout.x0[t0_idx], 0.5 * t0_upper)
    end

    obj = x -> neg_loglik(x, trial_rewards, chosen_idx, rt_sec, cue_type, r_max, layout)
    fit = optimize(obj, layout.lower, layout.upper, layout.x0, Fminbox(NelderMead()))
    best = Optim.minimizer(fit)
    nll = Optim.minimum(fit)

    return (
        best = best,
        nll = nll,
        layout = layout,
        trial_rewards = trial_rewards,
        chosen_idx = chosen_idx,
        rt_sec = rt_sec,
        cue_type = cue_type,
        r_max = r_max
    )
end

# Observed "accuracy" = chose a max-reward location
function observed_accuracy_by_condition(data, trial_rewards, conditions)
    out = Float64[]
    for cc in conditions
        idx = findall(i -> string(data.CueCondition[i]) == cc, 1:nrow(data))
        if isempty(idx)
            push!(out, NaN)
            continue
        end
        hits = 0.0
        for i in idx
            r = trial_rewards[i]
            best_reward = maximum(r)
            best_locs = findall(==(best_reward), r)
            if data.Choice[i] in best_locs
                hits += 1
            end
        end
        push!(out, hits / length(idx))
    end
    return out
end

# Observed RT quantiles by condition
function observed_rt_quantiles_by_condition(data, rt_sec, conditions, q_list)
    # returns matrix: rows = conditions, cols = quantiles
    mat = fill(NaN, length(conditions), length(q_list))
    for (j, cc) in enumerate(conditions)
        idx = findall(i -> string(data.CueCondition[i]) == cc, 1:nrow(data))
        isempty(idx) && continue
        vals = rt_sec[idx]
        for (k, q) in enumerate(q_list)
            mat[j, k] = quantile(vals, q)
        end
    end
    return mat
end

# Simulate one full predictive dataset from fitted parameters
function simulate_once(trial_rewards, cue_type, r_max, best, layout, rng)
    n = length(trial_rewards)
    sim_choice = Vector{Int}(undef, n)
    sim_rt = Vector{Float64}(undef, n)

    A = best[layout.idx_A]

    for i in 1:n
        cue_key = cue_type[i]
        CI = best[layout.idx_CI[vary_CI_by_cue_number ? cue_key : :all]]
        k = best[layout.idx_k[vary_k_by_cue_number ? cue_key : :all]]
        t0 = best[layout.idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        p = choice_probs(trial_rewards[i], best, layout, r_max)
        lba = LBA(ν = CI .* p, A = A, k = k, τ = t0)
        sample = rand(rng, lba)
        sim_choice[i] = sample.choice
        sim_rt[i] = sample.rt
    end
    return sim_choice, sim_rt
end

# Build prediction mean + 95% interval for accuracy and RT quantiles
function predictive_summaries(data, fit_out, conditions, q_list, n_sims, rng)
    n_cond = length(conditions)
    n_q = length(q_list)

    acc_sims = zeros(n_sims, n_cond)
    rtq_sims = zeros(n_sims, n_cond, n_q)

    for s in 1:n_sims
        sim_choice, sim_rt = simulate_once(
            fit_out.trial_rewards, fit_out.cue_type, fit_out.r_max,
            fit_out.best, fit_out.layout, rng
        )

        for (j, cc) in enumerate(conditions)
            idx = findall(i -> string(data.CueCondition[i]) == cc, 1:nrow(data))
            if isempty(idx)
                acc_sims[s, j] = NaN
                rtq_sims[s, j, :] .= NaN
                continue
            end

            hits = 0.0
            for i in idx
                r = fit_out.trial_rewards[i]
                best_reward = maximum(r)
                best_locs = findall(==(best_reward), r)
                if sim_choice[i] in best_locs
                    hits += 1
                end
            end
            acc_sims[s, j] = hits / length(idx)

            vals = sim_rt[idx]
            for (k, q) in enumerate(q_list)
                rtq_sims[s, j, k] = quantile(vals, q)
            end
        end
    end

    acc_mean = fill(NaN, n_cond)
    acc_lo = fill(NaN, n_cond)
    acc_hi = fill(NaN, n_cond)
    for j in 1:n_cond
        vals = filter(!isnan, acc_sims[:, j])
        isempty(vals) && continue
        acc_mean[j] = mean(vals)
        acc_lo[j] = quantile(vals, 0.025)
        acc_hi[j] = quantile(vals, 0.975)
    end

    rtq_mean = fill(NaN, n_cond, n_q)
    rtq_lo = fill(NaN, n_cond, n_q)
    rtq_hi = fill(NaN, n_cond, n_q)
    for j in 1:n_cond, k in 1:n_q
        vals = filter(!isnan, rtq_sims[:, j, k])
        isempty(vals) && continue
        rtq_mean[j, k] = mean(vals)
        rtq_lo[j, k] = quantile(vals, 0.025)
        rtq_hi[j, k] = quantile(vals, 0.975)
    end

    return (
        acc_mean = acc_mean, acc_lo = acc_lo, acc_hi = acc_hi,
        rtq_mean = rtq_mean, rtq_lo = rtq_lo, rtq_hi = rtq_hi
    )
end

# One accuracy panel: white = data, black = prediction with 95% CI
function make_accuracy_panel(conditions, data_acc, pred_mean, pred_lo, pred_hi, title_str)
    x = collect(1:length(conditions))
    keep = .!isnan.(data_acc)
    xk = x[keep]
    data_k = data_acc[keep]
    pred_k = pred_mean[keep]
    lo_k = pred_lo[keep]
    hi_k = pred_hi[keep]
    labels = conditions[keep]

    p = plot(
        xk, data_k;
        seriestype = :path,
        linestyle = :dash,
        color = :black,
        lw = 1,
        label = false,
        title = title_str,
        ylabel = "Accuracy",
        xlabel = "",
        xticks = (xk, labels),
        xrotation = 30,
        legend = :bottomright,
        size = (500, 350)
    )
    scatter!(
        p, xk, data_k;
        marker = :circle,
        markersize = 6,
        markercolor = :white,
        markerstrokecolor = :black,
        markerstrokewidth = 1.5,
        label = "Data"
    )
    scatter!(
        p, xk, pred_k;
        yerror = (pred_k .- lo_k, hi_k .- pred_k),
        marker = :circle,
        markersize = 5,
        markercolor = :black,
        markerstrokecolor = :black,
        label = "Prediction"
    )
    return p
end

# One RT quantile panel (0.1 / 0.5 / 0.9)
function make_rt_panel(conditions, data_mat, pred_mean, pred_lo, pred_hi, title_str)
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

    for k in 1:size(data_mat, 2)
        data_y = data_mat[:, k]
        pred_y = pred_mean[:, k]
        lo_y = pred_lo[:, k]
        hi_y = pred_hi[:, k]
        keep = .!isnan.(data_y)
        xk = x[keep]
        data_k = data_y[keep]
        pred_k = pred_y[keep]
        lo_k = lo_y[keep]
        hi_k = hi_y[keep]

        plot!(p, xk, data_k; linestyle = :dash, color = :black, lw = 1, label = false)
        scatter!(
            p, xk, data_k;
            marker = :circle, markersize = 5,
            markercolor = :white, markerstrokecolor = :black, markerstrokewidth = 1.5,
            label = false
        )
        scatter!(
            p, xk, pred_k;
            yerror = (pred_k .- lo_k, hi_k .- pred_k),
            marker = :circle, markersize = 4,
            markercolor = :black, markerstrokecolor = :black,
            label = false
        )
    end
    return p
end

# Make the 2x2 figure for one subject and save it
function save_subject_figure(sub_num, data, fit_out, fig_dir)
    # Accuracy / RT for single-cue and dual-cue conditions
    single_present = [cc for cc in single_cue_conditions if cc in Set(string.(data.CueCondition))]
    dual_present = [cc for cc in dual_cue_conditions if cc in Set(string.(data.CueCondition))]

    data_acc_single = observed_accuracy_by_condition(data, fit_out.trial_rewards, single_present)
    data_acc_dual = observed_accuracy_by_condition(data, fit_out.trial_rewards, dual_present)
    data_rt_single = observed_rt_quantiles_by_condition(data, fit_out.rt_sec, single_present, rt_quantiles)
    data_rt_dual = observed_rt_quantiles_by_condition(data, fit_out.rt_sec, dual_present, rt_quantiles)

    pred_single = predictive_summaries(data, fit_out, single_present, rt_quantiles, n_pred_sims, MersenneTwister(RANDOM_SEED + sub_num))
    pred_dual = predictive_summaries(data, fit_out, dual_present, rt_quantiles, n_pred_sims, MersenneTwister(RANDOM_SEED + 1000 + sub_num))

    p_acc_single = make_accuracy_panel(
        single_present, data_acc_single,
        pred_single.acc_mean, pred_single.acc_lo, pred_single.acc_hi,
        "Single-cue accuracy"
    )
    p_acc_dual = make_accuracy_panel(
        dual_present, data_acc_dual,
        pred_dual.acc_mean, pred_dual.acc_lo, pred_dual.acc_hi,
        "Dual-cue accuracy"
    )
    p_rt_single = make_rt_panel(
        single_present, data_rt_single,
        pred_single.rtq_mean, pred_single.rtq_lo, pred_single.rtq_hi,
        "Single-cue RT quantiles (0.1/0.5/0.9)"
    )
    p_rt_dual = make_rt_panel(
        dual_present, data_rt_dual,
        pred_dual.rtq_mean, pred_dual.rtq_lo, pred_dual.rtq_hi,
        "Dual-cue RT quantiles (0.1/0.5/0.9)"
    )

    fig = plot(
        p_acc_single, p_acc_dual, p_rt_single, p_rt_dual;
        layout = (2, 2),
        size = (1100, 800),
        plot_title = "sub$(sub_num)  |  $(omega_mode_tag())  |  white=data, black=prediction (95% CI)"
    )

    isdir(fig_dir) || mkpath(fig_dir)
    # Example names: sub8_mis_lba_expW.png  or  sub8_mis_lba_freeW.png
    fig_path = joinpath(fig_dir, "sub$(sub_num)_mis_lba_$(omega_mode_tag()).png")
    savefig(fig, fig_path)
    println("Saved figure: ", fig_path)
    return fig_path
end

# =============================================================================
# MAIN
# =============================================================================
println("Lab data root: ", lab_data_root)
println("Subject range requested: ", subject_range)
println("free_omega_by_cue_value = ", free_omega_by_cue_value, "  (tag=", omega_mode_tag(), ")")
println("vary_CI_by_cue_number = ", vary_CI_by_cue_number)
println("vary_k_by_cue_number = ", vary_k_by_cue_number)
println("vary_t0_by_cue_number = ", vary_t0_by_cue_number)
println("prob_model = ", prob_model)

subjects = find_subjects_to_fit(lab_data_root, subject_range, MIN_N_SESSIONS)
@assert !isempty(subjects) "No valid subjects found in range $(subject_range)."
println("Subjects to fit: ", subjects)

isdir(fig_dir) || mkpath(fig_dir)

for sub_num in subjects
    println("="^60)
    println("Fitting sub$(sub_num) ...")
    subj_dir = joinpath(lab_data_root, "sub$(sub_num)")
    data = load_subject_data(subj_dir)
    println("Trials used: ", nrow(data))
    @assert nrow(data) > 50 "Too few usable trials for sub$(sub_num)"

    fit_out = fit_one_subject(data)
    best = fit_out.best
    layout = fit_out.layout

    println("NLL: ", round(fit_out.nll, digits=2))
    if free_omega_by_cue_value
        for v in 1:4
            println("omega$(v) = ", round(best[layout.idx_omega[v]], digits=4))
        end
    else
        println("theta = ", round(best[layout.idx_theta], digits=4))
    end
    println("omega0 = ", round(best[layout.idx_omega0], digits=6))
    println("A = ", round(best[layout.idx_A], digits=4))
    for key in (vary_CI_by_cue_number ? [:single, :multi] : [:all])
        println("CI_$(key) = ", round(best[layout.idx_CI[key]], digits=4))
    end
    for key in (vary_k_by_cue_number ? [:single, :multi] : [:all])
        println("k_$(key) = ", round(best[layout.idx_k[key]], digits=4))
    end
    for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
        println("t0_$(key) = ", round(best[layout.idx_t0[key]], digits=4))
    end

    save_subject_figure(sub_num, data, fit_out, fig_dir)
end

println("="^60)
println("Done. Figures are in: ", fig_dir)
