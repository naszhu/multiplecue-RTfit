# Multi-subject MIS + LBA fit for lab data in:
#   multiplecue-responsebox/exp/data_from_lab/subN
#
# Copied from mis_lba_basic.jl and adapted to fit several subjects in one run.
# Run:  julia mis_lba_basic_multisub.jl
# Or:   julia mis_lba_basic_multisub.jl subs=12:14 omega=freeW k=competitor
#
# CLI key=value map (optional; missing keys keep the defaults below):
#   subs=12:20          subject range
#   omega=freeW|expW    free omega1..4  vs  exponential MIS (theta)
#   k=shared|cueN|competitor|delta_r|conflict
#                       shared k | split by cue number | functional k rules
#   CI=shared|cueN      shared CI | split by cue number
#   t0=shared|cueN      shared t0 | split by cue number
#   nsims=100           number of predictive simulations

# Save plots to files only (do not open a display window).
ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Statistics, Optim, Plots, SequentialSamplingModels, Random
default(show = false)

# =============================================================================
# CONFIG (change these) — CLI args override these when provided
# =============================================================================

# Which subjects to fit. Example: 1:10 or 8:11
# Only folders named exactly "subN" are used (ignore sub1star, sub2*, etc.).
# Subjects with fewer than MIN_N_SESSIONS trial files are skipped (likely testing).
subject_range = 12:20
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
free_omega_by_cue_value = true

# LBA cue-number config (all false = one shared CI, k, t0 for all trials)
vary_CI_by_cue_number = false
vary_k_by_cue_number = false
vary_t0_by_cue_number = false

# Functional threshold (k) rule. Only allowed when vary_k_by_cue_number == false.
# :none       -> one shared k (same as before)
# :competitor -> k = k0 + beta_k * C,   C = lower (competitor) reward; 0 if single cue
# :delta_r    -> k = k0 + beta_k * ΔR,  ΔR = H - L; 0 if single cue
# :conflict   -> k = k0 + beta_k * conflict, conflict = 3 - (H - L) on dual cues
#                (closer rewards => higher conflict); 0 if single cue
# Only two threshold params (k0, beta_k) instead of one free k per condition.
k_functional = :competitor

# How many simulated datasets to build prediction means + 95% intervals
n_pred_sims = 100
RANDOM_SEED = 20260801

# Keep sessions from this number onward (same rule as mis_lba_basic.jl)
MIN_SESSION = 6
# Drop unrealistically fast RTs (milliseconds). Same idea as other lab scripts.
MIN_RT_MS = 100.0

# Output folders
fig_dir = joinpath(@__DIR__, "figs", "fit_aug1")
pred_dir = joinpath(@__DIR__, "data_prediction")

# =============================================================================
# Optional CLI overrides (key=value). No args -> keep defaults above.
# =============================================================================
function parse_subject_range(s::AbstractString)
    s = strip(s)
    if occursin(":", s)
        a, b = split(s, ":"; limit=2)
        return parse(Int, a):parse(Int, b)
    else
        n = parse(Int, s)
        return n:n
    end
end

function apply_cli_args!()
    isempty(ARGS) && return
    println("CLI args: ", join(ARGS, " "))
    for arg in ARGS
        occursin("=", arg) || error("Bad CLI arg '$(arg)'. Use key=value.")
        key, val = split(arg, "="; limit=2)
        key = lowercase(strip(key))
        val = lowercase(strip(val))
        if key == "subs"
            global subject_range = parse_subject_range(val)
        elseif key == "omega"
            if val in ("freew", "free")
                global free_omega_by_cue_value = true
            elseif val in ("expw", "exp")
                global free_omega_by_cue_value = false
            else
                error("omega must be freeW or expW, got: $(val)")
            end
        elseif key == "k"
            if val == "shared"
                global vary_k_by_cue_number = false
                global k_functional = :none
            elseif val in ("cuen", "cue_number", "varyk")
                global vary_k_by_cue_number = true
                global k_functional = :none
            elseif val in ("competitor", "comp", "kcomp")
                global vary_k_by_cue_number = false
                global k_functional = :competitor
            elseif val in ("delta_r", "deltar", "kdelta")
                global vary_k_by_cue_number = false
                global k_functional = :delta_r
            elseif val in ("conflict", "kconflict")
                global vary_k_by_cue_number = false
                global k_functional = :conflict
            else
                error("k must be shared|cueN|competitor|delta_r|conflict, got: $(val)")
            end
        elseif key == "ci"
            if val == "shared"
                global vary_CI_by_cue_number = false
            elseif val in ("cuen", "cue_number")
                global vary_CI_by_cue_number = true
            else
                error("CI must be shared or cueN, got: $(val)")
            end
        elseif key == "t0"
            if val == "shared"
                global vary_t0_by_cue_number = false
            elseif val in ("cuen", "cue_number")
                global vary_t0_by_cue_number = true
            else
                error("t0 must be shared or cueN, got: $(val)")
            end
        elseif key == "nsims"
            global n_pred_sims = parse(Int, val)
        else
            error("Unknown CLI key '$(key)'. See map at top of file.")
        end
    end
end

apply_cli_args!()

# =============================================================================
# Checks
# =============================================================================
@assert prob_model in (:original, :noise_model)
@assert 0.0 <= epsilon <= 1.0
@assert n_pred_sims >= 20
@assert k_functional in (:none, :competitor, :delta_r, :conflict) "k_functional must be :none, :competitor, :delta_r, or :conflict."
@assert !(vary_k_by_cue_number && k_functional != :none) "Cannot use k_functional with vary_k_by_cue_number. Set one or the other."

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

# Short tag used in saved figure names, e.g. freeW, freeW_varyK, freeW_kDeltaR
function fig_mode_tag()
    tag = free_omega_by_cue_value ? "freeW" : "expW"
    extras = String[]
    vary_CI_by_cue_number && push!(extras, "varyCI")
    if vary_k_by_cue_number
        push!(extras, "varyK")
    elseif k_functional == :competitor
        push!(extras, "kComp")
    elseif k_functional == :delta_r
        push!(extras, "kDeltaR")
    elseif k_functional == :conflict
        push!(extras, "kConflict")
    end
    vary_t0_by_cue_number && push!(extras, "varyT0")
    if !isempty(extras)
        tag = tag * "_" * join(extras, "_")
    end
    return tag
end

# Feature used by k_functional (0 on single-cue trials)
function k_feature(rewards)
    nonzero = filter(>(0), rewards)
    length(nonzero) < 2 && return 0.0
    H = maximum(nonzero)
    L = minimum(nonzero)
    if k_functional == :competitor
        return Float64(L)
    elseif k_functional == :delta_r
        return Float64(H - L)
    elseif k_functional == :conflict
        # Closer competitors => higher conflict. Max gap in this design is 3.
        return Float64(3 - (H - L))
    else
        return 0.0
    end
end

# Resolve trial threshold gap k from the current config
function trial_k(x, layout, rewards, cue_key)
    if vary_k_by_cue_number
        # Original split: one k for single-cue, one k for dual-cue
        return x[layout.idx_k[cue_key]]
    elseif k_functional != :none
        # Functional rule: k = k0 + beta_k * feature
        return x[layout.idx_k0] + x[layout.idx_beta_k] * k_feature(rewards)
    else
        # Shared k for all trials
        return x[layout.idx_k[:all]]
    end
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

    # Threshold (k) parameters
    idx_k = Dict{Symbol, Int}()
    idx_k0 = 0
    idx_beta_k = 0
    if vary_k_by_cue_number
        # Separate k for one-cue vs two-cue trials
        for key in [:single, :multi]
            push!(names, "k_$(key)")
            push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
            idx_k[key] = length(names)
        end
    elseif k_functional != :none
        # Functional: k = k0 + beta_k * feature
        push!(names, "k0")
        push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
        idx_k0 = length(names)
        push!(names, "beta_k")
        # beta can be negative (stronger competitor / more conflict => lower threshold)
        push!(lower, -5.0); push!(upper, 5.0); push!(x0, 0.0)
        idx_beta_k = length(names)
    else
        # One shared k
        push!(names, "k_all")
        push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
        idx_k[:all] = length(names)
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
        idx_k0 = idx_k0,
        idx_beta_k = idx_beta_k,
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
        k = trial_k(x, layout, trial_rewards[i], cue_key)
        t0 = x[layout.idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        (k <= 0 || rt_sec[i] <= t0) && return Inf

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
        k = trial_k(best, layout, trial_rewards[i], cue_key)
        t0 = best[layout.idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        # Keep k positive for sampling (same guard as in the likelihood)
        k = max(k, 1e-6)
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

# Build one long table of observed + predicted summaries for CSV / compare plots
function build_prediction_table(sub_num, data, fit_out)
    mode = fig_mode_tag()
    rows = NamedTuple[]

    for (group_name, cond_list) in (
        ("single", single_cue_conditions),
        ("dual", dual_cue_conditions),
    )
        conditions = [cc for cc in cond_list if cc in Set(string.(data.CueCondition))]
        isempty(conditions) && continue

        data_acc = observed_accuracy_by_condition(data, fit_out.trial_rewards, conditions)
        data_rt = observed_rt_quantiles_by_condition(data, fit_out.rt_sec, conditions, rt_quantiles)
        seed = group_name == "single" ? RANDOM_SEED + sub_num : RANDOM_SEED + 1000 + sub_num
        pred = predictive_summaries(data, fit_out, conditions, rt_quantiles, n_pred_sims, MersenneTwister(seed))

        for (j, cc) in enumerate(conditions)
            push!(rows, (
                subject = sub_num,
                mode_tag = mode,
                cue_group = group_name,
                condition = cc,
                data_acc = data_acc[j],
                pred_acc = pred.acc_mean[j],
                pred_acc_lo = pred.acc_lo[j],
                pred_acc_hi = pred.acc_hi[j],
                data_rt10 = data_rt[j, 1],
                data_rt50 = data_rt[j, 2],
                data_rt90 = data_rt[j, 3],
                pred_rt10 = pred.rtq_mean[j, 1],
                pred_rt10_lo = pred.rtq_lo[j, 1],
                pred_rt10_hi = pred.rtq_hi[j, 1],
                pred_rt50 = pred.rtq_mean[j, 2],
                pred_rt50_lo = pred.rtq_lo[j, 2],
                pred_rt50_hi = pred.rtq_hi[j, 2],
                pred_rt90 = pred.rtq_mean[j, 3],
                pred_rt90_lo = pred.rtq_lo[j, 3],
                pred_rt90_hi = pred.rtq_hi[j, 3],
                nll = fit_out.nll,
            ))
        end
    end
    return DataFrame(rows)
end

# Make the 2x2 figure for one subject and save prediction CSV
function save_subject_figure(sub_num, data, fit_out, fig_dir, pred_dir)
    pred_table = build_prediction_table(sub_num, data, fit_out)
    mode = fig_mode_tag()

    isdir(pred_dir) || mkpath(pred_dir)
    pred_path = joinpath(pred_dir, "sub$(sub_num)_mis_lba_$(mode)_pred.csv")
    CSV.write(pred_path, pred_table)
    println("Saved prediction CSV: ", pred_path)

    function rows_for(group)
        return filter(r -> r.cue_group == group, pred_table)
    end

    function panel_from_rows(df_g, title_acc, title_rt)
        conditions = String.(df_g.condition)
        data_acc = Float64.(df_g.data_acc)
        pred_acc = Float64.(df_g.pred_acc)
        pred_acc_lo = Float64.(df_g.pred_acc_lo)
        pred_acc_hi = Float64.(df_g.pred_acc_hi)

        data_rt = hcat(Float64.(df_g.data_rt10), Float64.(df_g.data_rt50), Float64.(df_g.data_rt90))
        pred_rt = hcat(Float64.(df_g.pred_rt10), Float64.(df_g.pred_rt50), Float64.(df_g.pred_rt90))
        pred_rt_lo = hcat(Float64.(df_g.pred_rt10_lo), Float64.(df_g.pred_rt50_lo), Float64.(df_g.pred_rt90_lo))
        pred_rt_hi = hcat(Float64.(df_g.pred_rt10_hi), Float64.(df_g.pred_rt50_hi), Float64.(df_g.pred_rt90_hi))

        p_acc = make_accuracy_panel(conditions, data_acc, pred_acc, pred_acc_lo, pred_acc_hi, title_acc)
        p_rt = make_rt_panel(conditions, data_rt, pred_rt, pred_rt_lo, pred_rt_hi, title_rt)
        return p_acc, p_rt
    end

    df_single = rows_for("single")
    df_dual = rows_for("dual")
    p_acc_single, p_rt_single = panel_from_rows(
        df_single, "Single-cue accuracy", "Single-cue RT quantiles (0.1/0.5/0.9)"
    )
    p_acc_dual, p_rt_dual = panel_from_rows(
        df_dual, "Dual-cue accuracy", "Dual-cue RT quantiles (0.1/0.5/0.9)"
    )

    fig = plot(
        p_acc_single, p_acc_dual, p_rt_single, p_rt_dual;
        layout = (2, 2),
        size = (1100, 800),
        plot_title = "sub$(sub_num)  |  $(mode)  |  white=data, black=prediction (95% CI)"
    )

    isdir(fig_dir) || mkpath(fig_dir)
    fig_path = joinpath(fig_dir, "sub$(sub_num)_mis_lba_$(mode).png")
    savefig(fig, fig_path)
    println("Saved figure: ", fig_path)
    return fig_path, pred_path
end

# =============================================================================
# MAIN
# =============================================================================
println("Lab data root: ", lab_data_root)
println("Subject range requested: ", subject_range)
println("free_omega_by_cue_value = ", free_omega_by_cue_value, "  (tag=", fig_mode_tag(), ")")
println("vary_CI_by_cue_number = ", vary_CI_by_cue_number)
println("vary_k_by_cue_number = ", vary_k_by_cue_number)
println("k_functional = ", k_functional)
println("vary_t0_by_cue_number = ", vary_t0_by_cue_number)
println("prob_model = ", prob_model)

subjects = find_subjects_to_fit(lab_data_root, subject_range, MIN_N_SESSIONS)
@assert !isempty(subjects) "No valid subjects found in range $(subject_range)."
println("Subjects to fit: ", subjects)

isdir(fig_dir) || mkpath(fig_dir)
isdir(pred_dir) || mkpath(pred_dir)

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
    if vary_k_by_cue_number
        for key in [:single, :multi]
            println("k_$(key) = ", round(best[layout.idx_k[key]], digits=4))
        end
    elseif k_functional != :none
        println("k0 = ", round(best[layout.idx_k0], digits=4))
        println("beta_k = ", round(best[layout.idx_beta_k], digits=4))
    else
        println("k_all = ", round(best[layout.idx_k[:all]], digits=4))
    end
    for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
        println("t0_$(key) = ", round(best[layout.idx_t0[key]], digits=4))
    end

    save_subject_figure(sub_num, data, fit_out, fig_dir, pred_dir)
end

println("="^60)
println("Done. Figures are in: ", fig_dir)
println("Prediction CSVs are in: ", pred_dir)
