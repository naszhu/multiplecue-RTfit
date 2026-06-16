using CSV, DataFrames, Statistics, Optim, Plots, SequentialSamplingModels

# Multi-subject minimal MIS+LBA for CCP001 (completed subjects with ses17).
# Single-subject / other datasets: use mis_lba_basic.jl

# Probability model config:
# :original -> P(i) = exp(theta*r_i)/sum_j exp(theta*r_j)
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
prob_model = :noise_model
epsilon = 0.05
@assert prob_model in (:original, :noise_model) "prob_model must be :original or :noise_model."
@assert 0.0 <= epsilon <= 1.0 "epsilon must be in [0, 1]."

# Minimal MIS+LBA: shared theta, omega0, CI, A, k, t0 (no single/multi splits).
vary_CI_by_cue_number = false
vary_k_by_cue_number = false
vary_t0_by_cue_number = false

const CCP001_BASE = joinpath(@__DIR__, "..", "data", "CCP001")
const CCP001_PROCESSED_FALLBACK = joinpath(
    @__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab", "extracted_data_processed"
)
const REQUIRED_SESSION = 17
const MIN_SESSION = 6
const MIN_RT_MS = 100.0
const LINE_LW = 2.5
const MARKER_SIZE = 5

const SINGLE_CUE_CONDITIONS = Set(["(1)", "(2)", "(3)", "(4)"])
const DUAL_CUE_CONDITIONS = Set(["(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"])
const ALLOWED_CONDITIONS = union(SINGLE_CUE_CONDITIONS, DUAL_CUE_CONDITIONS)

# :full = all cue conditions; :dual_cue = dual-cue trials only (exclude single-cue).
const DATA_CHUNK = :dual_cue
@assert DATA_CHUNK in (:full, :dual_cue)

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

function has_required_session(data_dir::String, session::Int)::Bool
    any(occursin("ses$(session)", basename(f)) for f in readdir(data_dir))
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

function discover_ccp001_subjects(base_dir::String)
    subjects = Tuple{String, String}[]
    for name in sort(readdir(base_dir))
        m = match(r"^CCP001 - sub (\d+)$", name)
        m === nothing && continue
        subj_id = "S$(m.captures[1])"
        subj_dir = joinpath(base_dir, name)
        isdir(subj_dir) || continue
        push!(subjects, (subj_id, subj_dir))
    end
    subjects
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
            Session = Int(row.Session),
            CueValues = row.CueValues,
            Choice = choice,
            CueCondition = cond,
            RT_ms = rt_ms
        ))
    end
    DataFrame(rows)
end

function filter_by_data_chunk(data::DataFrame, chunk::Symbol)::DataFrame
    if chunk == :full
        return data
    elseif chunk == :dual_cue
        return filter(row -> string(row.CueCondition) in DUAL_CUE_CONDITIONS, data)
    else
        error("Unknown data chunk: $chunk")
    end
end

function results_csv_name(chunk::Symbol)
    chunk == :full ? "ccp001_mis_lba_params_full.csv" : "ccp001_mis_lba_params_dual_cue.csv"
end

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

function build_minimal_lba_indices(t0_upper::Float64)
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

function fit_subject_lba(trial_rewards_arrarr, chosen_idx, rt_sec, r_max::Int; prob_model::Symbol, epsilon::Float64)
    cue_number_type = [count(!=(0), r) == 1 ? :single : :multi for r in trial_rewards_arrarr]
    t0_upper = t0_upper_bound(rt_sec)

    spec = build_minimal_lba_indices(t0_upper)
    obj = x -> mis_lba_neg_loglik(
        x, trial_rewards_arrarr, chosen_idx, rt_sec, r_max, cue_number_type,
        spec.idx_theta, spec.idx_omega0, spec.idx_CI, spec.idx_A, spec.idx_k, spec.idx_t0;
        prob_model, epsilon
    )

    fit = optimize(obj, spec.lower, spec.upper, spec.x0, Fminbox(NelderMead()))
    best = Optim.minimizer(fit)
    (params=params_namedtuple(spec.param_names, best), nll=Optim.minimum(fit))
end

function params_namedtuple(param_names::Vector{String}, best::Vector{Float64})
    NamedTuple{Tuple(Symbol.(param_names))}(Tuple(best))
end

function fit_result_row(subject_id::String, n_trials::Int, param_names::Vector{String}, best::Vector{Float64}, nll::Float64, data_chunk::Symbol)
    fit = params_namedtuple(param_names, best)
    (
        subject_id = subject_id,
        data_chunk = String(data_chunk),
        n_trials = n_trials,
        theta = fit.theta,
        omega0 = fit.omega0,
        CI = fit.CI_all,
        A = fit.A,
        k = fit.k_all,
        t0 = fit.t0_all,
        nll = nll
    )
end

function plot_parameter_lines(results::DataFrame, fig_path::String; prob_model::Symbol, epsilon::Float64)
    subjects = results.subject_id
    x = 1:nrow(results)
    param_cols = [:theta, :omega0, :CI, :A, :k, :t0]
    param_labels = ["theta", "omega0", "CI", "A", "k", "t0"]
    colors = [:steelblue, :darkorange, :seagreen, :purple, :crimson, :goldenrod]

    plots = [
        plot(
            x, results[!, col];
            label=false, color=colors[i], lw=LINE_LW, marker=:circle, markersize=MARKER_SIZE,
            xticks=(x, subjects), xrotation=45,
            ylabel=param_labels[i], legend=false
        )
        for (i, col) in enumerate(param_cols)
    ]

    title_suffix = prob_model == :noise_model ? "noise eps=$(epsilon)" : "original"
    plot(
        plots...;
        layout=(length(param_cols), 1), size=(720, 900),
        plot_title="CCP001 MIS+LBA parameters ($title_suffix)",
        bottom_margin=5Plots.mm
    )
    savefig(fig_path)
end

# --- Main ---
@assert isdir(CCP001_BASE) "CCP001 data folder not found: $(CCP001_BASE)"

fig_dir = joinpath(@__DIR__, "figs", "figs_jun16")
mkpath(fig_dir)

all_subjects = discover_ccp001_subjects(CCP001_BASE)
completed = Tuple{String, String}[]
incomplete = String[]
for (subj_id, subj_dir) in all_subjects
    if has_required_session(subj_dir, REQUIRED_SESSION)
        push!(completed, (subj_id, subj_dir))
    else
        push!(incomplete, subj_id)
    end
end

println("CCP001 base: ", CCP001_BASE)
println("Data chunk: ", DATA_CHUNK)
println("Required session for inclusion: ", REQUIRED_SESSION)
println("Completed subjects: ", [s for (s, _) in completed])
println("Excluded (no ses$(REQUIRED_SESSION)): ", incomplete)
println("prob_model: ", prob_model)
println("epsilon: ", epsilon)
println("vary_CI_by_cue_number: ", vary_CI_by_cue_number)
println("vary_k_by_cue_number: ", vary_k_by_cue_number)
println("vary_t0_by_cue_number: ", vary_t0_by_cue_number)
@assert !isempty(completed) "No completed CCP001 subjects found with session $(REQUIRED_SESSION)."

fit_rows = NamedTuple[]
for (subj_id, subj_dir) in completed
    println("\n--- Fitting ", subj_id, " ---")
    println("data_dir: ", subj_dir)
    data = load_ccp001_subject(subj_dir)
    data = filter_by_data_chunk(data, DATA_CHUNK)
    @assert nrow(data) > 0 "No usable trials for $(subj_id) with data_chunk=$(DATA_CHUNK)"
    trial_rewards_arrarr, chosen_idx, rt_sec, r_max = prepare_lba_trials(data)

    cue_number_type = [count(!=(0), r) == 1 ? :single : :multi for r in trial_rewards_arrarr]
    t0_upper = t0_upper_bound(rt_sec)
    spec = build_minimal_lba_indices(t0_upper)
    obj = x -> mis_lba_neg_loglik(
        x, trial_rewards_arrarr, chosen_idx, rt_sec, r_max, cue_number_type,
        spec.idx_theta, spec.idx_omega0, spec.idx_CI, spec.idx_A, spec.idx_k, spec.idx_t0;
        prob_model, epsilon
    )
    fit = optimize(obj, spec.lower, spec.upper, spec.x0, Fminbox(NelderMead()))
    best = Optim.minimizer(fit)
    nll = Optim.minimum(fit)
    fit_nt = params_namedtuple(spec.param_names, best)

    println("Trials used: ", nrow(data))
    for name in spec.param_names
        println("$(name)_hat: ", round(fit_nt[Symbol(name)], digits=6))
    end
    println("NLL: ", round(nll, digits=2))

    push!(fit_rows, fit_result_row(subj_id, nrow(data), spec.param_names, best, nll, DATA_CHUNK))
end

results = DataFrame(fit_rows)
results_path = joinpath(fig_dir, results_csv_name(DATA_CHUNK))
CSV.write(results_path, results)
println("\nSaved parameter table: ", results_path)

fig_suffix = prob_model == :noise_model ? "_noise_eps$(replace(string(round(epsilon, digits=3)), "." => "p"))" : "_original"
chunk_suffix = DATA_CHUNK == :full ? "_full" : "_dual_cue"
fig_path = joinpath(fig_dir, "ccp001_mis_lba_param_lines$(chunk_suffix)$(fig_suffix).png")
plot_parameter_lines(results, fig_path; prob_model, epsilon)
println("Saved plot: ", fig_path)
println("Combine full + dual_cue with: julia plot_ccp001_mis_lba_params_by_chunk.jl")
