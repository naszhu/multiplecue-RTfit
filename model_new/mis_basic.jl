using CSV, DataFrames, Statistics, Optim, Plots

# Probability model config:
# :original -> P(i) = exp(theta*r_i)/sum_j exp(theta*r_j)
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
prob_model = :noise_model
epsilon = 0.05
@assert prob_model in (:original, :noise_model) "prob_model must be :original or :noise_model."
@assert 0.0 <= epsilon <= 1.0 "epsilon must be in [0, 1]."

const CCP001_BASE = joinpath(@__DIR__, "..", "data", "CCP001")
const CCP001_PROCESSED_FALLBACK = joinpath(
    @__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab", "extracted_data_processed"
)
const REQUIRED_SESSION = 17
const MIN_SESSION = 6

const ALLOWED_CONDITIONS = Set([
    "(1)", "(2)", "(3)", "(4)",
    "(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"
])

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
    # Prefer RespLoc for CCP trials.csv; fall back to PointTargetResponse for raw extracts.
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
    # Raw session CSVs in CCP001 may lack processed choice labels; use lab-processed copy when present.
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

    col_types = Dict(
        :Cues => String,
        :CueRanks => String
    )

    frames = DataFrame[]
    for f in files
        src = resolve_csv_path(data_dir, f)
        hdr = names(CSV.read(src, DataFrame; limit=0))
        has_warmup = "WarmUpTrial" in string.(hdr)
        if has_warmup
            push!(frames, CSV.read(src, DataFrame;
                types=col_types,
                select=[:Session, :WarmUpTrial, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition]
            ))
        else
            df = CSV.read(src, DataFrame;
                types=col_types,
                select=[:Session, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition]
            )
            df.WarmUpTrial = ones(Int, nrow(df))
            select!(df, [:Session, :WarmUpTrial, :CueValues, :RespLoc, :PointTargetResponse, :CueCondition])
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
        cond = string(row.CueCondition)
        if !(startswith(cond, "(") && cond in ALLOWED_CONDITIONS)
            cond = derive_condition_label(row.CueValues)
            cond === nothing && continue
        end
        push!(rows, (
            Session = Int(row.Session),
            CueValues = row.CueValues,
            Choice = choice,
            CueCondition = cond
        ))
    end
    DataFrame(rows)
end

function prepare_trials(data::DataFrame)
    trial_rewards_arrarr = [cuevalues_to_rewards(v) for v in data.CueValues]
    chosen_idx = data.Choice
    r_max = maximum(vcat(trial_rewards_arrarr...))
    (trial_rewards_arrarr, chosen_idx, r_max)
end

function mis_neg_loglik(
    x::Vector{Float64},
    trial_rewards_arrarr,
    chosen_idx,
    r_max::Int;
    prob_model::Symbol,
    epsilon::Float64
)
    theta = x[1]
    omega0 = x[2]
    eps_val = 1e-12
    nll = 0.0
    for i in eachindex(trial_rewards_arrarr)
        r = trial_rewards_arrarr[i]
        w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in r]
        p = w ./ sum(w)
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
        end
        nll -= log(p[chosen_idx[i]] + eps_val)
    end
    nll
end

function fit_subject(
    trial_rewards_arrarr,
    chosen_idx,
    r_max::Int;
    prob_model::Symbol,
    epsilon::Float64
)
    obj = x -> mis_neg_loglik(x, trial_rewards_arrarr, chosen_idx, r_max; prob_model, epsilon)
    lower = [0.0, 1e-6]
    upper = [30.0, 50.0]
    x0 = [1.0, 0.1]
    fit = optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
    best = Optim.minimizer(fit)
    (theta = best[1], omega0 = best[2], nll = Optim.minimum(fit))
end

function plot_parameter_distribution(results::DataFrame, fig_path::String; prob_model::Symbol, epsilon::Float64)
    subjects = results.subject_id
    x = 1:nrow(results)

    p_theta = bar(
        x, results.theta;
        xticks=(x, subjects), xrotation=45,
        ylabel="theta", legend=false, color=:steelblue
    )
    p_omega = bar(
        x, results.omega0;
        xticks=(x, subjects), xrotation=45,
        ylabel="omega0", legend=false, color=:darkorange
    )

    title_suffix = prob_model == :noise_model ? "noise eps=$(epsilon)" : "original"
    plot(
        p_theta, p_omega;
        layout=(2, 1), size=(700, 520),
        plot_title="CCP001 MIS parameters ($title_suffix)",
        bottom_margin=5Plots.mm
    )
    savefig(fig_path)
end

# --- Main: fit all completed CCP001 subjects ---
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
println("Required session for inclusion: ", REQUIRED_SESSION)
println("Completed subjects: ", [s for (s, _) in completed])
println("Excluded (no ses$(REQUIRED_SESSION)): ", incomplete)
@assert !isempty(completed) "No completed CCP001 subjects found with session $(REQUIRED_SESSION)."

fit_rows = NamedTuple[]
for (subj_id, subj_dir) in completed
    println("\n--- Fitting ", subj_id, " ---")
    println("data_dir: ", subj_dir)
    data = load_ccp001_subject(subj_dir)
    @assert nrow(data) > 0 "No usable trials for $(subj_id) in $(subj_dir)"
    trial_rewards_arrarr, chosen_idx, r_max = prepare_trials(data)
    fit = fit_subject(trial_rewards_arrarr, chosen_idx, r_max; prob_model, epsilon)
    println("Trials used: ", nrow(data))
    println("theta_hat: ", round(fit.theta, digits=6))
    println("omega0_hat: ", round(fit.omega0, digits=6))
    println("NLL: ", round(fit.nll, digits=2))
    push!(fit_rows, (
        subject_id = subj_id,
        n_trials = nrow(data),
        theta = fit.theta,
        omega0 = fit.omega0,
        nll = fit.nll
    ))
end

results = DataFrame(fit_rows)
results_path = joinpath(fig_dir, "ccp001_mis_params.csv")
CSV.write(results_path, results)
println("\nSaved parameter table: ", results_path)

fig_suffix = prob_model == :noise_model ? "_noise_eps$(replace(string(round(epsilon, digits=3)), "." => "p"))" : "_original"
fig_path = joinpath(fig_dir, "ccp001_param_dist$(fig_suffix).png")
plot_parameter_distribution(results, fig_path; prob_model, epsilon)
println("Saved plot: ", fig_path)
