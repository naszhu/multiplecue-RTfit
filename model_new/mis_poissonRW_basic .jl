using CSV, DataFrames, Statistics, Optim, Plots, KernelDensity

# Probability model config:
# :original -> P(i) = exp(theta*r_i)/sum_j exp(theta*r_j)
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
# The MIS-PRW model uses the MIS reward weights as the probabilities of
# successive Poisson evidence samples, then applies the star random walk from
# Blurton et al. (2020) as the response-time decision stage.
prob_model = :original
epsilon = 0.05
@assert prob_model in (:original, :noise_model) "prob_model must be :original or :noise_model."
@assert 0.0 <= epsilon <= 1.0 "epsilon must be in [0, 1]."

# Fixed numerical controls.
max_prw_steps = 120
use_contaminant_floor = false
contaminant_alpha = 0.05
contaminant_rt_max = 3.0

# Data source config (set ONE ID here, then extend the mapping below as needed):
# - CCP001_S1 (lab-processed source)
# - CPP001_S1 / CPP001_S2 / CPP001_S3
# - CPP002_S1 / CPP002_S2 / CPP002_S3
data_source_id = "CPP002_S1"
# data_source_id = "CCP001_S1"
data_source_paths = Dict(
    "CCP001_S1" => joinpath(@__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab", "extracted_data_processed"),
    "CPP001_S1" => joinpath(@__DIR__, "..", "data", "CPP001 - subj 1", "extracted data"),
    "CPP001_S2" => joinpath(@__DIR__, "..", "data", "CPP001 - subj 2", "extracted data"),
    "CPP001_S3" => joinpath(@__DIR__, "..", "data", "CPP001 - subj 3", "extracted data"),
    "CPP002_S1" => joinpath(@__DIR__, "..", "data", "ParticipantCPP002-001", "extracted data"),
    "CPP002_S2" => joinpath(@__DIR__, "..", "data", "ParticipantCPP002-002", "extracted data"),
    "CPP002_S3" => joinpath(@__DIR__, "..", "data", "ParticipantCPP002-003", "extracted data")
)
@assert haskey(data_source_paths, data_source_id) "Unknown data_source_id: $(data_source_id). Add it to data_source_paths."
data_label = data_source_id
data_dir = data_source_paths[data_source_id]
files = filter(f -> endswith(f, ".csv"), readdir(data_dir; join=true))
@assert !isempty(files) "No CSV files found in data_dir: $(data_dir)"
println("data_source_id: ", data_source_id)
println("data_label: ", data_label)
println("data_dir: ", data_dir)
col_types = Dict(
    :Cues => String,
    :CueValues => String,
    :CueRanks => String,
    :RespLoc => String
)
frames = DataFrame[]
for f in files
    hdr = names(CSV.read(f, DataFrame; limit=0))
    has_warmup = "WarmUpTrial" in string.(hdr)
    if has_warmup
        push!(frames, CSV.read(f, DataFrame;
            types=col_types,
            select=[:Session, :WarmUpTrial, :CueValues, :PointTargetResponse, :CueCondition, :RT]
        ))
    else
        println("WARNING: Missing WarmUpTrial in file: ", f, " -> assigning WarmUpTrial=1 for all rows (will be excluded by WarmUpTrial==0 filter).")
        df = CSV.read(f, DataFrame;
            types=col_types,
            select=[:Session, :CueValues, :PointTargetResponse, :CueCondition, :RT]
        )
        df.WarmUpTrial = ones(Int, nrow(df))
        select!(df, [:Session, :WarmUpTrial, :CueValues, :PointTargetResponse, :CueCondition, :RT])
        push!(frames, df)
    end
end
data = reduce(vcat, frames; cols=:setequal)
@assert eltype(data.PointTargetResponse) <: Number "PointTargetResponse must be read as a numeric column."
@assert eltype(data.Session) <: Number "Session must be read as a numeric column."
@assert eltype(data.WarmUpTrial) <: Number "WarmUpTrial must be read as a numeric column."
@assert eltype(data.RT) <: Number "RT must be read as a numeric column."

# Keep only session 6 and later.
data = filter(row -> Int(row.Session) >= 6, data)

# Keep only main trials (WarmUpTrial == 0).
data = filter(row -> Int(row.WarmUpTrial) == 0, data)

# Keep only trials with an actual choice location; 0 means timeout.
data = filter(row -> Int(row.PointTargetResponse) in 1:4, data)
# Keep only positive RT trials. CPP RT is already logged in seconds.
data = filter(row -> Float64(row.RT) > 0, data)
# Require CueValues to be exactly 4 characters in every row.
@assert all(ncodeunits(string(v)) == 4 for v in data.CueValues) "CueValues must be 4-character strings in all rows."

# Convert CPP cue-condition numeric codes (1..10) to CCP-style labels ((1)..(4,3)).
condition_code_to_label = Dict(
    "1" => "(1)", "2" => "(2)", "3" => "(3)", "4" => "(4)",
    "5" => "(2,1)", "6" => "(3,1)", "7" => "(4,1)", "8" => "(3,2)", "9" => "(4,2)", "10" => "(4,3)"
)
if startswith(data_source_id, "CPP")
    data.CueCondition = [get(condition_code_to_label, string(cc), string(cc)) for cc in data.CueCondition]
end

# Precompute arrays for faster MLE optimization.
trial_rewards_arrarr = [parse.(Int, collect(string(v))) for v in data.CueValues]
chosen_idx = Int.(data.PointTargetResponse)
rt_sec = startswith(data_source_id, "CPP") ? Float64.(data.RT) : Float64.(data.RT) ./ 1000
r_max = maximum(vcat(trial_rewards_arrarr...))
eps_val = 1e-12

logfactorials = zeros(Float64, max_prw_steps)
for n in 2:max_prw_steps
    logfactorials[n] = logfactorials[n - 1] + log(n - 1)
end

function reward_probabilities(rewards, theta, omega0, r_max, prob_model, epsilon)
    w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in rewards]
    p = w ./ sum(w)
    if prob_model == :noise_model
        p = (1 - epsilon) .* p .+ epsilon .* (1 / length(rewards))
    end
    return p
end

function build_transition_matrix(step_probs::Vector{Float64}, k::Int)
    n_opts = length(step_probs)
    n_states = 1 + n_opts * (k - 1)
    T = zeros(Float64, n_states, n_states)

    if k > 1
        for i in 1:n_opts
            target_state = 1 + (i - 1) * (k - 1) + 1
            T[1, target_state] = step_probs[i]
        end
    end

    for i in 1:n_opts
        branch_start = 1 + (i - 1) * (k - 1)
        for step in 1:(k - 1)
            row_idx = branch_start + step
            p_forward = step_probs[i]
            p_backward = 1.0 - p_forward
            if step < k - 1
                T[row_idx, row_idx + 1] = p_forward
            end
            if step > 1
                T[row_idx, row_idx - 1] = p_backward
            else
                T[row_idx, 1] = p_backward
            end
        end
    end
    return T
end

function compute_absorption_probs(step_probs::Vector{Float64}, k::Int, max_steps::Int)
    n_opts = length(step_probs)
    n_states = 1 + n_opts * (k - 1)
    current_state_probs = zeros(Float64, n_states)
    current_state_probs[1] = 1.0
    abs_probs = zeros(Float64, max_steps, n_opts)

    if k == 1
        abs_probs[1, :] .= step_probs
        return abs_probs
    end

    T = build_transition_matrix(step_probs, k)
    for n in 1:max_steps
        for i in 1:n_opts
            pre_abs_state = 1 + i * (k - 1)
            abs_probs[n, i] = current_state_probs[pre_abs_state] * step_probs[i]
        end
        current_state_probs = transpose(T) * current_state_probs
    end
    return abs_probs
end

function prw_pdf_point(rt::Float64, t0::Float64, C::Float64, abs_probs::Matrix{Float64}, choice::Int, logfactorials::Vector{Float64})
    decision_time = rt - t0
    if decision_time <= 0
        return 0.0
    end

    density = 0.0
    log_Ct = log(C) + log(decision_time)
    for n in 1:size(abs_probs, 1)
        p_absorb = abs_probs[n, choice]
        if p_absorb > 0
            log_erlang = log(C) + (n - 1) * log_Ct - logfactorials[n] - C * decision_time
            density += p_absorb * exp(log_erlang)
        end
    end
    return density
end

function mixed_absorption(step_probs::Vector{Float64}, k::Float64, max_steps::Int)
    k_floor = max(1, floor(Int, k))
    k_ceil = max(1, ceil(Int, k))
    p_ceil = k - k_floor
    p_floor = 1.0 - p_ceil
    abs_floor = compute_absorption_probs(step_probs, k_floor, max_steps)
    abs_ceil = k_ceil == k_floor ? abs_floor : compute_absorption_probs(step_probs, k_ceil, max_steps)
    return abs_floor, abs_ceil, p_floor, p_ceil
end

function prw_choice_probabilities(step_probs::Vector{Float64}, k::Float64, max_steps::Int)
    abs_floor, abs_ceil, p_floor, p_ceil = mixed_absorption(step_probs, k, max_steps)
    p_choice = p_floor .* vec(sum(abs_floor; dims=1)) .+ p_ceil .* vec(sum(abs_ceil; dims=1))
    total_abs = sum(p_choice)
    return total_abs > 0 ? p_choice ./ total_abs : fill(1 / length(step_probs), length(step_probs))
end

group_map = Dict{Any,Int}()
group_rewards = Vector{Vector{Int}}()
group_trial_indices = Vector{Vector{Int}}()
for i in eachindex(trial_rewards_arrarr)
    r = trial_rewards_arrarr[i]
    key = (r[1], r[2], r[3], r[4])
    if !haskey(group_map, key)
        group_map[key] = length(group_rewards) + 1
        push!(group_rewards, r)
        push!(group_trial_indices, Int[])
    end
    push!(group_trial_indices[group_map[key]], i)
end

t0_upper = minimum(rt_sec) - 0.001
@assert t0_upper > 0.001 "All RTs are too small for estimating positive t0."

param_names = String[]
lower = Float64[]
upper = Float64[]
x0 = Float64[]

push_param!(name, lo, hi, start) = begin
    push!(param_names, name)
    push!(lower, lo)
    push!(upper, hi)
    push!(x0, start)
    length(param_names)
end

idx_theta = push_param!("theta", 0.0, 30.0, 1.0)
idx_omega0 = push_param!("omega0", 1e-6, 50.0, 0.1)
idx_C = push_param!("C", 1.0, 120.0, 20.0)
idx_k = push_param!("k", 1.0, 12.0, 4.0)
idx_t0 = push_param!("t0", 0.001, t0_upper, min(0.25, t0_upper / 2))

# MLE over parameters needed in this experiment:
# theta  = reward-to-weight sensitivity
# omega0 = baseline weight for zero-reward options
# C      = Poisson evidence-sampling rate
# k      = random-walk response threshold; fractional k mixes floor/ceil thresholds
# t0     = non-decision time
obj = x -> begin
    theta = x[idx_theta]
    omega0 = x[idx_omega0]
    C = x[idx_C]
    k = x[idx_k]
    t0 = x[idx_t0]
    C <= 0 && return Inf
    k < 1 && return Inf
    t0 <= 0 && return Inf

    nll = 0.0

    for g in eachindex(group_rewards)
        p_step = reward_probabilities(group_rewards[g], theta, omega0, r_max, prob_model, epsilon)
        abs_floor, abs_ceil, p_floor, p_ceil = mixed_absorption(p_step, k, max_prw_steps)

        for trial_i in group_trial_indices[g]
            rt_sec[trial_i] <= t0 && return Inf
            choice = chosen_idx[trial_i]
            lik = p_floor * prw_pdf_point(rt_sec[trial_i], t0, C, abs_floor, choice, logfactorials) +
                  p_ceil * prw_pdf_point(rt_sec[trial_i], t0, C, abs_ceil, choice, logfactorials)
            if use_contaminant_floor
                lik = (1 - contaminant_alpha) * lik + contaminant_alpha * (1 / (4 * contaminant_rt_max))
            end
            nll -= log(max(lik, eps_val))
        end
    end
    nll
end

fit = optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
best = Optim.minimizer(fit)
theta_hat = best[idx_theta]
omega0_hat = best[idx_omega0]
nll = Optim.minimum(fit)

chosen_prob = Float64[]
for i in eachindex(trial_rewards_arrarr)
    r = trial_rewards_arrarr[i]
    k = best[idx_k]
    p_step = reward_probabilities(r, theta_hat, omega0_hat, r_max, prob_model, epsilon)
    p_choice = prw_choice_probabilities(p_step, k, max_prw_steps)
    push!(chosen_prob, p_choice[chosen_idx[i]])
end

println("Trials used: ", nrow(data))
println("prob_model: ", prob_model)
println("epsilon: ", epsilon)
println("max_prw_steps: ", max_prw_steps)
println("use_contaminant_floor: ", use_contaminant_floor)
println("theta_hat: ", round(theta_hat, digits=6))
println("omega0_hat: ", round(omega0_hat, digits=6))
println("C_hat: ", round(best[idx_C], digits=6))
println("k_hat: ", round(best[idx_k], digits=6))
println("t0_hat: ", round(best[idx_t0], digits=6))
println("Mean predicted p(chosen): ", round(mean(chosen_prob), digits=4))
println("NLL: ", round(nll, digits=2))

plot_cond = String[]
plot_pred = Float64[]
plot_data = Float64[]

# Canonical condition order from metadata.
condition_order = ["(1)", "(2)", "(3)", "(4)", "(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"]
present_conditions = Set(string.(unique(data.CueCondition)))
ordered_conditions = [cc for cc in condition_order if cc in present_conditions]

for cc in ordered_conditions
    cond_rows = filter(row -> string(row.CueCondition) == cc, data)
    cond_chosen_prob = Float64[]
    cond_pred_best = Float64[]
    cond_data_best = Float64[]
    for trial_row in eachrow(cond_rows)
        r = parse.(Int, collect(string(trial_row.CueValues)))
        k = best[idx_k]
        p_step = reward_probabilities(r, theta_hat, omega0_hat, r_max, prob_model, epsilon)
        p = prw_choice_probabilities(p_step, k, max_prw_steps)
        chosen_location = Int(trial_row.PointTargetResponse)
        push!(cond_chosen_prob, p[chosen_location])
        best_reward = maximum(r)
        best_idx = findall(==(best_reward), r)
        push!(cond_pred_best, sum(p[best_idx]))
        push!(cond_data_best, chosen_location in best_idx ? 1.0 : 0.0)
    end
    println(cc, " -> n=", nrow(cond_rows), ", mean p(chosen)=", round(mean(cond_chosen_prob), digits=4))
    push!(plot_cond, cc)
    push!(plot_pred, mean(cond_pred_best))
    push!(plot_data, mean(cond_data_best))
end

x = collect(1:length(plot_cond))
pfig = plot(
    x, plot_pred;
    label="Prediction", marker=:circle, lw=2, color=:blue,
    ylim=(0.8, 1.1)
)
plot!(
    pfig, x, plot_data;
    label="Data", marker=:diamond, lw=2, color=:red
)
xticks!(pfig, (x, plot_cond))
xlabel!(pfig, "Condition")
ylabel!(pfig, "Probability chosen")
title!(pfig, "MIS-PRW Prediction vs Data by CueCondition ($(data_label))")
fig_dir = joinpath(@__DIR__, "figs")
isdir(fig_dir) || mkdir(fig_dir)
model_suffix = prob_model == :noise_model ? "_noise_model_eps$(replace(string(round(epsilon, digits=3)), "." => "p"))" : "_original"
contam_suffix = use_contaminant_floor ? "_contam$(replace(string(round(contaminant_alpha, digits=3)), "." => "p"))" : ""
fig_suffix = "$(model_suffix)$(contam_suffix)"
fig_path = joinpath(fig_dir, "mis_poissonRW_basic_$(lowercase(data_label))_pred_vs_data$(fig_suffix).png")
savefig(pfig, fig_path)
println("Saved plot: ", fig_path)

rt_plots = []
for cc in ordered_conditions
    cond_idx = findall(i -> string(data.CueCondition[i]) == cc, 1:nrow(data))
    cond_rt = rt_sec[cond_idx]
    isempty(cond_rt) && continue

    rt_min = max(best[idx_t0] + 0.001, minimum(cond_rt) - 0.05)
    rt_max = maximum(cond_rt) + 0.05
    rt_grid = collect(range(rt_min, rt_max; length=150))

    data_kde = kde(cond_rt)
    data_density = pdf(data_kde, rt_grid)

    pred_density = zeros(length(rt_grid))
    for trial_i in cond_idx
        r = trial_rewards_arrarr[trial_i]
        C = best[idx_C]
        k = best[idx_k]
        t0 = best[idx_t0]
        p_step = reward_probabilities(r, theta_hat, omega0_hat, r_max, prob_model, epsilon)
        abs_floor, abs_ceil, p_floor, p_ceil = mixed_absorption(p_step, k, max_prw_steps)
        for (j, rt) in enumerate(rt_grid)
            dens = 0.0
            for choice in 1:4
                dens += p_floor * prw_pdf_point(rt, t0, C, abs_floor, choice, logfactorials)
                dens += p_ceil * prw_pdf_point(rt, t0, C, abs_ceil, choice, logfactorials)
            end
            if use_contaminant_floor
                dens = (1 - contaminant_alpha) * dens + contaminant_alpha * (1 / contaminant_rt_max)
            end
            pred_density[j] += dens
        end
    end
    pred_density ./= length(cond_idx)

    push!(rt_plots, plot(
        rt_grid, pred_density;
        label="Prediction", color=:blue, lw=2,
        title=cc, xlabel="RT (s)", ylabel="Density"
    ))
    plot!(rt_plots[end], rt_grid, data_density; label="Data", color=:red, lw=2)
end

rt_fig = plot(rt_plots...; layout=(ceil(Int, length(rt_plots) / 2), 2), size=(1000, 250 * ceil(Int, length(rt_plots) / 2)))
rt_fig_path = joinpath(fig_dir, "mis_poissonRW_basic_$(lowercase(data_label))_rt_distribution$(fig_suffix).png")
savefig(rt_fig, rt_fig_path)
println("Saved RT distribution plot: ", rt_fig_path)
