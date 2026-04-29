using CSV, DataFrames, Statistics, Optim, Plots, SequentialSamplingModels, KernelDensity

# Probability model config:
# :original -> P(i) = exp(theta*r_i)/sum_j exp(theta*r_j)
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
prob_model = :noise_model
epsilon = 0.05
@assert prob_model in (:original, :noise_model) "prob_model must be :original or :noise_model."
@assert 0.0 <= epsilon <= 1.0 "epsilon must be in [0, 1]."

# Basic fixed LBA settings. CI is fitted and scales the MIS relative weights into drift rates.
lba_A = 0.3
lba_k = 0.3
lba_t0 = 0.05

# Data source config (set ONE ID here, then extend the mapping below as needed):
# - CCP001_S1 (lab-processed source)
# - CPP001_S1 / CPP001_S2 / CPP001_S3
# - CPP002_S1 / CPP002_S2 / CPP002_S3
data_source_id = "CCP001_S1"
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
# Keep only trials where RT is usable for the LBA likelihood. RT is logged in ms.
data = filter(row -> Float64(row.RT) / 1000 > lba_t0, data)
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
rt_sec = Float64.(data.RT) ./ 1000
r_max = maximum(vcat(trial_rewards_arrarr...))
eps_val = 1e-12

# MLE over parameters needed in this experiment:
# theta  = reward-to-weight sensitivity
# omega0 = baseline weight for zero-reward options
# CI     = LBA processing capacity, scales relative weights into drift rates
obj = x -> begin
    theta = x[1]
    omega0 = x[2]
    CI = x[3]
    nll = 0.0
    for i in eachindex(trial_rewards_arrarr)
        r = trial_rewards_arrarr[i]
        w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in r]
        p = w ./ sum(w) # softmax-like probability
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
        end
        drift_rates = CI .* p
        lba = LBA(ν=drift_rates, A=lba_A, k=lba_k, τ=lba_t0)
        lik = try
            pdf(lba, (choice=chosen_idx[i], rt=rt_sec[i]))
        catch
            eps_val
        end
        nll -= log(max(lik, eps_val))
    end
    nll
end

# lower/upper: [theta, omega0, CI]
lower = [0.0, 1e-6, 0.1]
# Cap theta to reduce near-deterministic softmax collapse in noise-model fits.
upper = [30.0, 50.0, 30.0]
x0 = [1.0, 0.1, 5.0]
fit = optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
best = Optim.minimizer(fit)
theta_hat = best[1]
omega0_hat = best[2]
CI_hat = best[3]
nll = Optim.minimum(fit)

chosen_prob = Float64[]
for i in eachindex(trial_rewards_arrarr)
    r = trial_rewards_arrarr[i]
    w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
    p = w ./ sum(w)
    if prob_model == :noise_model
        p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
    end
    push!(chosen_prob, p[chosen_idx[i]])
end

println("Trials used: ", nrow(data))
println("prob_model: ", prob_model)
println("epsilon: ", epsilon)
println("lba_A fixed: ", lba_A)
println("lba_k fixed: ", lba_k)
println("lba_t0 fixed: ", lba_t0)
println("theta_hat: ", round(theta_hat, digits=6))
println("omega0_hat: ", round(omega0_hat, digits=6))
println("CI_hat: ", round(CI_hat, digits=6))
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
        w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
        p = w ./ sum(w)
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
        end
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
title!(pfig, "MIS-LBA Prediction vs Data by CueCondition ($(data_label))")
fig_dir = joinpath(@__DIR__, "figs")
isdir(fig_dir) || mkdir(fig_dir)
fig_suffix = prob_model == :noise_model ? "_noise_model_eps$(replace(string(round(epsilon, digits=3)), "." => "p"))" : "_original"
fig_path = joinpath(fig_dir, "mis_lba_basic_$(lowercase(data_label))_pred_vs_data$(fig_suffix).png")
savefig(pfig, fig_path)
println("Saved plot: ", fig_path)

rt_plots = []
for cc in ordered_conditions
    cond_idx = findall(i -> string(data.CueCondition[i]) == cc, 1:nrow(data))
    cond_rt = rt_sec[cond_idx]
    isempty(cond_rt) && continue

    rt_min = max(lba_t0 + 0.001, minimum(cond_rt) - 0.05)
    rt_max = maximum(cond_rt) + 0.05
    rt_grid = collect(range(rt_min, rt_max; length=150))

    data_kde = kde(cond_rt)
    data_density = pdf(data_kde, rt_grid)

    pred_density = zeros(length(rt_grid))
    for trial_i in cond_idx
        r = trial_rewards_arrarr[trial_i]
        w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
        p_choice = w ./ sum(w)
        if prob_model == :noise_model
            p_choice = (1 - epsilon) .* p_choice .+ epsilon .* (1 / 4)
        end
        lba = LBA(ν=CI_hat .* p_choice, A=lba_A, k=lba_k, τ=lba_t0)
        for (j, rt) in enumerate(rt_grid)
            pred_density[j] += sum(pdf(lba, (choice=choice, rt=rt)) for choice in 1:4)
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
rt_fig_path = joinpath(fig_dir, "mis_lba_basic_$(lowercase(data_label))_rt_distribution$(fig_suffix).png")
savefig(rt_fig, rt_fig_path)
println("Saved RT distribution plot: ", rt_fig_path)
