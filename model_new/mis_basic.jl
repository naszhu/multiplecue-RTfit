using CSV, DataFrames, Statistics, Optim, Plots

# Probability model config:
# :original -> P(i) = exp(theta*r_i)/sum_j exp(theta*r_j)
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
prob_model = :noise_model
epsilon = 0.05
@assert prob_model in (:original, :noise_model) "prob_model must be :original or :noise_model."
@assert 0.0 <= epsilon <= 1.0 "epsilon must be in [0, 1]."

# Data source config (set ONE ID here, then extend the mapping below as needed):
# - CCP001_S1 (lab-processed source)
# - CPP001_S1 / CPP001_S2 / CPP001_S3
# - CPP002_S1 / CPP002_S2 / CPP002_S3
data_source_id = "CPP001_S1"
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
data = reduce(vcat, [CSV.read(f, DataFrame;
    types=col_types,
    select=[:Session, :CueValues, :PointTargetResponse, :CueCondition]
) for f in files]; cols=:setequal)
@assert eltype(data.PointTargetResponse) <: Number "PointTargetResponse must be read as a numeric column."
@assert eltype(data.Session) <: Number "Session must be read as a numeric column."

# Keep only session 6 and later.
data = filter(row -> Int(row.Session) >= 6, data)



# Keep only trials with an actual choice location; 0 means timeout.
data = filter(row -> Int(row.PointTargetResponse) in 1:4, data)
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
r_max = maximum(vcat(trial_rewards_arrarr...))
eps_val = 1e-12

# MLE over parameters needed in this experiment:
# theta  = reward-to-weight sensitivity
# omega0 = baseline weight for zero-reward options
obj = x -> begin
    theta = x[1]
    omega0 = x[2]
    nll = 0.0
    for i in eachindex(trial_rewards_arrarr)
        r = trial_rewards_arrarr[i]
        w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in r]
        p = w ./ sum(w) # softmax-like probability
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
        end
        nll -= log(p[chosen_idx[i]] + eps_val)
    end
    nll
end

lower = [0.0, 1e-6]
upper = [50.0, 10.0]
x0 = [1.0, 0.1]
fit = optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
best = Optim.minimizer(fit)
theta_hat = best[1]
omega0_hat = best[2]
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
println("theta_hat: ", round(theta_hat, digits=6))
println("omega0_hat: ", round(omega0_hat, digits=6))
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
    ylim=(0.5, 1.1)
)
plot!(
    pfig, x, plot_data;
    label="Data", marker=:diamond, lw=2, color=:red
)
xticks!(pfig, (x, plot_cond))
xlabel!(pfig, "Condition")
ylabel!(pfig, "Probability chosen")
title!(pfig, "Prediction vs Data by CueCondition ($(data_label))")
fig_dir = joinpath(@__DIR__, "figs")
isdir(fig_dir) || mkdir(fig_dir)
fig_suffix = prob_model == :noise_model ? "_noise_model_eps$(replace(string(round(epsilon, digits=3)), "." => "p"))" : "_original"
fig_path = joinpath(fig_dir, "mis_basic_$(lowercase(data_label))_pred_vs_data$(fig_suffix).png")
savefig(pfig, fig_path)
println("Saved plot: ", fig_path)
