using CSV, DataFrames, Statistics, Optim, Plots

data_dir = joinpath(@__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab", "extracted_data_processed")
files = filter(f -> endswith(f, ".csv"), readdir(data_dir; join=true))
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
        p = w ./ sum(w)
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
    push!(chosen_prob, p[chosen_idx[i]])
end

println("Trials used: ", nrow(data))
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
    cond_rows = filter(row -> row.CueCondition == cc, data)
    cond_chosen_prob = Float64[]
    cond_pred_best = Float64[]
    cond_data_best = Float64[]
    for trial_row in eachrow(cond_rows)
        r = parse.(Int, collect(string(trial_row.CueValues)))
        w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
        p = w ./ sum(w)
        chosen_location = Int(trial_row.PointTargetResponse)
        push!(cond_chosen_prob, p[chosen_location])
        best_reward = maximum(r)
        best_idx = findall(==(best_reward), r)
        push!(cond_pred_best, sum(p[best_idx]))
        push!(cond_data_best, chosen_location in best_idx ? 1.0 : 0.0)
    end
    println(cc, " -> n=", nrow(cond_rows), ", mean p(chosen)=", round(mean(cond_chosen_prob), digits=4))
    push!(plot_cond, string(cc))
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
title!(pfig, "Prediction vs Data by CueCondition")
fig_dir = joinpath(@__DIR__, "figs")
isdir(fig_dir) || mkdir(fig_dir)
fig_path = joinpath(fig_dir, "mis_basic_pred_vs_data.png")
savefig(pfig, fig_path)
println("Saved plot: ", fig_path)
