using CSV, DataFrames, Statistics

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
    select=[:CueValues, :PointTargetResponse, :CueCondition]
) for f in files]; cols=:setequal)
@assert eltype(data.PointTargetResponse) <: Number "PointTargetResponse must be read as a numeric column."



# Keep only trials with an actual choice location; 0 means timeout.
data = filter(row -> Int(row.PointTargetResponse) in 1:4, data)
# Require CueValues to be exactly 4 characters in every row.
@assert all(ncodeunits(string(v)) == 4 for v in data.CueValues) "CueValues must be 4-character strings in all rows."

theta = 1.0
eps_val = 1e-12
chosen_prob = Float64[]

for trial_row in eachrow(data)
    cue_values_str = string(trial_row.CueValues)                   # 4 spatial slots
    trial_reward_array = parse.(Int, collect(cue_values_str))     # reward per slot
    trial_weight_array = exp.(theta .* trial_reward_array)         # MIS weighting
    trial_prob_array = trial_weight_array ./ sum(trial_weight_array) # normalized p(choice)
    chosen_location_idx = Int(trial_row.PointTargetResponse)
    push!(chosen_prob, trial_prob_array[chosen_location_idx])
end

nll = -sum(log.(chosen_prob .+ eps_val))

println("Trials used: ", nrow(data))
println("Mean predicted p(chosen): ", round(mean(chosen_prob), digits=4))
println("NLL: ", round(nll, digits=2))

for cc in sort(unique(data.CueCondition))
    cond_rows = filter(row -> row.CueCondition == cc, data)
    cond_chosen_prob = Float64[]
    for trial_row in eachrow(cond_rows)
        cue_values_str = string(trial_row.CueValues)
        trial_reward_array = parse.(Int, collect(cue_values_str))
        trial_prob_array = exp.(theta .* trial_reward_array)
        trial_prob_array ./= sum(trial_prob_array)
        chosen_location_idx = Int(trial_row.PointTargetResponse)
        push!(cond_chosen_prob, trial_prob_array[chosen_location_idx])
    end
    println(cc, " -> n=", nrow(cond_rows), ", mean p(chosen)=", round(mean(cond_chosen_prob), digits=4))
end
