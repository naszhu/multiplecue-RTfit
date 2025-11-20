# ==========================================================================
# Diagnostic Script: Investigate Excluded Trials
# ==========================================================================
# This script investigates why trials are being excluded during data loading
# ==========================================================================

using Pkg
include("data_utils.jl")
using .DataUtils

using CSV
using DataFrames
using Glob
using Statistics

# Helper function for counting
function countmap(v)
    d = Dict{eltype(v), Int}()
    for x in v
        d[x] = get(d, x, 0) + 1
    end
    return d
end

const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"

println("=" ^ 70)
println("DIAGNOSTIC: INVESTIGATING EXCLUDED TRIALS")
println("=" ^ 70)

# Load data files
files = glob(FILE_PATTERN, DATA_PATH)
non_practice_files = filter(f -> !occursin("-Prac-", f), files)

println("\nFound $(length(files)) files total, $(length(non_practice_files)) non-practice files.")

df_list = DataFrame[]
for (i, file) in enumerate(non_practice_files)
    dt = DataUtils.read_psychopy_dat(file)
    if !isempty(dt)
        # Keep ALL columns for investigation
        push!(df_list, dt)
    end
end

if isempty(df_list)
    error("No valid data could be read from files.")
end

full_df = vcat(df_list...)
initial_rows = nrow(full_df)
println("\nTotal rows after merging: $initial_rows")

# Check what columns we have
println("\n" * "=" ^ 70)
println("AVAILABLE COLUMNS IN DATA")
println("=" ^ 70)
println("Columns: ", names(full_df))
println("\nColumn types:")
for col in names(full_df)
    println("  $col: $(eltype(full_df[!, col]))")
end

# Check for required columns
required_cols = ["RT", "CueValues", "RespLoc", "PointTargetResponse", "CueResponseValue", "CueCondition"]
println("\n" * "=" ^ 70)
println("REQUIRED COLUMNS CHECK")
println("=" ^ 70)
for col in required_cols
    present = col in names(full_df)
    println("  $col: $(present ? "✓ PRESENT" : "✗ MISSING")")
    if present
        missing_count = sum(ismissing.(full_df[!, col]))
        println("      Missing values: $missing_count / $initial_rows ($(round(100*missing_count/initial_rows, digits=1))%)")
    end
end

# Parse CueValues
full_df.ParsedRewards = DataUtils.parse_array_string.(full_df.CueValues)

# Check CueValues parsing
println("\n" * "=" ^ 70)
println("CUEVALUES PARSING CHECK")
println("=" ^ 70)
empty_rewards = sum(length.(full_df.ParsedRewards) .== 0)
println("Trials with empty ParsedRewards: $empty_rewards / $initial_rows")
if empty_rewards > 0
    println("\nSample rows with empty ParsedRewards:")
    empty_idx = findall(length.(full_df.ParsedRewards) .== 0)
    for i in empty_idx[1:min(5, length(empty_idx))]
        println("  Row $i: CueValues = '$(full_df.CueValues[i])'")
    end
end

# Clean RT
full_df.CleanRT = DataUtils.parse_clean_float.(full_df.RT)
missing_rt = sum(ismissing.(full_df.CleanRT))
println("\n" * "=" ^ 70)
println("RT FILTERING CHECK")
println("=" ^ 70)
println("Missing RT: $missing_rt / $initial_rows")
valid_rt = .!ismissing.(full_df.CleanRT)
rt_in_range = valid_rt .& (full_df.CleanRT .> 0.05) .& (full_df.CleanRT .< 3.0)
rt_filtered = sum(.!rt_in_range)
println("RT out of range (<=0.05 or >=3.0): $rt_filtered / $initial_rows")

if rt_filtered > 0
    println("\nRT distribution (before filtering):")
    valid_rt_vals = filter(x -> !ismissing(x), full_df.CleanRT)
    if !isempty(valid_rt_vals)
        println("  Min: $(minimum(valid_rt_vals))")
        println("  Max: $(maximum(valid_rt_vals))")
        println("  Mean: $(mean(valid_rt_vals))")
        println("  Median: $(median(valid_rt_vals))")
    end
    println("\nRT values outside range:")
    out_of_range = full_df[.!rt_in_range, :]
    println("  Rows with RT <= 0.05: $(sum(full_df.CleanRT .<= 0.05))")
    println("  Rows with RT >= 3.0: $(sum(full_df.CleanRT .>= 3.0))")
end

# Apply RT filter for next steps
filter!(row -> !ismissing(row.CleanRT) && 0.05 < row.CleanRT < 3.0, full_df)
after_rt_filter = nrow(full_df)
println("\nAfter RT filtering: $after_rt_filter (removed $(initial_rows - after_rt_filter))")

# Determine Choice (same logic as data_utils.jl)
println("\n" * "=" ^ 70)
println("CHOICE DETERMINATION ANALYSIS")
println("=" ^ 70)

choices = Int[]
choice_strategy_used = String[]

for row in eachrow(full_df)
    c = 0
    strategy = "none"

    # Strategy A: Check PointTargetResponse
    if "PointTargetResponse" in names(full_df)
        val = DataUtils.parse_clean_float(row.PointTargetResponse)
        if !ismissing(val)
            c = Int(val)
            strategy = "PointTargetResponse"
        end
    end

    # Strategy B: Check RespLoc if PointTargetResponse failed
    if c == 0 && "RespLoc" in names(full_df)
        val = DataUtils.parse_clean_float(row.RespLoc)
        if !ismissing(val)
            c = Int(val)
            strategy = "RespLoc"
        end
    end

    # Strategy C: Infer from Reward if both failed
    if c == 0 && "CueResponseValue" in names(full_df)
        val = DataUtils.parse_clean_float(row.CueResponseValue)
        if !ismissing(val)
            idx = findfirst(x -> x == val, row.ParsedRewards)
            if !isnothing(idx)
                c = idx
                strategy = "CueResponseValue"
            end
        end
    end

    push!(choices, c)
    push!(choice_strategy_used, strategy)
end

full_df.Choice = choices
full_df.ChoiceStrategy = choice_strategy_used

# Analyze choice determination
println("\nChoice determination results:")
choice_counts = countmap(choice_strategy_used)
for (strategy, count) in sort(collect(choice_counts), by=x->x[2], rev=true)
    pct = round(100*count/after_rt_filter, digits=1)
    println("  $strategy: $count ($pct%)")
end

invalid_choices = (full_df.Choice .<= 0) .| (full_df.Choice .> length.(full_df.ParsedRewards))
invalid_count = sum(invalid_choices)
println("\nInvalid choices: $invalid_count / $after_rt_filter")

if invalid_count > 0
    println("\n" * "=" ^ 70)
    println("DETAILED ANALYSIS OF INVALID CHOICES")
    println("=" ^ 70)
    
    invalid_df = full_df[invalid_choices, :]
    
    println("\nBreakdown by reason:")
    zero_choice = sum(full_df.Choice .<= 0)
    out_of_bounds = sum((full_df.Choice .> 0) .& (full_df.Choice .> length.(full_df.ParsedRewards)))
    println("  Choice <= 0: $zero_choice")
    println("  Choice > length(ParsedRewards): $out_of_bounds")
    
    println("\nBreakdown by strategy used:")
    for strategy in unique(invalid_df.ChoiceStrategy)
        count = sum(invalid_df.ChoiceStrategy .== strategy)
        println("  $strategy: $count")
    end
    
    println("\nSample invalid choice rows (first 10):")
    for i in 1:min(10, nrow(invalid_df))
        row = invalid_df[i, :]
        println("\n  Row $i:")
        println("    Choice: $(row.Choice)")
        println("    Strategy: $(row.ChoiceStrategy)")
        println("    ParsedRewards length: $(length(row.ParsedRewards))")
        println("    ParsedRewards: $(row.ParsedRewards)")
        if "PointTargetResponse" in names(full_df)
            println("    PointTargetResponse: $(row.PointTargetResponse)")
        end
        if "RespLoc" in names(full_df)
            println("    RespLoc: $(row.RespLoc)")
        end
        if "CueResponseValue" in names(full_df)
            println("    CueResponseValue: $(row.CueResponseValue)")
        end
        println("    CueValues: $(row.CueValues)")
        println("    RT: $(row.CleanRT)")
    end
    
    # Check if there are patterns in missing data
    println("\n" * "=" ^ 70)
    println("MISSING DATA PATTERNS IN INVALID CHOICES")
    println("=" ^ 70)
    
    if "PointTargetResponse" in names(full_df)
        pt_missing = sum(ismissing.(invalid_df.PointTargetResponse))
        println("PointTargetResponse missing: $pt_missing / $invalid_count")
    end
    if "RespLoc" in names(full_df)
        rl_missing = sum(ismissing.(invalid_df.RespLoc))
        println("RespLoc missing: $rl_missing / $invalid_count")
    end
    if "CueResponseValue" in names(full_df)
        crv_missing = sum(ismissing.(invalid_df.CueResponseValue))
        println("CueResponseValue missing: $crv_missing / $invalid_count")
    end
end

# Final summary
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println("Initial rows: $initial_rows")
println("After RT filtering: $after_rt_filter (removed $(initial_rows - after_rt_filter))")
valid_after_choice = sum((full_df.Choice .> 0) .& (full_df.Choice .<= length.(full_df.ParsedRewards)))
println("Valid after choice filtering: $valid_after_choice (removed $(after_rt_filter - valid_after_choice))")
println("\nTotal excluded: $(initial_rows - valid_after_choice) / $initial_rows ($(round(100*(initial_rows - valid_after_choice)/initial_rows, digits=1))%)")

