# ==========================================================================
# Investigate Choice Encoding
# ==========================================================================
# This script investigates the relationship between PointTargetResponse,
# CueResponseValue, and ParsedRewards to understand the encoding
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
println("INVESTIGATING CHOICE ENCODING")
println("=" ^ 70)

# Load data
files = glob(FILE_PATTERN, DATA_PATH)
non_practice_files = filter(f -> !occursin("-Prac-", f), files)

df_list = DataFrame[]
for file in non_practice_files
    dt = DataUtils.read_psychopy_dat(file)
    if !isempty(dt)
        push!(df_list, dt)
    end
end

full_df = vcat(df_list...)
full_df.ParsedRewards = DataUtils.parse_array_string.(full_df.CueValues)
full_df.CleanRT = DataUtils.parse_clean_float.(full_df.RT)
filter!(row -> !ismissing(row.CleanRT) && 0.05 < row.CleanRT < 3.0, full_df)

println("\nTotal rows after RT filtering: $(nrow(full_df))")

# Analyze PointTargetResponse vs CueResponseValue
println("\n" * "=" ^ 70)
println("POINTTARGETRESPONSE vs CUERESPONSEVALUE ANALYSIS")
println("=" ^ 70)

# Create a comparison
comparisons = []
for row in eachrow(full_df)
    ptr = DataUtils.parse_clean_float(row.PointTargetResponse)
    crv = DataUtils.parse_clean_float(row.CueResponseValue)
    rewards = row.ParsedRewards
    n_options = length(rewards)
    
    # Find index of CueResponseValue in ParsedRewards
    crv_idx = isnothing(crv) || ismissing(crv) ? nothing : findfirst(x -> x == crv, rewards)
    
    push!(comparisons, (
        ptr = isnothing(ptr) || ismissing(ptr) ? 0 : Int(ptr),
        crv = isnothing(crv) || ismissing(crv) ? 0.0 : crv,
        crv_idx = isnothing(crv_idx) ? 0 : crv_idx,
        n_options = n_options,
        rewards = rewards
    ))
end

comparison_df = DataFrame(comparisons)

println("\nPointTargetResponse value distribution:")
ptr_counts = countmap(comparison_df.ptr)
for (val, count) in sort(collect(ptr_counts), by=x->x[1])
    println("  PTR=$val: $count trials")
end

println("\nRelationship between PointTargetResponse and CueResponseValue index:")
for ptr_val in sort(unique(comparison_df.ptr))
    if ptr_val > 0
        subset = comparison_df[comparison_df.ptr .== ptr_val, :]
        println("\n  PointTargetResponse = $ptr_val:")
        println("    Total trials: $(nrow(subset))")
        
        # Check how many match
        matches = sum(subset.ptr .== subset.crv_idx)
        println("    Matches crv_idx: $matches ($(round(100*matches/nrow(subset), digits=1))%)")
        
        # Check n_options distribution
        n_opts_counts = countmap(subset.n_options)
        println("    Number of options distribution:")
        for (n, count) in sort(collect(n_opts_counts), by=x->x[1])
            println("      $n options: $count trials")
        end
        
        # Show sample where they don't match
        mismatches = subset[subset.ptr .!= subset.crv_idx, :]
        if nrow(mismatches) > 0
            println("    Sample mismatches (first 5):")
            for i in 1:min(5, nrow(mismatches))
                r = mismatches[i, :]
                println("      PTR=$ptr_val, crv_idx=$(r.crv_idx), n_options=$(r.n_options), rewards=$(r.rewards)")
            end
        end
    end
end

# Check invalid choices
println("\n" * "=" ^ 70)
println("INVALID CHOICE ANALYSIS")
println("=" ^ 70)

invalid_cases = []
for (idx, row) in enumerate(eachrow(comparison_df))
    ptr = row.ptr
    crv_idx = row.crv_idx
    n_options = row.n_options
    
    # Current logic: use PTR directly
    choice_current = ptr
    is_invalid_current = choice_current <= 0 || choice_current > n_options
    
    # Alternative: use crv_idx
    choice_alt = crv_idx
    is_invalid_alt = choice_alt <= 0 || choice_alt > n_options
    
    if is_invalid_current
        push!(invalid_cases, (
            ptr = ptr,
            crv_idx = crv_idx,
            n_options = n_options,
            rewards = row.rewards,
            would_be_valid_with_crv = !is_invalid_alt
        ))
    end
end

invalid_df = DataFrame(invalid_cases)

println("\nInvalid choices with current logic (using PointTargetResponse):")
println("  Total: $(length(invalid_cases))")

if !isempty(invalid_df)
    println("\n  How many would be valid if we used CueResponseValue index instead?")
    would_be_valid = sum(invalid_df.would_be_valid_with_crv)
    println("    Would be valid: $would_be_valid / $(nrow(invalid_df)) ($(round(100*would_be_valid/nrow(invalid_df), digits=1))%)")
    
    println("\n  Breakdown by PointTargetResponse value:")
    for ptr_val in sort(unique(invalid_df.ptr))
        if ptr_val > 0
            subset = invalid_df[invalid_df.ptr .== ptr_val, :]
            println("    PTR=$ptr_val: $(nrow(subset)) invalid trials")
            println("      n_options distribution:")
            n_opts_counts = countmap(subset.n_options)
            for (n, count) in sort(collect(n_opts_counts), by=x->x[1])
                println("        $n options: $count trials")
            end
        end
    end
end

# Summary recommendation
println("\n" * "=" ^ 70)
println("RECOMMENDATION")
println("=" ^ 70)

if !isempty(invalid_df)
    would_be_valid_pct = 100 * sum(invalid_df.would_be_valid_with_crv) / nrow(invalid_df)
    println("\nIf we use CueResponseValue index instead of PointTargetResponse:")
    println("  Would recover: $would_be_valid_pct% of invalid trials")
    println("  This suggests PointTargetResponse may use a different encoding")
    println("  (e.g., fixed positions 1-4) while CueResponseValue matches actual selection")
end

