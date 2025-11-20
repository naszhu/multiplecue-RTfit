# ==========================================================================
# Data Utilities Module
# Functions for reading and processing PsychoPy data files
# ==========================================================================

module DataUtils

using CSV
using DataFrames
using Glob

export parse_clean_float, parse_array_string, read_psychopy_dat, load_and_process_data

"""
    parse_clean_float(val)
    Safely parses a value that might be a number, a string "1", or a bracketed string "[1]".
"""
function parse_clean_float(val)
    if ismissing(val) return missing end
    if isa(val, Number) return Float64(val) end

    s = string(val)
    # Remove brackets and whitespace
    clean_s = replace(s, r"[\[\]\s]" => "")
    if isempty(clean_s) || clean_s == "None" || clean_s == "nan"
        return missing
    end

    # Try parsing
    try
        return parse(Float64, clean_s)
    catch
        return missing
    end
end

"""
    parse_array_string(str)
    Parses "[1, 2, 3]" -> [1.0, 2.0, 3.0] or "0410" -> [0.0, 4.0, 1.0, 0.0]
"""
function parse_array_string(str)
    if ismissing(str) return Float64[] end
    s = string(str)
    clean_str = replace(s, r"[\[\]\s]" => "")
    if isempty(strip(clean_str)) return Float64[] end

    # Check if comma-separated (e.g., "1,2,3,4") or digit string (e.g., "0410")
    if occursin(",", clean_str)
        return parse.(Float64, split(clean_str, ","))
    else
        # Parse as individual digits - convert each Char to String first
        return parse.(Float64, string.(collect(clean_str)))
    end
end

"""
    read_psychopy_dat(filepath)
    Reads a PsychoPy .dat file, finding the correct header line.
"""
function read_psychopy_dat(filepath)
    # Find header line by reading all lines first
    lines = readlines(filepath)
    header_line = findfirst(l -> occursin("ExperimentName", l) && occursin("RT", l), lines)

    if isnothing(header_line)
        println("Warning: Could not find valid header in $filepath. Skipping.")
        return DataFrame()
    end

    try
        # Force CueValues to be read as String to preserve leading zeros
        # CueValues is always a 4-digit string (e.g., "0420", "1234")
        df = CSV.read(filepath, DataFrame;
                      delim='\t',
                      header=header_line,
                      silencewarnings=true,
                      types=Dict("CueValues" => String))
        return df
    catch e
        println("Error reading $filepath: $e")
        return DataFrame()
    end
end

"""
    load_and_process_data(path, file_pattern="*.dat")
    Loads and processes all data files from the specified path.
    Returns a cleaned DataFrame ready for model fitting.
"""
function load_and_process_data(path, file_pattern="*.dat")
    files = glob(file_pattern, path)
    if isempty(files)
        error("No data files found in $path")
    end

    # Filter out practice files based on filename pattern
    non_practice_files = filter(f -> !occursin("-Prac-", f), files)

    println("Found $(length(files)) files total, $(length(non_practice_files)) non-practice files. Processing...")

    df_list = DataFrame[]
    for (i, file) in enumerate(non_practice_files)
        dt = read_psychopy_dat(file)
        if !isempty(dt)
            # Filter out warm-up trials if WarmUpTrial column exists
            if "WarmUpTrial" in names(dt)
                before_warmup_filter = nrow(dt)
                filter!(row -> begin
                    val = row.WarmUpTrial
                    # Keep row if WarmUpTrial is missing, 0, false, or "0"
                    # Filter out if WarmUpTrial is 1, true, or "1"
                    if ismissing(val)
                        true  # Keep rows with missing WarmUpTrial
                    else
                        !(val == 1 || val == true || (isa(val, AbstractString) && strip(string(val)) == "1"))
                    end
                end, dt)
                if nrow(dt) < before_warmup_filter
                    println("  Filtered out $(before_warmup_filter - nrow(dt)) warm-up trials from $(basename(file))")
                end
            end
            
            # Keep relevant columns if they exist
            cols_needed = ["RT", "CueValues", "RespLoc", "PointTargetResponse", "CueResponseValue", "CueCondition"]
            cols_present = intersect(names(dt), cols_needed)
            select!(dt, cols_present)
            push!(df_list, dt)
        end
    end

    if isempty(df_list)
        error("No valid data could be read from files.")
    end

    println("Merging datasets...")
    full_df = vcat(df_list...)
    initial_rows = nrow(full_df)
    println("Total rows after merging: $initial_rows")

    # Parse CueValues
    full_df.ParsedRewards = parse_array_string.(full_df.CueValues)

    # Clean RT
    full_df.CleanRT = parse_clean_float.(full_df.RT)
    before_rt_filter = nrow(full_df)
    filter!(row -> !ismissing(row.CleanRT) && 0.05 < row.CleanRT < 3.0, full_df)
    println("After RT filtering: $(nrow(full_df)) (removed $(before_rt_filter - nrow(full_df)))")

    # Determine Choice
    choices = Int[]
    mismatches = 0
    mismatch_details = []
    
    for (row_idx, row) in enumerate(eachrow(full_df))
        c = 0
        n_options = length(row.ParsedRewards)
        choice_from_pt = 0
        choice_from_crv = 0

        # Strategy A: Use PointTargetResponse directly (primary method)
        # PointTargetResponse directly indicates which position was clicked (1-4)
        if "PointTargetResponse" in names(full_df)
            val = parse_clean_float(row.PointTargetResponse)
            if !ismissing(val)
                c_candidate = Int(val)
                # Validate: PointTargetResponse must be within bounds
                if c_candidate > 0 && c_candidate <= n_options
                    choice_from_pt = c_candidate
                    c = c_candidate
                end
            end
        end

        # Validation: Check if Choice from PointTargetResponse matches CueResponseValue
        # Verify that the reward value at the chosen position matches CueResponseValue
        if choice_from_pt > 0 && "CueResponseValue" in names(full_df)
            crv_val = parse_clean_float(row.CueResponseValue)
            if !ismissing(crv_val)
                # Check if the reward value at the chosen position matches CueResponseValue
                expected_reward = row.ParsedRewards[choice_from_pt]
                if crv_val != expected_reward
                    mismatches += 1
                    if length(mismatch_details) < 10  # Store first 10 mismatches for reporting
                        push!(mismatch_details, (
                            row=row_idx,
                            PointTargetResponse=row.PointTargetResponse,
                            CueResponseValue=row.CueResponseValue,
                            Choice_from_PT=choice_from_pt,
                            ExpectedReward_at_PT=expected_reward,
                            CueValues=row.CueValues,
                            ParsedRewards=row.ParsedRewards
                        ))
                    end
                end
            end
        end

        # Strategy B: Infer from CueResponseValue if PointTargetResponse failed
        # This handles cases where PointTargetResponse might be missing or invalid
        if c == 0 && "CueResponseValue" in names(full_df)
            val = parse_clean_float(row.CueResponseValue)
            if !ismissing(val)
                # Find all positions with this reward value
                matching_indices = findall(x -> x == val, row.ParsedRewards)
                if !isempty(matching_indices)
                    # If multiple positions have the same reward value, use first match
                    # (This is ambiguous, but we have no other information)
                    choice_from_crv = matching_indices[1]
                    c = choice_from_crv
                end
            end
        end

        # Strategy C: Check RespLoc if both failed
        if c == 0 && "RespLoc" in names(full_df)
            val = parse_clean_float(row.RespLoc)
            if !ismissing(val)
                c_candidate = Int(val)
                # Validate: RespLoc must be within bounds
                if c_candidate > 0 && c_candidate <= n_options
                    c = c_candidate
                end
            end
        end

        push!(choices, c)
    end
    
    # Report validation results
    if mismatches > 0
        println("\n⚠️  WARNING: Found $mismatches mismatches between PointTargetResponse and CueResponseValue")
        println("   (Reward value at chosen position ≠ CueResponseValue)")
        if !isempty(mismatch_details)
            println("\n   First $(length(mismatch_details)) example mismatches:")
            for (i, detail) in enumerate(mismatch_details)
                println("   Mismatch $i (Row $(detail.row)):")
                println("     PointTargetResponse: $(detail.PointTargetResponse) → Choice = $(detail.Choice_from_PT)")
                println("     Reward at position $(detail.Choice_from_PT): $(detail.ExpectedReward_at_PT)")
                println("     CueResponseValue: $(detail.CueResponseValue)")
                println("     CueValues: $(detail.CueValues) → ParsedRewards: $(detail.ParsedRewards)")
            end
        end
        println("\n   Note: Using Choice from PointTargetResponse (Strategy A) as it directly indicates response location.")
    else
        println("\n✓ Validation passed: All PointTargetResponse choices match CueResponseValue")
    end
    full_df.Choice = choices

    # Filter invalid choices
    before_choice_filter = nrow(full_df)
    filter!(row -> row.Choice > 0 && row.Choice <= length(row.ParsedRewards), full_df)
    println("After choice filtering: $(nrow(full_df)) (removed $(before_choice_filter - nrow(full_df)))")

    println("\nData loaded. Total rows: $initial_rows -> Valid trials: $(nrow(full_df))")
    if nrow(full_df) == 0
        error("All trials were filtered out! Check column names and data format.")
    end

    return full_df
end

end # module
