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
    # Find header line by scanning once; avoids loading whole file into memory
    header_line = nothing
    for (i, line) in enumerate(eachline(filepath))
        if occursin("ExperimentName", line) && occursin("RT", line)
            header_line = i
            break
        end
    end

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
            cols_needed = ["RT", "CueValues", "PointTargetResponse", "CueCondition"]
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

    # Determine Choice from PointTargetResponse
    nrows = nrow(full_df)
    choices = Vector{Int}(undef, nrows)
    point_responses = full_df.PointTargetResponse
    parsed_rewards = full_df.ParsedRewards
    @inbounds for i in 1:nrows
        n_options = length(parsed_rewards[i])
        val = parse_clean_float(point_responses[i])
        c = (ismissing(val) || val <= 0 || val > n_options) ? 0 : Int(val)
        choices[i] = c
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
