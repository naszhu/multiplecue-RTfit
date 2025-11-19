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
        df = CSV.read(filepath, DataFrame;
                      delim='\t',
                      header=header_line,
                      silencewarnings=true)
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
            # Keep relevant columns if they exist
            cols_needed = ["RT", "CueValues", "RespLoc", "PointTargetResponse", "CueResponseValue"]
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
    for row in eachrow(full_df)
        c = 0

        # Strategy A: Check PointTargetResponse (most reliable)
        if "PointTargetResponse" in names(full_df)
            val = parse_clean_float(row.PointTargetResponse)
            if !ismissing(val)
                c = Int(val)
            end
        end

        # Strategy B: Check RespLoc if PointTargetResponse failed
        if c == 0 && "RespLoc" in names(full_df)
            val = parse_clean_float(row.RespLoc)
            if !ismissing(val)
                c = Int(val)
            end
        end

        # Strategy C: Infer from Reward if both failed
        if c == 0 && "CueResponseValue" in names(full_df)
            val = parse_clean_float(row.CueResponseValue)
            if !ismissing(val)
                idx = findfirst(x -> x == val, row.ParsedRewards)
                if !isnothing(idx)
                    c = idx
                end
            end
        end

        push!(choices, c)
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
