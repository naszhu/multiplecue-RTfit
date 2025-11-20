using CSV
using DataFrames
using Glob

const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"

# Use the fixed read function
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

# Test with all files
files = glob(FILE_PATTERN, DATA_PATH)
println("Found $(length(files)) files")

total_rows = 0
practice_rows = 0
non_practice_rows = 0

for (i, file) in enumerate(files)
    println("Reading file $i: $(basename(file))")
    df = read_psychopy_dat(file)
    println("  Rows: $(nrow(df)), Columns: $(ncol(df))")
    if !isempty(df)
        println("  Has Practice column: $("Practice" in names(df))")
        if "Practice" in names(df)
            global total_rows += nrow(df)
            global practice_rows += sum(df.Practice .== "Y")
            global non_practice_rows += sum(df.Practice .!= "Y")
        end
    end
end

println("\n=== Summary ===")
println("Total rows across all files: $total_rows")
println("Practice trials (Practice == 'Y'): $practice_rows")
println("Non-practice trials (Practice != 'Y'): $non_practice_rows")
