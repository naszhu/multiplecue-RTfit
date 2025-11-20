using CSV, DataFrames

filepath = "../data/ParticipantCPP002-003/ParticipantCPP002-003/CPP002 - 4 cues, reward 1234, square, complex, nocuemask-subj-003-ses-005-2025-03-26_10-33-00.dat"

# Test 1: Find header line
println("Finding header line...")
lines = readlines(filepath)
header_idx = findfirst(l -> occursin("ExperimentName", l) && occursin("RT", l), lines)
println("Header found at line: $header_idx")

# Test 2: Read with that header line
println("\nReading CSV with header=$header_idx...")
df = CSV.read(filepath, DataFrame; delim='\t', header=header_idx, silencewarnings=true)
println("DataFrame has $(nrow(df)) rows")
println("Column names (first 10): $(names(df)[1:10])")
