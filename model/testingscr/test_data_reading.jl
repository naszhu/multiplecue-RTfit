using CSV, DataFrames

# Test reading one file
filepath = "../data/ParticipantCPP002-003/ParticipantCPP002-003/CPP002 - 4 cues, reward 1234, square, complex, nocuemask-subj-003-ses-005-2025-03-26_10-33-00.dat"

println("=== Testing Data Reading ===")

# Find header line
header_line = 0
open(filepath) do file
    for (i, line) in enumerate(eachline(file))
        if occursin("ExperimentName", line) && occursin("RT", line)
            header_line = i
            println("Found header at line: $i")
            break
        end
    end
end

println("\nReading file with header at line $header_line...")

# Read with CSV - the header parameter tells which ROW NUMBER contains the header
df = CSV.read(filepath, DataFrame; delim='\t', header=header_line, silencewarnings=true)

println("Rows in dataframe: $(nrow(df))")
println("First few column names: $(names(df)[1:10])")
println("\nUnique Practice values: $(unique(df.Practice))")
println("Practice == 'Y' count: $(sum(df.Practice .== "Y"))")
println("Practice == 'N' count: $(sum(df.Practice .== "N"))")
println("Non-practice rows (Practice != 'Y'): $(sum(df.Practice .!= "Y"))")

# Check if RT column has valid data
println("\nRT column sample (first 5 non-practice):")
non_prac = df[df.Practice .!= "Y", :]
println(non_prac.RT[1:min(5, nrow(non_prac))])
