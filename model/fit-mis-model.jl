# ==========================================================================
# MIS-LBA Mixture Model Fitting Script
# ==========================================================================
# This script fits a Mixture Model (Express Saccades + Linear Ballistic Accumulator)
# to reaction time data from the MIS paradigm. 
#
# The Drift Rates (v) of the LBA are constrained by the MIS theory:
# v = Capacity * (Weight / Sum(Weights))
# where Weight is a function of the Reward Value presented on screen.
# ==========================================================================

using CSV
using DataFrames
using Glob
using Distributions
using SequentialSamplingModels
using Optim
using Statistics
using Random

# ==========================================================================
# 1. CONFIGURATION & PATHS
# ==========================================================================

# Define the relative path to the data
# Root is assumed to be one level up from 'model' folder
const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"

# ==========================================================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================================================

"""
    parse_array_string(str)

Parses a string representation of an array (e.g., "[1, 2, 3, 4]") into a Julia Vector{Float64}.
Handles the specific format found in the PsychoPy data files.
"""
function parse_array_string(str::AbstractString)
    # Remove brackets and extra whitespace
    clean_str = replace(str, r"[\[\]]" => "")
    if isempty(clean_str)
        return Float64[]
    end
    # Split by comma and parse
    return parse.(Float64, split(clean_str, ","))
end

"""
    find_header_and_columns(file)

Finds the header line and identifies column indices for RT, RespLoc, CueValues, and Practice.
Returns (header_line, col_indices_dict).
"""
function find_header_and_columns(file)
    open(file, "r") do io
        line_num = 0
        for line in eachline(io)
            line_num += 1
            # Look for a line that has tab-separated values and contains column names we expect
            if occursin('\t', line) && 
               (occursin("RT", line) || occursin("RespLoc", line) || occursin("CueValues", line))
                # Parse header to find column indices
                headers = split(line, '\t')
                col_indices = Dict{String, Int}()
                for (idx, header) in enumerate(headers)
                    header = strip(header)
                    if header in ["RT", "RespLoc", "CueValues", "Practice"]
                        col_indices[header] = idx
                    end
                end
                return (line_num, col_indices)
            end
        end
        return (1, Dict{String, Int}())  # Default if not found
    end
end

"""
    process_single_file(file)

Processes a single data file: loads, filters, and returns cleaned DataFrame.
Uses streaming approach to only read needed columns for memory efficiency.
"""
function process_single_file(file)
    try
        # Find header and column indices
        header_line, col_indices = find_header_and_columns(file)
        
        if isempty(col_indices) || !("RT" in keys(col_indices)) || 
           !("RespLoc" in keys(col_indices)) || !("CueValues" in keys(col_indices))
            println("Warning: Required columns not found in $file")
            return nothing
        end
        
        # Read file line by line, extracting only needed columns
        rows = []
        open(file, "r") do io
            # Skip to data rows (after header)
            for _ in 1:header_line
                readline(io)
            end
            
            # Read data rows
            for line in eachline(io)
                fields = split(line, '\t')
                if length(fields) >= maximum(values(col_indices))
                    row = Dict{String, Any}()
                    if "RT" in keys(col_indices)
                        row["RT"] = fields[col_indices["RT"]]
                    end
                    if "RespLoc" in keys(col_indices)
                        row["RespLoc"] = fields[col_indices["RespLoc"]]
                    end
                    if "CueValues" in keys(col_indices)
                        row["CueValues"] = fields[col_indices["CueValues"]]
                    end
                    if "Practice" in keys(col_indices)
                        row["Practice"] = fields[col_indices["Practice"]]
                    end
                    push!(rows, row)
                end
            end
        end
        
        if isempty(rows)
            return nothing
        end
        
        # Convert to DataFrame
        dt = DataFrame(rows)
        
        # Early filtering to reduce memory: Remove practice trials
        if "Practice" in names(dt)
            filter!(row -> row.Practice != "Y", dt)
        end
        
        # Filter valid RTs early and convert to Float64
        filter!(row -> !ismissing(row.RT) && row.RT != "", dt)
        # Convert RT to Float64, handling string values
        rt_values = Float64[]
        for val in dt.RT
            try
                push!(rt_values, parse(Float64, string(val)))
            catch
                push!(rt_values, NaN)
            end
        end
        dt.RT = rt_values
        filter!(row -> !isnan(row.RT) && 0.05 < row.RT < 2.0, dt)
        
        # Skip if no valid rows remain
        if nrow(dt) == 0
            return nothing
        end
        
        # Parse CueValues (The Rewards)
        dt.ParsedRewards = parse_array_string.(dt.CueValues)
        
        # Determine Choice Index
        if "RespLoc" in names(dt)
            choice_values = Int[]
            for val in dt.RespLoc
                try
                    push!(choice_values, parse(Int, string(val)))
                catch
                    push!(choice_values, 0)  # Invalid choice
                end
            end
            dt.Choice = choice_values
            filter!(row -> row.Choice >= 1 && row.Choice <= 4, dt)  # Valid choices are 1-4
        else
            error("Column 'RespLoc' not found in file $file")
        end
        
        # Only keep necessary columns to reduce memory
        required_cols = ["RT", "Choice", "ParsedRewards"]
        if "Practice" in names(dt)
            required_cols = ["RT", "Choice", "ParsedRewards", "Practice"]
        end
        select!(dt, required_cols)
        
        return dt
    catch e
        println("Warning: Could not process file $file. Error: $e")
        return nothing
    end
end

"""
    load_and_process_data(path)

Loads all .dat files from the directory, processes them incrementally, and returns cleaned data.
Memory-efficient: processes and filters each file immediately to avoid storing all raw data.
"""
function load_and_process_data(path)
    files = glob(FILE_PATTERN, path)
    
    if isempty(files)
        error("No data files found in $path using pattern $FILE_PATTERN")
    end
    
    println("Found $(length(files)) data files. Loading and processing incrementally...")
    
    # Process files one at a time and accumulate results
    processed_dfs = DataFrame[]
    
    for (idx, file) in enumerate(files)
        print("Processing file $idx/$(length(files)): $(basename(file))... ")
        processed = process_single_file(file)
        if processed !== nothing && nrow(processed) > 0
            push!(processed_dfs, processed)
            println("✓ ($(nrow(processed)) rows)")
        else
            println("✗ (skipped)")
        end
    end
    
    if isempty(processed_dfs)
        error("No valid data found in any files")
    end
    
    # Combine all processed sessions (now much smaller in memory)
    println("Combining $(length(processed_dfs)) processed files...")
    full_df = vcat(processed_dfs...)
    
    println("Data loaded successfully. Total valid trials: $(nrow(full_df))")
    return full_df
end

# ==========================================================================
# 3. MODEL DEFINITION (LIKELIHOOD FUNCTION)
# ==========================================================================

"""
    mis_lba_mixture_loglike(params, data)

Calculates the negative log-likelihood of the data given the parameters.
Combines MIS-driven Drift Rates, LBA decision model, and Express Saccade mixture.

Parameters expected in `params` vector:
1. C (Capacity) - MIS parameter
2. w_slope (Reward sensitivity) - MIS parameter
3. A (Start point variability) - LBA parameter
4. k (Threshold gap: b = A + k) - LBA parameter
5. t0 (Non-decision time) - LBA parameter
6. prob_express (Mixing probability for express saccades)
7. mu_express (Mean RT of express saccades)
8. sigma_express (SD of express saccades)
"""
function mis_lba_mixture_loglike(params, df::DataFrame)
    # --- 1. Unpack Parameters ---
    C            = params[1] # Capacity (Total processing rate)
    w_slope      = params[2] # Slope for Weight = 1 + slope * Reward
    A            = params[3] # Max start point
    k            = params[4] # Distance from A to Threshold (b - A)
    t0           = params[5] # Non-decision time
    prob_express = params[6] # Probability of express saccade
    mu_express   = params[7] # Mean of express peak
    sigma_express= params[8] # SD of express peak
    
    # Check constraints to avoid numerical errors (return Inf if invalid)
    if C <= 0 || w_slope < 0 || A <= 0 || k <= 0 || t0 <= 0 || 
       prob_express < 0 || prob_express > 1 || sigma_express <= 0
        return Inf
    end

    total_neg_ll = 0.0
    
    # Pre-define the Express Saccade Distribution (Normal)
    dist_express = Normal(mu_express, sigma_express)
    
    # --- 2. Loop through trials ---
    # (Note: For massive datasets, vectorization is faster, but loops are clearer for custom logic)
    for i in 1:nrow(df)
        rt = df.RT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i] # Vector of 4 rewards
        
        # Skip if data is invalid
        if choice < 1 || choice > length(rewards)
            continue
        end

        # --- A. THE MIS MODEL PART (Theory) ---
        # Calculate Drift Rates (v) based on Rewards
        # Assumption: Weight_i = 1 + slope * Reward_i
        # (We add 1.0 base weight to ensure drift is non-zero even if reward is 0)
        weights = 1.0 .+ (w_slope .* rewards)
        
        # Relative Weights (Competition)
        # MIS Theory: The intention competes for the total Capacity C
        rel_weights = weights ./ sum(weights)
        
        # Drift Rates
        # v_i = Capacity * Relative_Weight_i
        drift_rates = C .* rel_weights
        
        # --- B. THE LBA MODEL PART (Engine) ---
        # Create LBA distribution for this specific trial
        # SequentialSamplingModels uses: LBA(; ν, A, k, τ)
        # ν = drift rates, A = start point max, k = b-A, τ = t0
        lba_dist = LBA(ν=drift_rates, A=A, k=k, τ=t0)
        
        # Calculate Likelihood of Regular Saccade
        # pdf returns the probability density of making 'choice' at time 'rt'
        # We use a try-catch block because LBA can fail numerically for impossible RTs (e.g., rt < t0)
        lik_regular = 0.0
        try
            # Construct the named tuple expected by SequentialSamplingModels pdf
            lik_regular = pdf(lba_dist, (choice=choice, rt=rt))
        catch
            lik_regular = 1e-10 # Small epsilon if model fails
        end
        
        # --- C. THE MIXTURE MODEL PART (Bimodal Data) ---
        # Calculate Likelihood of Express Saccade
        lik_express = pdf(dist_express, rt)
        
        # Combine them
        # L_total = p * L_express + (1-p) * L_regular
        lik_total = (prob_express * lik_express) + ((1 - prob_express) * lik_regular)
        
        # Handle numerical zeroes
        if lik_total <= 0
            lik_total = 1e-10
        end
        
        total_neg_ll -= log(lik_total)
    end
    
    return total_neg_ll
end

# ==========================================================================
# 4. OPTIMIZATION ROUTINE
# ==========================================================================

function fit_data()
    # 1. Load Data
    println("--- Loading Data from $DATA_PATH ---")
    df = load_and_process_data(DATA_PATH)
    
    # 2. Define Initial Parameters and Bounds
    # Order: [C, w_slope, A, k, t0, prob_express, mu_express, sigma_express]
    
    # Initial Guesses (Heuristics based on literature)
    # C: ~3-10 Hz
    # A: ~0.2-0.5s
    # k: ~0.2-0.5s
    # t0: ~0.1-0.2s (Standard non-decision)
    # p_exp: ~0.2 (20% express saccades)
    # mu_exp: ~0.1s (100ms express peak)
    lower_bounds = [1.0,  0.0, 0.01, 0.01, 0.01, 0.0,  0.05, 0.001]
    upper_bounds = [20.0, 5.0, 1.0,  1.0,  0.5,  0.5,  0.15, 0.05]
    initial_x    = [5.0,  0.5, 0.3,  0.3,  0.15, 0.15, 0.10, 0.01] 
    
    println("--- Starting Optimization ---")
    println("Initial Guess: $initial_x")
    
    # Define the objective function (single argument for Optim)
    objective(x) = mis_lba_mixture_loglike(x, df)
    
    # Run Optimization using Fminbox (for bounds) and BFGS
    # We use Fminbox to strictly enforce parameters stay physically meaningful
    result = optimize(objective, lower_bounds, upper_bounds, initial_x, Fminbox(BFGS()); 
                      autodiff = :forward, show_trace=true, time_limit=300.0)
    
    println("\n--- Optimization Complete ---")
    println(result)
    
    best_params = Optim.minimizer(result)
    println("\n--- Best Fitting Parameters ---")
    println("Capacity (C):        $(round(best_params[1], digits=4))")
    println("Reward Slope (w):    $(round(best_params[2], digits=4))")
    println("Start Var (A):       $(round(best_params[3], digits=4))")
    println("Threshold Gap (k):   $(round(best_params[4], digits=4))")
    println("Non-decision (t0):   $(round(best_params[5], digits=4))")
    println("Prob Express:        $(round(best_params[6], digits=4))")
    println("Mu Express:          $(round(best_params[7], digits=4))")
    println("Sigma Express:       $(round(best_params[8], digits=4))")
    
    return best_params, df
end

# ==========================================================================
# 5. RUN THE SCRIPT
# ==========================================================================

# Call the main function
best_params, data = fit_data()

# If you want to calculate BIC (Bayesian Information Criterion)
n_params = 8
n_data = nrow(data)
log_likelihood = -mis_lba_mixture_loglike(best_params, data)
bic = log(n_data)*n_params - 2*log_likelihood
println("BIC: $bic")

# ==========================================================================
# 6. CSV EXPORT OF RESULTS
# ==========================================================================

# Create a results DataFrame
results_df = DataFrame(
    Parameter = [
        "Capacity_C",
        "Reward_Slope_w",
        "Start_Var_A",
        "Threshold_Gap_k",
        "Non_decision_t0",
        "Prob_Express",
        "Mu_Express",
        "Sigma_Express",
        "BIC",
        "Log_Likelihood",
        "N_Trials",
        "N_Parameters"
    ],
    Value = [
        best_params[1],
        best_params[2],
        best_params[3],
        best_params[4],
        best_params[5],
        best_params[6],
        best_params[7],
        best_params[8],
        bic,
        log_likelihood,
        n_data,
        n_params
    ]
)

# Export to CSV
output_file = joinpath("..", "results", "mis_lba_fit_results.csv")
mkpath(dirname(output_file))  # Create directory if it doesn't exist
CSV.write(output_file, results_df)
println("\n--- Results exported to: $output_file ---")
