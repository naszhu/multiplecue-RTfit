# ==========================================================================
# MIS-LBA Mixture Model Fitting Script (Robust Version)
# ==========================================================================

using Pkg

# Ensure required packages are installed
# Uncomment if running for the first time:
# Pkg.add(["CSV", "DataFrames", "Glob", "Distributions", "SequentialSamplingModels", "Optim", "Statistics", "Random", "Plots"])

using CSV
using DataFrames
using Glob
using Distributions
using SequentialSamplingModels
using Optim
using Statistics
using Random
using Plots

# ==========================================================================
# 1. CONFIGURATION & PATHS
# ==========================================================================

# Define path relative to this script
const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"
const OUTPUT_CSV = "model_fit_results.csv"
const OUTPUT_PLOT = "model_fit_plot.png"

# ==========================================================================
# 2. DATA LOADING (Robust Parser)
# ==========================================================================

"""
    parse_array_string(str)
    Parses "[1, 2, 3]" -> [1.0, 2.0, 3.0]
"""
function parse_array_string(str)
    if ismissing(str) return Float64[] end
    s = string(str)
    clean_str = replace(s, r"[\[\]]" => "")
    if isempty(strip(clean_str)) return Float64[] end
    return parse.(Float64, split(clean_str, ","))
end

"""
    read_psychopy_dat(filepath)
    Reads a PsychoPy .dat file, skipping metadata headers.
"""
function read_psychopy_dat(filepath)
    # 1. Detect where the header starts
    header_line = 0
    open(filepath) do file
        for (i, line) in enumerate(eachline(file))
            if startswith(line, "ExperimentName") # The first column name
                header_line = i
                break
            end
        end
    end

    if header_line == 0
        println("Warning: Could not find header in $filepath. Skipping.")
        return DataFrame()
    end

    # 2. Read data starting from header_line
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

function load_and_process_data(path)
    files = glob(FILE_PATTERN, path)
    if isempty(files)
        error("No data files found in $path")
    end
    
    println("Found $(length(files)) files. Processing...")
    
    df_list = DataFrame[]
    for (i, file) in enumerate(files)
        dt = read_psychopy_dat(file)
        if !isempty(dt)
            # Select only necessary columns to save memory
            # We need: RT, CueValues, RespLoc, Practice (if exists)
            cols_to_keep = ["RT", "CueValues", "RespLoc"]
            if "Practice" in names(dt) push!(cols_to_keep, "Practice") end
            
            # Keep only existing columns
            select!(dt, intersect(names(dt), cols_to_keep))
            
            push!(df_list, dt)
        end
        if i % 5 == 0 print(".") end # Progress indicator
    end
    println("\nMerging datasets...")
    full_df = vcat(df_list...)

    # --- CLEANING ---
    # 1. Remove Practice
    if "Practice" in names(full_df)
        filter!(row -> row.Practice != "Y", full_df)
    end

    # 2. Parse CueValues
    full_df.ParsedRewards = parse_array_string.(full_df.CueValues)

    # 3. Clean RT and Choice
    # Remove missing/invalid RTs
    filter!(row -> !ismissing(row.RT) && isa(row.RT, Number), full_df)
    # Convert to Float64
    full_df.RT = Float64.(full_df.RT)
    
    # Filter biologically plausible RTs (e.g., 0.05s to 2.0s)
    filter!(row -> 0.05 < row.RT < 2.0, full_df)

    # Parse Choice (RespLoc)
    if "RespLoc" in names(full_df)
        # Ensure RespLoc is numeric. PsychoPy sometimes saves "None" or "[]"
        filter!(row -> !ismissing(row.RespLoc) && isa(row.RespLoc, Number), full_df)
        full_df.Choice = Int.(full_df.RespLoc)
    else
        error("Critical: 'RespLoc' column missing.")
    end

    println("Data loaded. Valid trials: $(nrow(full_df))")
    return full_df
end

# ==========================================================================
# 3. MODEL LOGIC
# ==========================================================================

function mis_lba_mixture_loglike(params, df::DataFrame)
    # Unpack
    C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp = params
    
    # Constraints
    if C<=0 || w_slope<0 || A<=0 || k<=0 || t0<=0 || 
       p_exp<0 || p_exp>1 || sig_exp<=0
        return Inf
    end

    total_neg_ll = 0.0
    dist_express = Normal(mu_exp, sig_exp)

    for i in 1:nrow(df)
        rt = df.RT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]

        # Validation
        if choice < 1 || choice > length(rewards) continue end

        # --- MIS THEORY ---
        # Weight = 1 + w * Reward
        weights = 1.0 .+ (w_slope .* rewards)
        rel_weights = weights ./ sum(weights)
        drift_rates = C .* rel_weights

        # --- LBA ---
        # b = A + k (k ensures b > A)
        lba = LBA(ν=drift_rates, A=A, k=k, τ=t0)
        
        lik_reg = 0.0
        try
            lik_reg = pdf(lba, (choice=choice, rt=rt))
        catch
            lik_reg = 1e-10
        end

        # --- MIXTURE ---
        lik_exp = pdf(dist_express, rt)
        lik_tot = (p_exp * lik_exp) + ((1-p_exp) * lik_reg)

        if lik_tot <= 1e-20 lik_tot = 1e-20 end
        total_neg_ll -= log(lik_tot)
    end
    return total_neg_ll
end

# ==========================================================================
# 4. FITTING & PLOTTING
# ==========================================================================

function run_analysis()
    # 1. Load
    data = load_and_process_data(DATA_PATH)
    
    # 2. Optimize
    # [C, w, A, k, t0, p_exp, mu_exp, sig_exp]
    lower = [1.0, 0.0, 0.01, 0.01, 0.01, 0.0,  0.05, 0.001]
    upper = [30.0,10.0, 1.0,  1.0,  0.5,  0.8,  0.20, 0.1]
    x0    = [10.0, 1.0, 0.3,  0.3,  0.1,  0.2,  0.10, 0.01] # Start mu_exp at 100ms

    println("Fitting model...")
    func = x -> mis_lba_mixture_loglike(x, data)
    res = optimize(func, lower, upper, x0, Fminbox(BFGS()); 
                   autodiff=:forward, time_limit=600.0)
    
    best = Optim.minimizer(res)
    println("Best Params: $best")

    # 3. Save Results
    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)", "ThreshGap(k)", "NonDec(t0)", "ProbExp", "MuExp", "SigExp"],
        Value = best
    )
    CSV.write(OUTPUT_CSV, results_df)
    println("Saved parameters to $OUTPUT_CSV")

    # 4. Visualization
    println("Generating plot...")
    
    # Histogram of data
    histogram(data.RT, normalize=true, label="Data", alpha=0.5, bins=50,
              xlabel="Reaction Time (s)", ylabel="Density", title="MIS-LBA Fit")
    
    # Model Prediction Curve
    # We simulate the aggregate prediction by averaging model PDF across all trials
    t_grid = range(0.0, 1.5, length=200)
    y_pred = zeros(length(t_grid))
    
    # Use a subset of trials to estimate the average predictive curve (for speed)
    n_samples = min(500, nrow(data))
    subset_indices = rand(1:nrow(data), n_samples)
    
    for t_idx in 1:length(t_grid)
        t = t_grid[t_idx]
        avg_lik = 0.0
        for i in subset_indices
            rewards = data.ParsedRewards[i]
            # Re-calculate drift for this trial
            ws = 1.0 .+ (best[2] .* rewards)
            vs = best[1] .* (ws ./ sum(ws))
            
            # LBA Probability (summed over any choice for aggregate RT)
            lba = LBA(ν=vs, A=best[3], k=best[4], τ=best[5])
            # Probability of responding *anything* at time t
            # Simple way: sum pdf over all choices
            lik_reg_t = sum([pdf(lba, (choice=c, rt=t)) for c in 1:length(vs)])
            
            # Express Probability
            lik_exp_t = pdf(Normal(best[7], best[8]), t)
            
            avg_lik += (best[6]*lik_exp_t + (1-best[6])*lik_reg_t)
        end
        y_pred[t_idx] = avg_lik / n_samples
    end
    
    plot!(t_grid, y_pred, label="Model", linewidth=3, color=:red)
    savefig(OUTPUT_PLOT)
    println("Saved plot to $OUTPUT_PLOT")
end

run_analysis()