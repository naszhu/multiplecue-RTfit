# ==========================================================================
# MIS-LBA Mixture Model Fitting Script (Fixed & Robust)
# ==========================================================================

using Pkg

# Ensure required packages are installed
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

const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"
const OUTPUT_CSV = "model_fit_results.csv"
const OUTPUT_PLOT = "model_fit_plot.png"

# ==========================================================================
# 2. DATA LOADING (Robust Parser)
# ==========================================================================

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

function load_and_process_data(path)
    files = glob(FILE_PATTERN, path)
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

    # 2. Parse CueValues
    full_df.ParsedRewards = parse_array_string.(full_df.CueValues)

    # 3. Clean RT
    # Handle string RTs like "[0.53]" or "0.53"
    full_df.CleanRT = parse_clean_float.(full_df.RT)
    before_rt_filter = nrow(full_df)
    filter!(row -> !ismissing(row.CleanRT) && 0.05 < row.CleanRT < 3.0, full_df)
    println("After RT filtering: $(nrow(full_df)) (removed $(before_rt_filter - nrow(full_df)))")

    # 4. Determine Choice
    # Try PointTargetResponse first, then RespLoc, then fallback to matching reward value
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
                # Find index of this reward in the ParsedRewards vector
                # Note: This assumes unique rewards or that the first match is correct
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

# ==========================================================================
# 3. MODEL LOGIC
# ==========================================================================

function mis_lba_mixture_loglike(params, df::DataFrame)
    # Unpack
    C, w_slope, A, k, t0, p_exp, mu_exp, sig_exp = params
    
    # Constraints (Strict check to prevent integrator errors)
    if C<=0 || w_slope<0 || A<=0 || k<=0 || t0<=0 || t0 < 0.01 ||
       p_exp<0 || p_exp>0.99 || sig_exp<=0
        return Inf
    end

    total_neg_ll = 0.0
    dist_express = Normal(mu_exp, sig_exp)

    # Accumulate log-likelihood
    for i in 1:nrow(df)
        rt = df.CleanRT[i]
        choice = df.Choice[i]
        rewards = df.ParsedRewards[i]

        # --- MIS THEORY ---
        # Weight = 1 + w * Reward
        # We add 1.0 to base weight to ensure drift > 0 even for 0 reward
        weights = 1.0 .+ (w_slope .* rewards)
        rel_weights = weights ./ sum(weights)
        drift_rates = C .* rel_weights

        # --- LBA ---
        # Parameters: ν=drift, A=max start, k=b-A, τ=non-decision
        # Note: A must be strictly positive for LBA
        lba = LBA(ν=drift_rates, A=A, k=k, τ=t0)
        
        lik_reg = 0.0
        # LBA density is 0 if RT < t0. We handle this gracefully.
        if rt > t0
            try
                # pdf(d, (choice, rt))
                lik_reg = pdf(lba, (choice=choice, rt=rt))
                if isnan(lik_reg) || isinf(lik_reg) lik_reg = 1e-10 end
            catch
                lik_reg = 1e-10
            end
        else
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
    # Params: [C, w, A, k, t0, p_exp, mu_exp, sig_exp]
    lower = [1.0, 0.0, 0.01, 0.05, 0.05, 0.0,  0.05, 0.001]
    upper = [30.0,10.0, 1.0,  1.0,  0.6,  0.8,  0.20, 0.1]
    x0    = [10.0, 1.0, 0.3,  0.3,  0.2,  0.2,  0.10, 0.02]

    println("Fitting model (this may take a minute)...")
    func = x -> mis_lba_mixture_loglike(x, data)
    
    # FIX: Pass time_limit inside Optim.Options
    opt_options = Optim.Options(time_limit = 600.0, show_trace = true, show_every=5)
    
    res = optimize(func, lower, upper, x0, Fminbox(BFGS()), opt_options; 
                   autodiff=:forward)
    
    best = Optim.minimizer(res)
    println("\n--- Optimization Complete ---")
    println("Best Params: $best")
    println("Min LogLikelihood: $(Optim.minimum(res))")

    # 3. Save Results
    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)", "ThreshGap(k)", "NonDec(t0)", "ProbExp", "MuExp", "SigExp"],
        Value = best
    )
    CSV.write(OUTPUT_CSV, results_df)
    println("Saved parameters to $OUTPUT_CSV")

    # 4. Visualization
    println("Generating plot...")
    
    # Histogram of Observed RT
    histogram(data.CleanRT, normalize=true, label="Observed", alpha=0.5, bins=60,
              xlabel="Reaction Time (s)", ylabel="Density", title="MIS-LBA Mixture Fit",
              color=:blue, legend=:topright)
    
    # Simulate Model Curve
    # We calculate the 'average' predicted PDF across all trials
    t_grid = range(0.05, 1.5, length=200)
    y_pred = zeros(length(t_grid))
    
    # Use a random subset of trials to approximate the curve
    n_samples = min(200, nrow(data))
    subset_indices = rand(1:nrow(data), n_samples)
    
    for (j, t) in enumerate(t_grid)
        avg_pdf = 0.0
        for i in subset_indices
            rewards = data.ParsedRewards[i]
            
            # Reconstruct parameters
            ws = 1.0 .+ (best[2] .* rewards)
            vs = best[1] .* (ws ./ sum(ws))
            
            # LBA PDF (summed over all choices)
            lba = LBA(ν=vs, A=best[3], k=best[4], τ=best[5])
            
            # Check if t > t0 for LBA
            lba_dens = 0.0
            if t > best[5]
                # Sum pdf of all possible choices
                lba_dens = sum([pdf(lba, (choice=c, rt=t)) for c in 1:length(vs)])
            end
            
            # Express PDF
            exp_dens = pdf(Normal(best[7], best[8]), t)
            
            # Mixture
            avg_pdf += (best[6] * exp_dens) + ((1-best[6]) * lba_dens)
        end
        y_pred[j] = avg_pdf / n_samples
    end
    
    plot!(t_grid, y_pred, label="Model Prediction", linewidth=3, color=:red)
    
    savefig(OUTPUT_PLOT)
    println("Saved plot to $OUTPUT_PLOT")
end

run_analysis()