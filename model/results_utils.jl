# ==========================================================================
# Results Utilities Module
# Functions for saving model fitting results
# ==========================================================================

module ResultsUtils

using DataFrames
using CSV
using Optim

export save_results, save_results_dual, save_results_single, save_results_allconditions

"""
    save_results(result, output_csv="model_fit_results.csv"; cue_condition=nothing)

    Saves the optimization results to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename
    - cue_condition: Optional cue condition identifier (for multi-condition fits)

    Returns parameter names and values as a DataFrame.
"""
function save_results(result, output_csv="model_fit_results.csv"; cue_condition=nothing)
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)",
                     "ThreshGap(k)", "NonDec(t0)", "ProbExp", "MuExp", "SigExp"],
        Value = best
    )

    # Add cue condition column if provided
    if !isnothing(cue_condition)
        results_df.CueCondition = fill(cue_condition, nrow(results_df))
    end

    CSV.write(output_csv, results_df)
    println("Saved parameters to $output_csv")

    return results_df
end

"""
    save_results_dual(result, output_csv="model_fit_results.csv"; cue_condition=nothing)

    Saves the optimization results for dual-LBA model to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename
    - cue_condition: Optional cue condition identifier

    Returns parameter names and values as a DataFrame.
"""
function save_results_dual(result, output_csv="model_fit_results.csv"; cue_condition=nothing)
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar1(A1)", "ThreshGap1(k1)",
                     "NonDec1(t0_1)", "StartVar2(A2)", "ThreshGap2(k2)", "NonDec2(t0_2)", "ProbMix(p_mix)"],
        Value = best
    )

    if !isnothing(cue_condition)
        results_df.CueCondition = fill(cue_condition, nrow(results_df))
    end

    # Create outputdata subfolder if it doesn't exist
    outputdata_dir = joinpath(@__DIR__, "outputdata")
    if !isdir(outputdata_dir)
        mkdir(outputdata_dir)
        println("Created outputdata directory: $outputdata_dir")
    end

    # Save to outputdata subfolder
    output_path = joinpath(outputdata_dir, basename(output_csv))
    CSV.write(output_path, results_df)
    println("Saved parameters to $output_path")

    return results_df
end

"""
    save_results_single(result, output_csv="model_fit_results.csv"; cue_condition=nothing)

    Saves the optimization results for single-LBA model to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename
    - cue_condition: Optional cue condition identifier

    Returns parameter names and values as a DataFrame.
"""
function save_results_single(result, output_csv="model_fit_results.csv"; cue_condition=nothing)
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)", "ThreshGap(k)", "NonDec(t0)"],
        Value = best
    )

    if !isnothing(cue_condition)
        results_df.CueCondition = fill(cue_condition, nrow(results_df))
    end

    # Create outputdata subfolder if it doesn't exist
    outputdata_dir = joinpath(@__DIR__, "outputdata")
    if !isdir(outputdata_dir)
        mkdir(outputdata_dir)
        println("Created outputdata directory: $outputdata_dir")
    end

    # Save to outputdata subfolder
    output_path = joinpath(outputdata_dir, basename(output_csv))
    CSV.write(output_path, results_df)
    println("Saved parameters to $output_path")

    return results_df
end

"""
    save_results_allconditions(result, output_csv="model_fit_results.csv")

    Saves the optimization results for all-conditions model (shared parameters) to a CSV file.

    Arguments:
    - result: Optim result object
    - output_csv: Output filename

    Returns parameter names and values as a DataFrame.
"""
function save_results_allconditions(result, output_csv="model_fit_results.csv")
    best = Optim.minimizer(result)

    results_df = DataFrame(
        Parameter = ["Capacity(C)", "RewardSlope(w)", "StartVar(A)", "ThreshGap(k)", "NonDec(t0)"],
        Value = best
    )

    # Add note that these are shared parameters
    results_df.Note = fill("Shared across all conditions", nrow(results_df))

    # Create outputdata subfolder if it doesn't exist
    outputdata_dir = joinpath(@__DIR__, "outputdata")
    if !isdir(outputdata_dir)
        mkdir(outputdata_dir)
        println("Created outputdata directory: $outputdata_dir")
    end

    # Save to outputdata subfolder
    output_path = joinpath(outputdata_dir, basename(output_csv))
    CSV.write(output_path, results_df)
    println("Saved SHARED parameters to $output_path")

    return results_df
end

end # module
