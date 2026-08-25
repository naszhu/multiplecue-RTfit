# Line-by-line walkthrough of `mis_lba_basic.jl`

This document explains [`mis_lba_basic.jl`](mis_lba_basic.jl) in **program execution order**. The file is a **script**, not a package: there is no `main()` function and almost no named helper functions. Julia executes the file from top to bottom when you `include` or run it. The only “function” defined later is the anonymous objective `obj` used by the optimizer.

What the script does overall:

1. Set model / data config constants.
2. Load and filter trial data from CSV files.
3. Build the free-parameter vector (names, bounds, start values).
4. Define a negative-log-likelihood objective and run boxed Nelder–Mead MLE.
5. Print fitted parameters and summary stats.
6. Plot choice accuracy (prediction vs data) by cue condition.
7. Plot RT densities (prediction vs data) by cue condition.

Inline `#` comments in each chunk are the line-by-line notes. Text outside chunks is only higher-level context.

---

## 1. Imports

```julia
using CSV, DataFrames, Statistics, Optim, Plots, SequentialSamplingModels, KernelDensity
# CSV: read trial CSVs
# DataFrames: tabular trial data (DataFrame, filter, select!, …)
# Statistics: mean for summary / plot aggregates
# Optim: optimize, Fminbox, NelderMead, Optim.minimizer / minimum
# Plots: choice and RT figures
# SequentialSamplingModels: LBA(...) and pdf(lba, (choice=..., rt=...))
# KernelDensity: kde of observed RTs for the density plots
```

---

## 2. Probability-model config

Conceptually:

- `:original`: \(P(i) \propto w_i\), where \(w_i = e^{\theta r_i / r_{\max}}\) (or \(\omega_0\) if \(r_i = 0\)).
- `:noise_model`: \(P(i) = (1-\varepsilon) P_{\text{original}}(i) + \varepsilon/4\).

```julia
# Probability model config:
# :original -> P(i) = exp(theta*r_i)/sum_j exp(theta*r_j)
# :noise_model -> P(i) = (1-epsilon)*softmax_i + epsilon*(1/4)
prob_model = :original
# pure softmax-like mapping from reward ranks to option weights;
# switch to :noise_model to mix that distribution with uniform guessing
epsilon = 0.05
# guessing mixture weight; only used when prob_model == :noise_model
@assert prob_model in (:original, :noise_model) "prob_model must be :original or :noise_model."
# abort if prob_model is anything other than the two allowed symbols
@assert 0.0 <= epsilon <= 1.0 "epsilon must be in [0, 1]."
# force epsilon into [0, 1]
```

---

## 3. LBA cue-number config

Trials are later tagged as `:single` (exactly one non-zero cue value) or `:multi` (more than one). These three flags control whether LBA parameters are shared across both trial types (`false` → one parameter keyed `:all`) or split (`true` → separate `*_single` and `*_multi` parameters).

```julia
# LBA cue-number config. :single = one cue, :multi = more than one cue.
vary_CI_by_cue_number = false
# CI shared (:all); set true for separate CI_single / CI_multi
vary_k_by_cue_number = true
# k split by cue number: k_single and k_multi
vary_t0_by_cue_number = false
# t0 shared (:all); set true for separate t0_single / t0_multi
```

---

## 4. Data-source config and CSV discovery

```julia
# Data source config (set ONE ID here, then extend the mapping below as needed):
# - CCP001_S1 (lab-processed source)
# - CPP001_S1 / CPP001_S2 / CPP001_S3
# - CPP002_S1 / CPP002_S2 / CPP002_S3
data_source_id = "CPP002_S1"
# active subject/session ID
# data_source_id = "CCP001_S1"
# alternate ID left commented out
data_source_paths = Dict(
    "CCP001_S1" => joinpath(@__DIR__, "..", "..", "multiplecue-responsebox", "exp", "data_from_lab", "extracted_data_processed"),
    "CPP001_S1" => joinpath(@__DIR__, "..", "data", "CPP001 - subj 1", "extracted data"),
    "CPP001_S2" => joinpath(@__DIR__, "..", "data", "CPP001 - subj 2", "extracted data"),
    "CPP001_S3" => joinpath(@__DIR__, "..", "data", "CPP001 - subj 3", "extracted data"),
    "CPP002_S1" => joinpath(@__DIR__, "..", "data", "ParticipantCPP002-001", "extracted data"),
    "CPP002_S2" => joinpath(@__DIR__, "..", "data", "ParticipantCPP002-002", "extracted data"),
    "CPP002_S3" => joinpath(@__DIR__, "..", "data", "ParticipantCPP002-003", "extracted data")
)
# maps each ID to a folder; @__DIR__ is this .jl file's directory (paths relative to model_new/)
@assert haskey(data_source_paths, data_source_id) "Unknown data_source_id: $(data_source_id). Add it to data_source_paths."
# fail fast if the chosen ID is missing from the dict
data_label = data_source_id
# used later in plot titles / filenames (same string as the ID)
data_dir = data_source_paths[data_source_id]
# look up the folder path for this ID
files = filter(f -> endswith(f, ".csv"), readdir(data_dir; join=true))
# readdir(..., join=true) lists full paths; filter keeps only .csv files
@assert !isempty(files) "No CSV files found in data_dir: $(data_dir)"
# abort if the folder has no CSVs
println("data_source_id: ", data_source_id)
println("data_label: ", data_label)
println("data_dir: ", data_dir)
# print which dataset is being used
```

---

## 5. Reading CSVs into one DataFrame

```julia
col_types = Dict(
    :Cues => String,
    :CueValues => String,
    :CueRanks => String,
    :RespLoc => String
)
# force these columns to stay String if present; prevents CSV type-guessing
# from turning cue strings into numbers (even when select= drops some of them later)
frames = DataFrame[]
# empty vector of DataFrames; one frame per file
for f in files
    # loop over every CSV path in files
    hdr = names(CSV.read(f, DataFrame; limit=0))
    # limit=0 reads only the header; names(...) gets column names without loading the whole file
    has_warmup = "WarmUpTrial" in string.(hdr)
    # check whether WarmUpTrial exists in that header
    if has_warmup
        push!(frames, CSV.read(f, DataFrame;
            types=col_types,
            select=[:Session, :WarmUpTrial, :CueValues, :PointTargetResponse, :CueCondition, :RT]
        ))
        # read needed columns in one shot and append to frames
    else
        println("WARNING: Missing WarmUpTrial in file: ", f, " -> assigning WarmUpTrial=1 for all rows (will be excluded by WarmUpTrial==0 filter).")
        df = CSV.read(f, DataFrame;
            types=col_types,
            select=[:Session, :CueValues, :PointTargetResponse, :CueCondition, :RT]
        )
        # read without WarmUpTrial
        df.WarmUpTrial = ones(Int, nrow(df))
        # invent WarmUpTrial=1 for every row → later WarmUpTrial==0 filter drops all rows from this file
        select!(df, [:Session, :WarmUpTrial, :CueValues, :PointTargetResponse, :CueCondition, :RT])
        # reorder columns to match the warm-up files
        push!(frames, df)
    end
end
data = reduce(vcat, frames; cols=:setequal)
# vertically concatenate all frames; cols=:setequal requires the same column set (order may differ)
@assert eltype(data.PointTargetResponse) <: Number "PointTargetResponse must be read as a numeric column."
@assert eltype(data.Session) <: Number "Session must be read as a numeric column."
@assert eltype(data.WarmUpTrial) <: Number "WarmUpTrial must be read as a numeric column."
@assert eltype(data.RT) <: Number "RT must be read as a numeric column."
# assert response, session, warm-up, and RT were parsed as numbers (not strings)
```

---

## 6. Trial filtering and CueValues sanity check

Each `filter` returns a new DataFrame keeping only matching rows. `ncodeunits` counts UTF-8 code units; for ASCII digits this equals string length.

```julia
# Keep only session 6 and later.
data = filter(row -> Int(row.Session) >= 6, data)
# drop early / practice sessions

# Keep only main trials (WarmUpTrial == 0).
data = filter(row -> Int(row.WarmUpTrial) == 0, data)
# keep main trials only

# Keep only trials with an actual choice location; 0 means timeout.
data = filter(row -> Int(row.PointTargetResponse) in 1:4, data)
# drop timeouts (0) and invalid codes
# Keep only positive RT trials. RT is logged in ms.
data = filter(row -> Float64(row.RT) > 0, data)
# drop non-positive RTs
# Require CueValues to be exactly 4 characters in every row.
@assert all(ncodeunits(string(v)) == 4 for v in data.CueValues) "CueValues must be 4-character strings in all rows."
# one digit per option / location
```

---

## 7. CPP condition-code remapping

CCP lab data is assumed to already use the label strings, so it skips this block.

```julia
# Convert CPP cue-condition numeric codes (1..10) to CCP-style labels ((1)..(4,3)).
condition_code_to_label = Dict(
    "1" => "(1)", "2" => "(2)", "3" => "(3)", "4" => "(4)",
    "5" => "(2,1)", "6" => "(3,1)", "7" => "(4,1)", "8" => "(3,2)", "9" => "(4,2)", "10" => "(4,3)"
)
# map numeric codes "1"…"10" onto human-readable cue-condition labels for plots / aggregation
if startswith(data_source_id, "CPP")
    data.CueCondition = [get(condition_code_to_label, string(cc), string(cc)) for cc in data.CueCondition]
    # only for CPP sources: replace each CueCondition via get(dict, key, default);
    # unknown codes fall back to string(cc) unchanged
end
```

---

## 8. Precompute trial arrays for the likelihood

```julia
# Precompute arrays for faster MLE optimization.
trial_rewards_arrarr = [parse.(Int, collect(string(v))) for v in data.CueValues]
# e.g. "4100" → [4,1,0,0]; collect → chars, parse.(Int, ...) → ints; Vector{Vector{Int}}
chosen_idx = Int.(data.PointTargetResponse)
# chosen location indices as Int (1–4)
rt_sec = startswith(data_source_id, "CPP") ? Float64.(data.RT) : Float64.(data.RT) ./ 1000
# RT in seconds; CPP treated as already seconds; non-CPP (e.g. CCP) ms → s via /1000
cue_number_type = [count(!=(0), r) == 1 ? :single : :multi for r in trial_rewards_arrarr]
# exactly one non-zero entry → :single, else :multi
r_max = maximum(vcat(trial_rewards_arrarr...))
# global max reward across all locations/trials; used to normalize r / r_max in weights
eps_val = 1e-12
# floor for likelihoods / logs so log(0) never happens

t0_upper = minimum(rt_sec) - 0.001
# upper bound for non-decision time must stay strictly below every observed RT
@assert t0_upper > 0.001 "All RTs are too small for estimating positive t0."
# require that upper bound itself is > 0.001 so a positive t0 is identifiable
```

---

## 9. Building the free-parameter vector

This block grows parallel arrays `param_names`, `lower`, `upper`, `x0`, and index maps so the optimizer works on a flat `x::Vector{Float64}` whose layout depends on the vary-* flags.

With the **current** flags (`vary_CI=false`, `vary_k=true`, `vary_t0=false`), the parameter vector is:

| Index | Name | Lower | Upper | Start `x0` |
|---|---|---|---|---|
| 1 | `theta` | 0 | 30 | 1.0 |
| 2 | `omega0` | 1e-6 | 50 | 0.1 |
| 3 | `CI_all` | 0.01 | 100 | 5.0 |
| 4 | `A` | 0.001 | 5 | 0.3 |
| 5 | `k_single` | 0.001 | 5 | 0.3 |
| 6 | `k_multi` | 0.001 | 5 | 0.3 |
| 7 | `t0_all` | 0.001 | `t0_upper` | `min(0.2, t0_upper/2)` |

```julia
param_names = String[]
lower = Float64[]
upper = Float64[]
x0 = Float64[]
# start empty containers for names, bounds, and start values

push!(param_names, "theta")
push!(lower, 0.0); push!(upper, 30.0); push!(x0, 1.0)
# append theta bounds / start

push!(param_names, "omega0")
push!(lower, 1e-6); push!(upper, 50.0); push!(x0, 0.1)
# append omega0 (baseline weight for zero-reward options)

idx_CI = Dict{Symbol,Int}()
idx_k = Dict{Symbol,Int}()
idx_t0 = Dict{Symbol,Int}()
# empty index dictionaries keyed by :single / :multi / :all

for key in (vary_CI_by_cue_number ? [:single, :multi] : [:all])
    push!(param_names, "CI_$(key)")
    push!(lower, 0.01); push!(upper, 100.0); push!(x0, 5.0)
    idx_CI[key] = length(param_names)
    # for each CI key, append "CI_$(key)" and store its position in idx_CI
end

push!(param_names, "A")
push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
idx_A = length(param_names)
# append shared start-point variability A; remember idx_A

for key in (vary_k_by_cue_number ? [:single, :multi] : [:all])
    push!(param_names, "k_$(key)")
    push!(lower, 0.001); push!(upper, 5.0); push!(x0, 0.3)
    idx_k[key] = length(param_names)
    # same pattern for k
end

for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
    push!(param_names, "t0_$(key)")
    push!(lower, 0.001); push!(upper, t0_upper); push!(x0, min(0.2, t0_upper / 2))
    idx_t0[key] = length(param_names)
    # same pattern for t0, with data-dependent upper bound and a cautious start value
end
```

---

## 10. Negative log-likelihood objective (`obj`)

This is the core model. It is a **closure**: it captures `trial_rewards_arrarr`, `chosen_idx`, `rt_sec`, `cue_number_type`, `r_max`, flags, `idx_*`, `prob_model`, `epsilon`, `eps_val` from the enclosing script scope. Optimizers **minimize** this, so smaller NLL = better fit.

```julia
# MLE over parameters needed in this experiment:
# theta  = reward-to-weight sensitivity
# omega0 = baseline weight for zero-reward options
# CI     = LBA processing capacity, scales relative weights into drift rates
# A      = LBA start-point variability
# k      = LBA threshold gap
# t0     = LBA non-decision time
obj = x -> begin
    theta = x[1]
    omega0 = x[2]
    A = x[idx_A]
    # unpack shared params from fixed / named indices
    nll = 0.0
    # initialize negative log-likelihood
    for i in eachindex(trial_rewards_arrarr)
        cue_key = cue_number_type[i]
        # :single or :multi for this trial
        CI = x[idx_CI[vary_CI_by_cue_number ? cue_key : :all]]
        k = x[idx_k[vary_k_by_cue_number ? cue_key : :all]]
        t0 = x[idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        # pick CI, k, t0 from the right slot of x; if a vary-* flag is false, always use :all
        rt_sec[i] <= t0 && return Inf
        # RT ≤ t0 is impossible under LBA → return Inf (bad parameter set)
        r = trial_rewards_arrarr[i]
        # reward vector for the trial, length 4
        w = [rv == 0 ? omega0 : exp(theta * rv / r_max) for rv in r]
        # zero-reward options get omega0; others get exp(theta * r / r_max)
        p = w ./ sum(w) # softmax-like probability
        # normalize to a probability vector p
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
            # optional noise mixture toward uniform 1/4
        end
        drift_rates = CI .* p
        # LBA drift rates ν = CI · p (capacity scales the probability vector)
        lba = LBA(ν=drift_rates, A=A, k=k, τ=t0)
        # construct LBA with drifts ν, start-point range A, threshold gap k, non-decision time τ=t0
        lik = try
            pdf(lba, (choice=chosen_idx[i], rt=rt_sec[i]))
            # joint density of (observed choice, observed RT)
        catch
            eps_val
            # any numerical error → treat likelihood as eps_val
        end
        nll -= log(max(lik, eps_val))
        # accumulate -log(max(lik, ε))
    end
    nll
    # return total NLL
end
```

---

## 11. Run MLE and unpack the best fit

```julia
fit = optimize(obj, lower, upper, x0, Fminbox(NelderMead()))
# derivative-free Nelder–Mead inside a box [lower, upper], starting at x0
best = Optim.minimizer(fit)
# parameter vector at the reported minimum
theta_hat = best[1]
omega0_hat = best[2]
A_hat = best[idx_A]
# convenient named scalars for shared params used in later prediction loops
nll = Optim.minimum(fit)
# best (minimum) NLL value
```

---

## 12. Per-trial predicted probability of the chosen option

Same weight → probability mapping as in `obj`, but using **fitted** `theta_hat` / `omega0_hat`. Mean of `chosen_prob` is a descriptive “how probable were the observed choices under the fitted Softmax/MIS weights?” (not the full LBA joint likelihood).

```julia
chosen_prob = Float64[]
for i in eachindex(trial_rewards_arrarr)
    r = trial_rewards_arrarr[i]
    w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
    # fitted Softmax weights
    p = w ./ sum(w)
    if prob_model == :noise_model
        p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
    end
    push!(chosen_prob, p[chosen_idx[i]])
    # record P(chosen location) for this trial
end
```

---

## 13. Print fit summary

Prints config flags, fitted parameters (looping over whichever CI / k / t0 keys exist), mean Softmax \(P(\text{chosen})\), and NLL.

```julia
println("Trials used: ", nrow(data))
println("prob_model: ", prob_model)
println("epsilon: ", epsilon)
println("vary_CI_by_cue_number: ", vary_CI_by_cue_number)
println("vary_k_by_cue_number: ", vary_k_by_cue_number)
println("vary_t0_by_cue_number: ", vary_t0_by_cue_number)
println("theta_hat: ", round(theta_hat, digits=6))
println("omega0_hat: ", round(omega0_hat, digits=6))
println("A_hat: ", round(A_hat, digits=6))
for key in (vary_CI_by_cue_number ? [:single, :multi] : [:all])
    println("CI_$(key)_hat: ", round(best[idx_CI[key]], digits=6))
end
for key in (vary_k_by_cue_number ? [:single, :multi] : [:all])
    println("k_$(key)_hat: ", round(best[idx_k[key]], digits=6))
end
for key in (vary_t0_by_cue_number ? [:single, :multi] : [:all])
    println("t0_$(key)_hat: ", round(best[idx_t0[key]], digits=6))
end
println("Mean predicted p(chosen): ", round(mean(chosen_prob), digits=4))
println("NLL: ", round(nll, digits=2))
```

---

## 14. Condition-wise “best option” prediction vs data

For each condition: **prediction** = mean Softmax mass on max-reward locations; **data** = empirical rate of choosing a max-reward location.

```julia
plot_cond = String[]
plot_pred = Float64[]
plot_data = Float64[]
# accumulators for the upcoming choice plot

# Canonical condition order from metadata.
condition_order = ["(1)", "(2)", "(3)", "(4)", "(2,1)", "(3,1)", "(4,1)", "(3,2)", "(4,2)", "(4,3)"]
# fixed display order of the 10 cue conditions
present_conditions = Set(string.(unique(data.CueCondition)))
# which conditions actually appear in the filtered data
ordered_conditions = [cc for cc in condition_order if cc in present_conditions]
# keep canonical order, drop absent conditions

for cc in ordered_conditions
    cond_rows = filter(row -> string(row.CueCondition) == cc, data)
    # subset of rows in that condition
    cond_chosen_prob = Float64[]
    cond_pred_best = Float64[]
    cond_data_best = Float64[]
    # per-condition accumulators
    for trial_row in eachrow(cond_rows)
        r = parse.(Int, collect(string(trial_row.CueValues)))
        w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
        p = w ./ sum(w)
        # rebuild r, w, p from fitted Softmax weights
        if prob_model == :noise_model
            p = (1 - epsilon) .* p .+ epsilon .* (1 / 4)
        end
        chosen_location = Int(trial_row.PointTargetResponse)
        push!(cond_chosen_prob, p[chosen_location])
        # Softmax P(chosen)
        best_reward = maximum(r)
        best_idx = findall(==(best_reward), r)
        # all locations tied for the maximum reward
        push!(cond_pred_best, sum(p[best_idx]))
        # prediction: Softmax probability mass on those best locations
        push!(cond_data_best, chosen_location in best_idx ? 1.0 : 0.0)
        # data: 1 if subject chose a best location, else 0
    end
    println(cc, " -> n=", nrow(cond_rows), ", mean p(chosen)=", round(mean(cond_chosen_prob), digits=4))
    # print condition sample size and mean Softmax P(chosen)
    push!(plot_cond, cc)
    push!(plot_pred, mean(cond_pred_best))
    push!(plot_data, mean(cond_data_best))
    # store condition label and mean pred / data "choose best" rates for plotting
end
```

---

## 15. Choice plot: prediction vs data

Example filename with current settings:  
`figs/mis_lba_basic_cpp002_s1_pred_vs_data_original_varyK.png`.

```julia
x = collect(1:length(plot_cond))
# numeric x-axis positions 1…n conditions
pfig = plot(
    x, plot_pred;
    label="Prediction", marker=:circle, lw=2, color=:blue,
    ylim=(0.8, 1.1)
)
# blue line = model mean P(choose a best location)
plot!(
    pfig, x, plot_data;
    label="Data", marker=:diamond, lw=2, color=:red
)
# red line = empirical rate of choosing a best location
xticks!(pfig, (x, plot_cond))
xlabel!(pfig, "Condition")
ylabel!(pfig, "Probability chosen")
title!(pfig, "MIS-LBA Prediction vs Data by CueCondition ($(data_label))")
# ticks labeled with condition strings; title includes data_label
fig_dir = joinpath(@__DIR__, "figs")
isdir(fig_dir) || mkdir(fig_dir)
# ensure figs/ exists under this script's directory
model_suffix = prob_model == :noise_model ? "_noise_model_eps$(replace(string(round(epsilon, digits=3)), "." => "p"))" : "_original"
# filename fragment for probability model (_original or _noise_model_eps0p05, etc.)
vary_suffix = join([
    vary_CI_by_cue_number ? "varyCI" : "",
    vary_k_by_cue_number ? "varyK" : "",
    vary_t0_by_cue_number ? "varyT0" : ""
] |> x -> filter(!isempty, x), "_")
# filename fragment listing which vary-* flags are on; empty strings filtered out
vary_suffix = isempty(vary_suffix) ? "" : "_$(vary_suffix)"
# if none are on, vary_suffix stays ""
fig_suffix = "$(model_suffix)$(vary_suffix)"
# combine model + vary suffixes
fig_path = joinpath(fig_dir, "mis_lba_basic_$(lowercase(data_label))_pred_vs_data$(fig_suffix).png")
savefig(pfig, fig_path)
println("Saved plot: ", fig_path)
# full path, save PNG, print confirmation
```

---

## 16. RT density plots by condition

For each condition: build an RT grid, overlay **data** KDE vs **model** marginal RT density averaged over trials (choice integrated out by summing \(f(c, RT)\) over \(c = 1..4\)).

```julia
rt_plots = []
for cc in ordered_conditions
    cond_idx = findall(i -> string(data.CueCondition[i]) == cc, 1:nrow(data))
    # row indices for condition cc
    cond_rt = rt_sec[cond_idx]
    isempty(cond_rt) && continue
    # RTs for this condition; skip if empty

    cond_t0 = [best[idx_t0[vary_t0_by_cue_number ? cue_number_type[i] : :all]] for i in cond_idx]
    rt_min = max(maximum(cond_t0) + 0.001, minimum(cond_rt) - 0.05)
    # lower edge: just above largest t0 used, and not far below smallest observed RT
    rt_max = maximum(cond_rt) + 0.05
    # upper edge slightly above max observed RT
    rt_grid = collect(range(rt_min, rt_max; length=150))
    # 150 evaluation points

    data_kde = kde(cond_rt)
    data_density = pdf(data_kde, rt_grid)
    # kernel density estimate of data RTs on rt_grid

    pred_density = zeros(length(rt_grid))
    for trial_i in cond_idx
        r = trial_rewards_arrarr[trial_i]
        w = [rv == 0 ? omega0_hat : exp(theta_hat * rv / r_max) for rv in r]
        p_choice = w ./ sum(w)
        if prob_model == :noise_model
            p_choice = (1 - epsilon) .* p_choice .+ epsilon .* (1 / 4)
        end
        # rebuild Softmax p_choice for this trial
        cue_key = cue_number_type[trial_i]
        CI = best[idx_CI[vary_CI_by_cue_number ? cue_key : :all]]
        k = best[idx_k[vary_k_by_cue_number ? cue_key : :all]]
        t0 = best[idx_t0[vary_t0_by_cue_number ? cue_key : :all]]
        # trial-appropriate CI / k / t0
        lba = LBA(ν=CI .* p_choice, A=A_hat, k=k, τ=t0)
        for (j, rt) in enumerate(rt_grid)
            pred_density[j] += sum(pdf(lba, (choice=choice, rt=rt)) for choice in 1:4)
            # add marginal RT density (choice integrated out)
        end
    end
    pred_density ./= length(cond_idx)
    # mean predictive density across trials in the condition

    push!(rt_plots, plot(
        rt_grid, pred_density;
        label="Prediction", color=:blue, lw=2,
        title=cc, xlabel="RT (s)", ylabel="Density"
    ))
    # subplot: blue = prediction
    plot!(rt_plots[end], rt_grid, data_density; label="Data", color=:red, lw=2)
    # overlay red = data KDE
end

rt_fig = plot(rt_plots...; layout=(ceil(Int, length(rt_plots) / 2), 2), size=(1000, 250 * ceil(Int, length(rt_plots) / 2)))
# combine subplots into a 2-column layout
rt_fig_path = joinpath(fig_dir, "mis_lba_basic_$(lowercase(data_label))_rt_distribution$(fig_suffix).png")
savefig(rt_fig, rt_fig_path)
println("Saved RT distribution plot: ", rt_fig_path)
# same fig_suffix as the choice plot
```

---

## Call / data-flow summary

There is no `main`. Top-level execution is itself the program. The only nested callable defined here is `obj`; everything else is sequential script code.

```text
script start
  ├─ CSV.read / DataFrames.filter  → build `data`
  ├─ build param vector (param_names, lower, upper, x0, idx_*)
  ├─ define obj(x)
  │     └─ per trial: weights → p → LBA → pdf → −log lik
  ├─ Optim.optimize(obj, …)       → best, nll
  ├─ Softmax diagnostics + println
  ├─ Plots: choice pred vs data    → savefig
  └─ Plots: RT density vs KDE      → savefig
         └─ per trial: LBA pdf summed over choices
```

| Symbol | Meaning |
|---|---|
| `theta` | reward-to-weight sensitivity |
| `omega0` | baseline weight for zero-reward options |
| `CI` | LBA processing capacity (scales \(p\) into drifts) |
| `A` | LBA start-point variability |
| `k` | LBA threshold gap |
| `t0` / `τ` | LBA non-decision time |

For the uninterrupted original source, open [`mis_lba_basic.jl`](mis_lba_basic.jl).
