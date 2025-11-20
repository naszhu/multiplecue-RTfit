# ==========================================================================
# RT Distribution Analysis
# Investigating the bimodal nature of response times
# ==========================================================================

include("data_utils.jl")
using .DataUtils
using Statistics
using Plots

const DATA_PATH = joinpath("..", "data", "ParticipantCPP002-003", "ParticipantCPP002-003")
const FILE_PATTERN = "*.dat"

println("=" ^ 70)
println("LOADING DATA FOR RT ANALYSIS")
println("=" ^ 70)

# Load the data
data = load_and_process_data(DATA_PATH, FILE_PATTERN)

println("\n" * "=" ^ 70)
println("RT DISTRIBUTION STATISTICS")
println("=" ^ 70)

rts = data.CleanRT

println("Number of trials: $(length(rts))")
println("Mean RT: $(round(mean(rts), digits=4)) s")
println("Median RT: $(round(median(rts), digits=4)) s")
println("Std RT: $(round(std(rts), digits=4)) s")
println("Min RT: $(round(minimum(rts), digits=4)) s")
println("Max RT: $(round(maximum(rts), digits=4)) s")
println("5th percentile: $(round(quantile(rts, 0.05), digits=4)) s")
println("25th percentile: $(round(quantile(rts, 0.25), digits=4)) s")
println("75th percentile: $(round(quantile(rts, 0.75), digits=4)) s")
println("95th percentile: $(round(quantile(rts, 0.95), digits=4)) s")

println("\n" * "=" ^ 70)
println("CHECKING FOR BIMODALITY")
println("=" ^ 70)

# Create histogram with many bins to see detail
nbins = 100
edges = range(minimum(rts), maximum(rts), length=nbins+1)
counts = zeros(Int, nbins)
for rt in rts
    idx = searchsortedlast(edges, rt)
    if 1 <= idx <= nbins
        counts[idx] += 1
    end
end
bin_centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

# Find peaks in the histogram
function find_peaks(counts, min_prominence=10)
    peaks = Int[]
    for i in 2:length(counts)-1
        if counts[i] > counts[i-1] && counts[i] > counts[i+1]
            if counts[i] >= min_prominence
                push!(peaks, i)
            end
        end
    end
    return peaks
end

peaks = find_peaks(counts, 5)
println("Number of peaks found: $(length(peaks))")
if !isempty(peaks)
    println("\nPeak locations (RT values):")
    for (i, peak_idx) in enumerate(peaks)
        println("  Peak $i: $(round(bin_centers[peak_idx], digits=4)) s (count: $(counts[peak_idx]))")
    end
end

# Identify potential express responses (very fast RTs)
fast_threshold = 0.3  # Common threshold for express responses
n_fast = sum(rts .< fast_threshold)
pct_fast = 100 * n_fast / length(rts)
println("\nRTs < $(fast_threshold)s (potential express): $n_fast ($(round(pct_fast, digits=2))%)")

# Check for natural split around typical express/deliberate boundary
for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]
    n_below = sum(rts .< threshold)
    pct_below = 100 * n_below / length(rts)
    mean_below = n_below > 0 ? mean(rts[rts .< threshold]) : NaN
    mean_above = sum(rts .>= threshold) > 0 ? mean(rts[rts .>= threshold]) : NaN

    println("  Threshold $(threshold)s: $(n_below) below ($(round(pct_below, digits=1))%), mean_below=$(round(mean_below, digits=3))s, mean_above=$(round(mean_above, digits=3))s")
end

println("\n" * "=" ^ 70)
println("GENERATING DETAILED PLOTS")
println("=" ^ 70)

# Plot 1: Overall RT distribution with fine bins
p1 = histogram(rts, bins=100, normalize=:pdf,
               label="RT Distribution", alpha=0.6,
               xlabel="Reaction Time (s)", ylabel="Density",
               title="RT Distribution (100 bins)")

# Add lines for potential express/deliberate boundary
vline!([0.3], label="0.3s threshold", color=:red, linestyle=:dash, linewidth=2)

# Plot 2: Log-scale histogram to see both modes clearly
p2 = histogram(rts, bins=100, normalize=:pdf,
               label="RT Distribution", alpha=0.6, yscale=:log10,
               xlabel="Reaction Time (s)", ylabel="Log Density",
               title="RT Distribution (log scale)")
vline!([0.3], label="0.3s threshold", color=:red, linestyle=:dash, linewidth=2)

# Plot 3: Separate fast vs slow RTs
fast_rts = rts[rts .< 0.3]
slow_rts = rts[rts .>= 0.3]

p3 = histogram(fast_rts, bins=50, normalize=:pdf,
               label="Fast RTs (<0.3s)", alpha=0.6, color=:blue,
               xlabel="Reaction Time (s)", ylabel="Density",
               title="Fast vs Slow RT Distributions")
histogram!(slow_rts, bins=50, normalize=:pdf,
          label="Slow RTs (≥0.3s)", alpha=0.6, color=:green)

# Plot 4: Cumulative distribution
sorted_rts = sort(rts)
p4 = plot(sorted_rts, (1:length(sorted_rts)) ./ length(sorted_rts),
          xlabel="Reaction Time (s)", ylabel="Cumulative Probability",
          title="Cumulative RT Distribution", label="CDF", linewidth=2)
vline!([0.3], label="0.3s threshold", color=:red, linestyle=:dash, linewidth=2)

# Combine plots
combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
savefig(combined, "rt_distribution_analysis.png")
println("Saved detailed RT analysis to: rt_distribution_analysis.png")

# Plot 5: Very detailed view of the full range
p5 = histogram(rts, bins=200, normalize=:pdf,
               xlabel="Reaction Time (s)", ylabel="Density",
               title="Detailed RT Distribution (200 bins)",
               label="Observed RT", alpha=0.7, size=(1000, 600))

# Mark the peaks
if !isempty(peaks)
    for peak_idx in peaks
        vline!([bin_centers[peak_idx]],
               color=:red, linestyle=:dash, linewidth=2,
               label=(peak_idx == peaks[1] ? "Peaks" : ""))
    end
end

savefig(p5, "rt_distribution_detailed.png")
println("Saved detailed histogram to: rt_distribution_detailed.png")

println("\n" * "=" ^ 70)
println("ANALYSIS COMPLETE")
println("=" ^ 70)
println("\nSummary:")
println("- $(length(rts)) valid trials")
println("- Mean RT: $(round(mean(rts), digits=3))s")
println("- Fast RTs (<0.3s): $n_fast ($(round(pct_fast, digits=1))%)")
println("- Slow RTs (≥0.3s): $(length(rts) - n_fast) ($(round(100-pct_fast, digits=1))%)")

if length(fast_rts) > 0 && length(slow_rts) > 0
    println("\nTwo-component statistics:")
    println("  Fast component: mean=$(round(mean(fast_rts), digits=3))s, std=$(round(std(fast_rts), digits=3))s")
    println("  Slow component: mean=$(round(mean(slow_rts), digits=3))s, std=$(round(std(slow_rts), digits=3))s")
    println("  Separation: $(round(mean(slow_rts) - mean(fast_rts), digits=3))s")
end
