using CSV, DataFrames, Plots

const FIG_DIR = joinpath(@__DIR__, "figs", "figs_jun16")
const FULL_CSV = joinpath(FIG_DIR, "ccp001_mis_lba_params_full.csv")
const FULL_CSV_LEGACY = joinpath(FIG_DIR, "ccp001_mis_lba_params.csv")
const DUAL_CUE_CSV = joinpath(FIG_DIR, "ccp001_mis_lba_params_dual_cue.csv")
const LINE_LW = 2.5
const MARKER_SIZE = 5

const CHUNK_COLORS = Dict(
    "full" => :steelblue,
    "dual_cue" => :darkorange
)
const CHUNK_LABELS = Dict(
    "full" => "full data",
    "dual_cue" => "dual-cue only"
)

function load_chunk_csv(path::String, default_chunk::String)
  isfile(path) || error("Missing parameter CSV: $(path)")
  df = CSV.read(path, DataFrame)
  if !("data_chunk" in names(df))
    df.data_chunk = fill(default_chunk, nrow(df))
    select!(df, [:subject_id, :data_chunk, :n_trials, :theta, :omega0, :CI, :A, :k, :t0, :nll])
  end
  df
end

function resolve_full_csv()
  if isfile(FULL_CSV)
    return load_chunk_csv(FULL_CSV, "full")
  elseif isfile(FULL_CSV_LEGACY)
    return load_chunk_csv(FULL_CSV_LEGACY, "full")
  else
    error("Missing full-data CSV. Expected $(FULL_CSV) or $(FULL_CSV_LEGACY)")
  end
end

function plot_params_by_chunk(combined::DataFrame, fig_path::String)
  subjects = sort(unique(combined.subject_id))
  x = 1:length(subjects)
  subject_index = Dict(s => i for (i, s) in enumerate(subjects))
  param_cols = [:theta, :omega0, :CI, :A, :k, :t0]
  param_labels = ["theta", "omega0", "CI", "A", "k", "t0"]
  chunks = ["full", "dual_cue"]

  plots = Vector{Any}(undef, length(param_cols))
  for (i, col) in enumerate(param_cols)
    p = plot(legend=i == 1 ? :topright : false)
    for chunk in chunks
      sub = filter(row -> row.data_chunk == chunk, combined)
      isempty(sub) && continue
      ys = [sub[sub.subject_id .== sid, col][1] for sid in subjects]
      plot!(
        p, x, ys;
        label=CHUNK_LABELS[chunk],
        color=CHUNK_COLORS[chunk],
        lw=LINE_LW,
        marker=:circle,
        markersize=MARKER_SIZE
      )
    end
    plot!(
      p;
      xticks=(x, subjects),
      xrotation=45,
      ylabel=param_labels[i],
      legend=(i == 1)
    )
    plots[i] = p
  end

  plot(
    plots...;
    layout=(length(param_cols), 1),
    size=(760, 920),
    plot_title="CCP001 MIS+LBA parameters by data chunk",
    bottom_margin=5Plots.mm
  )
  savefig(fig_path)
end

full_df = resolve_full_csv()
dual_df = load_chunk_csv(DUAL_CUE_CSV, "dual_cue")
combined = vcat(full_df, dual_df; cols=:union)
sort!(combined, [:subject_id, :data_chunk])

combined_path = joinpath(FIG_DIR, "ccp001_mis_lba_params_by_chunk.csv")
CSV.write(combined_path, combined)
println("Saved combined table: ", combined_path)

fig_path = joinpath(FIG_DIR, "ccp001_mis_lba_param_lines_by_chunk_noise_eps0p05.png")
plot_params_by_chunk(combined, fig_path)
println("Saved plot: ", fig_path)
