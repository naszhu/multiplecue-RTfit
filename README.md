# multiplecue-RTfit

Cognitive model fitting for the multiple-cue reward selection task: **MIS** (intention / option weights from rewards) + **LBA** (choice and RT), plus older mixture / dual-LBA / PRW explorations.

Fits joint choice+RT by cue condition, mainly using lab data from [multiplecue-responsebox](https://github.com/naszhu/multiplecue-responsebox).

There is **no single fixed entry script** — pick the fitter that matches the question (basic MLE, multi-subject, vary-\(k\), overlays, bootstrap, etc.).

## Where to start

| Area | What |
| ---- | ---- |
| `model_new/` | Current MIS+LBA work (recommended) |
| `model_new/mis_lba_basic_explained.md` | Walkthrough of the basic model |
| `model/` | Older mixture / dual-LBA / PRW pipeline |
| `data-analyse/` | Exploratory RT-bimodality hypotheses (R) |
| `data/` | Local / older subject extracts |

Example (multi-subject fit):

```bash
cd model_new
julia mis_lba_basic_multisub.jl subs=12:14 omega=expW k=delta_r
```

Single-subject starter: `model_new/mis_lba_basic.jl`. Predictive overlays: shell scripts + `plot_*_overlay.R` in `model_new/`.

## Data

- Sibling path used by current multisub scripts: `../multiplecue-responsebox/exp/data_from_lab/`
- Optional local CSVs under `data/` (see `data/codebook.txt`)
- `.dat` → CSV helper: `convert_dat_to_csv.py`

## Dependencies

- **Julia**: CSV, DataFrames, Distributions, SequentialSamplingModels, Optim, Plots, … (no pinned `Project.toml` yet)
- **R**: for overlay / `data-analyse` scripts (tidyverse, etc.)

## Related repos

- [multiplecue-responsebox](https://github.com/naszhu/multiplecue-responsebox) — CCRP experiment + primary lab data
- [multiplecue-prospectiveM](https://github.com/naszhu/multiplecue-prospectiveM) — PM extension of the same paradigm
