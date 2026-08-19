#!/usr/bin/env bash
# Config + call R to overlay model predictions on observed data.
#
# Two plot types:
#   competitor_cost  -- ΔRT and P(chose high) by RT bin and competitor (original)
#                       Needs *_simtrials.csv  (written by Julia fitter)
#   rt_by_session    -- Median RT by session: observed dots+line vs prediction flat line
#                       Needs *_pred.csv       (written by Julia fitter)
#
# Incomplete subjects (<10 session files) are skipped — never abort the run.
#
# Requires: mis_lba_basic_multisub.jl fit for that mode first.

set -euo pipefail

# =============================================================================
# CONFIG (edit these) — same option map as compare_two_modes.jl / multisub
# =============================================================================
#
#   PLOT_TYPE: competitor_cost | rt_by_session
#
#   omega: freeW | expW
#          freeW = free omega1..4
#          expW  = exponential MIS (theta)
#
#   k:     shared | cueN | competitor | delta_r | conflict
#          shared     = one shared threshold k
#          cueN       = separate k for 1-cue vs 2-cue
#          competitor = k = k0 + beta_k * C     (C = lower reward)
#          delta_r    = k = k0 + beta_k * (H-L)
#          conflict   = k = k0 + beta_k * (3-(H-L))
#
#   CI:    shared | cueN
#   t0:    shared | cueN
#
# File-name tag (built automatically below):
#   freeW / expW
#   + _varyK | _kComp | _kDeltaR | _kConflict
#   + _varyCI / _varyT0
#   Example: expW_kDeltaR , freeW_kComp , freeW_varyK
# =============================================================================

# Which plot to make:
#   competitor_cost  = ΔRT and P(chose high) by RT bin (original)
#   rt_by_session    = observed dots + flat prediction line per condition
PLOT_TYPE="rt_by_session"

# Subjects to plot, e.g. 18:29 or 10,12,14
SUBJECTS="18:29"

# Model settings (change freely)
OMEGA="expW"          # freeW | expW
K="delta_r"           # shared | cueN | competitor | delta_r | conflict
CI="shared"           # shared | cueN
T0="shared"           # shared | cueN
NSIMS=40              # predictive / simtrial reps when fitting

# Sessions used for observed data (same as analysis default)
SESSIONS="6:17"

# Same rule as mis_lba_basic_multisub.jl / compare_two_modes.jl
MIN_N_SESSIONS=10

# If 1, also fit missing subjects with Julia before plotting
FIT_MISSING=1

# =============================================================================
# Build MODE_TAG + Julia CLI from the settings above (do not edit by hand)
# =============================================================================
build_mode_tag() {
  local omega="$1" k="$2" ci="$3" t0="$4"
  local tag="" extra=""

  if [[ "${omega}" == "freeW" || "${omega}" == "free" || "${omega}" == "freew" ]]; then
    tag="freeW"
  elif [[ "${omega}" == "expW" || "${omega}" == "exp" || "${omega}" == "expw" ]]; then
    tag="expW"
  else
    echo "ERROR: OMEGA must be freeW or expW, got: ${omega}" >&2
    exit 1
  fi

  if [[ "${ci}" == "cueN" || "${ci}" == "cuen" || "${ci}" == "cue_number" ]]; then
    extra="${extra}_varyCI"
  elif [[ "${ci}" != "shared" ]]; then
    echo "ERROR: CI must be shared or cueN, got: ${ci}" >&2
    exit 1
  fi

  if [[ "${k}" == "cueN" || "${k}" == "cuen" || "${k}" == "cue_number" || "${k}" == "varyk" ]]; then
    extra="${extra}_varyK"
  elif [[ "${k}" == "competitor" || "${k}" == "comp" || "${k}" == "kcomp" ]]; then
    extra="${extra}_kComp"
  elif [[ "${k}" == "delta_r" || "${k}" == "deltar" || "${k}" == "kdelta" ]]; then
    extra="${extra}_kDeltaR"
  elif [[ "${k}" == "conflict" || "${k}" == "kconflict" ]]; then
    extra="${extra}_kConflict"
  elif [[ "${k}" != "shared" ]]; then
    echo "ERROR: K must be shared|cueN|competitor|delta_r|conflict, got: ${k}" >&2
    exit 1
  fi

  if [[ "${t0}" == "cueN" || "${t0}" == "cuen" || "${t0}" == "cue_number" ]]; then
    extra="${extra}_varyT0"
  elif [[ "${t0}" != "shared" ]]; then
    echo "ERROR: T0 must be shared or cueN, got: ${t0}" >&2
    exit 1
  fi

  echo "${tag}${extra}"
}

MODE_TAG="$(build_mode_tag "${OMEGA}" "${K}" "${CI}" "${T0}")"
JULIA_FIT_ARGS=( "omega=${OMEGA}" "k=${K}" "CI=${CI}" "t0=${T0}" "nsims=${NSIMS}" )

echo "Plot type:    ${PLOT_TYPE}"
echo "Model config: OMEGA=${OMEGA}  K=${K}  CI=${CI}  T0=${T0}  NSIMS=${NSIMS}"
echo "Derived tag:  ${MODE_TAG}"
echo "Julia args:   ${JULIA_FIT_ARGS[*]}"

# =============================================================================
# Paths (usually leave alone)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_DIR="${SCRIPT_DIR}/data_prediction"
FIG_DIR="${SCRIPT_DIR}/figs/fit_aug3"
JULIA_FIT="${SCRIPT_DIR}/mis_lba_basic_multisub.jl"
DATA_DIR="${SCRIPT_DIR}/../../multiplecue-responsebox/exp/data_from_lab"

# R scripts
R_COMPETITOR="${SCRIPT_DIR}/plot_competitor_cost_pred_overlay.R"
R_RT_SESSION="${SCRIPT_DIR}/plot_rt_by_session_pred_overlay.R"

mkdir -p "${PRED_DIR}" "${FIG_DIR}"

# Validate PLOT_TYPE
if [[ "${PLOT_TYPE}" != "competitor_cost" && "${PLOT_TYPE}" != "rt_by_session" ]]; then
  echo "ERROR: PLOT_TYPE must be competitor_cost or rt_by_session, got: ${PLOT_TYPE}" >&2
  exit 1
fi

# Which prediction file does each plot type need?
pred_file_suffix() {
  if [[ "${PLOT_TYPE}" == "competitor_cost" ]]; then
    echo "_simtrials.csv"
  else
    echo "_pred.csv"
  fi
}

# subject has the required prediction file
has_pred_file() {
  local sid="$1"
  local suffix
  suffix="$(pred_file_suffix)"
  [[ -f "${PRED_DIR}/sub${sid}_mis_lba_${MODE_TAG}${suffix}" ]]
}

# Expand SUBJECTS like 10:15 or 10,12,14 -> newline-separated ids
expand_subjects() {
  local s="$1"
  if [[ "$s" == *:* ]]; then
    local a="${s%%:*}"
    local b="${s##*:}"
    seq "$a" "$b"
  else
    echo "$s" | tr ',' '\n'
  fi
}

# Count trial CSVs the same way Julia list_trial_csvs does
count_trial_csvs() {
  local subdir="$1"
  local n=0
  local f base
  shopt -s nullglob
  for f in "${subdir}"/*; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    if [[ "$base" == *_trials.csv ]]; then
      n=$((n + 1))
    elif [[ "$base" == *.csv && "$base" != *metadata* ]]; then
      n=$((n + 1))
    fi
  done
  shopt -u nullglob
  echo "$n"
}

# subject is fitable only if folder exists and has enough sessions
subject_is_complete() {
  local sid="$1"
  local subdir="${DATA_DIR}/sub${sid}"
  local n
  if [[ ! -d "${subdir}" ]]; then
    echo "  skip sub${sid}: no folder"
    return 1
  fi
  n="$(count_trial_csvs "${subdir}")"
  if [[ "${n}" -lt "${MIN_N_SESSIONS}" ]]; then
    echo "  skip sub${sid}: only ${n} session files (<${MIN_N_SESSIONS})"
    return 1
  fi
  return 0
}

mapfile -t ALL_SIDS < <(expand_subjects "${SUBJECTS}")

# ---- optional fit of missing complete subjects ----
if [[ "${FIT_MISSING}" == "1" ]]; then
  echo "Checking / fitting missing predictions for MODE_TAG=${MODE_TAG} ..."
  for sid in "${ALL_SIDS[@]}"; do
    [[ -n "${sid}" ]] || continue
    if has_pred_file "${sid}"; then
      echo "  cache hit: sub${sid}"
      continue
    fi
    if ! subject_is_complete "${sid}"; then
      continue
    fi
    echo "  fitting sub${sid} ..."
    if ! julia "${JULIA_FIT}" "subs=${sid}" "${JULIA_FIT_ARGS[@]}"; then
      echo "  skip sub${sid}: fit failed; will omit from plot"
    fi
  done
fi

# ---- subjects actually passed to R (must have prediction file) ----
PLOT_SIDS=()
for sid in "${ALL_SIDS[@]}"; do
  [[ -n "${sid}" ]] || continue
  if has_pred_file "${sid}"; then
    PLOT_SIDS+=("${sid}")
  else
    echo "  skip sub${sid} for plot: missing prediction (${MODE_TAG})"
  fi
done

if [[ "${#PLOT_SIDS[@]}" -eq 0 ]]; then
  echo "ERROR: no subjects have predictions for MODE_TAG=${MODE_TAG}. Nothing to plot." >&2
  exit 1
fi

# Build R subjects arg as first:last if contiguous, else comma list
PLOT_SUBJECTS="$(IFS=,; echo "${PLOT_SIDS[*]}")"
if [[ "${#PLOT_SIDS[@]}" -gt 1 ]]; then
  contiguous=1
  for ((i = 1; i < ${#PLOT_SIDS[@]}; i++)); do
    prev="${PLOT_SIDS[$((i - 1))]}"
    cur="${PLOT_SIDS[$i]}"
    if [[ "$((prev + 1))" -ne "${cur}" ]]; then
      contiguous=0
      break
    fi
  done
  if [[ "${contiguous}" -eq 1 ]]; then
    PLOT_SUBJECTS="${PLOT_SIDS[0]}:${PLOT_SIDS[-1]}"
  fi
fi

# ---- call the right R script ----
if [[ "${PLOT_TYPE}" == "competitor_cost" ]]; then
  echo "Calling R: competitor cost RT-bin overlay for subjects: ${PLOT_SUBJECTS} ..."
  Rscript "${R_COMPETITOR}" \
    --subjects="${PLOT_SUBJECTS}" \
    --mode_tag="${MODE_TAG}" \
    --pred_dir="${PRED_DIR}" \
    --fig_dir="${FIG_DIR}" \
    --data_dir="${DATA_DIR}" \
    --sessions="${SESSIONS}"
else
  echo "Calling R: RT-by-session prediction overlay for subjects: ${PLOT_SUBJECTS} ..."
  Rscript "${R_RT_SESSION}" \
    --subjects="${PLOT_SUBJECTS}" \
    --mode_tag="${MODE_TAG}" \
    --pred_dir="${PRED_DIR}" \
    --fig_dir="${FIG_DIR}" \
    --data_dir="${DATA_DIR}" \
    --sessions="${SESSIONS}"
fi

echo "Done. Figures in: ${FIG_DIR}"
