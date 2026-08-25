#!/usr/bin/env Rscript
# Overlay model predictions on competitor-cost RT-bin plots (obs dashed, pred solid).
#
# Plots (same structure as plot_perf_by_condition.R):
#   1) Competitor cost ΔRT by equal-n RT bin
#   2) P(chose high | RT bin) by competitor
#
# Called by: run_competitor_cost_overlay.sh
# Args (all optional; shell usually passes them):
#   --subjects=10:15
#   --mode_tag=expW_kDeltaR
#   --pred_dir=.../data_prediction
#   --fig_dir=.../figs/fit_aug3
#   --data_dir=.../data_from_lab
#   --sessions=6:17

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

# ---- parse CLI ----
args <- commandArgs(trailingOnly = TRUE)
get_opt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}

parse_range <- function(s) {
  s <- trimws(s)
  if (grepl(":", s, fixed = TRUE)) {
    ab <- strsplit(s, ":", fixed = TRUE)[[1]]
    return(as.integer(ab[1]):as.integer(ab[2]))
  }
  as.integer(strsplit(s, ",", fixed = TRUE)[[1]])
}

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) {
    # Rscript --file=...
    cmd <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", cmd[grep("^--file=", cmd)])
    if (length(f)) dirname(normalizePath(f[[1]])) else getwd()
  }
)

model_new <- normalizePath(script_dir)
proj_selection <- normalizePath(file.path(model_new, "..", ".."))
default_data_dir <- file.path(proj_selection, "multiplecue-responsebox", "exp", "data_from_lab")
default_pred_dir <- file.path(model_new, "data_prediction")
default_fig_dir <- file.path(model_new, "figs", "fit_aug3")

subject_ids <- parse_range(get_opt("subjects", "10:15"))
mode_tag <- get_opt("mode_tag", "expW_kDeltaR")
pred_dir <- get_opt("pred_dir", default_pred_dir)
fig_dir <- get_opt("fig_dir", default_fig_dir)
data_dir <- get_opt("data_dir", default_data_dir)
session_ids <- parse_range(get_opt("sessions", "6:17"))

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

message("subjects: ", paste(subject_ids, collapse = ","))
message("mode_tag: ", mode_tag)
message("pred_dir: ", pred_dir)
message("fig_dir:  ", fig_dir)

# ---- constants matching plot_perf_by_condition.R ----
competitor_triangle <- list("4" = c(1L, 2L, 3L), "3" = c(1L, 2L), "2" = c(1L))
focal_levels <- c("4", "3", "2")
competitor_levels <- c(1L, 2L, 3L)
n_rt_bins <- 5L
min_trials_rtbin <- n_rt_bins * 5L
rt_bin_labels <- c("Fastest 20%", "20–40%", "40–60%", "60–80%", "Slowest 20%")
comp_colors <- c("1" = "#1B9E77", "2" = "#D95F02", "3" = "#7570B3")
high_labels <- c(
  "4" = "high=4  (pairs vs single 4)",
  "3" = "high=3  (pairs vs single 3)",
  "2" = "high=2  (pairs vs single 2)"
)

contrast_map <- bind_rows(lapply(names(competitor_triangle), function(focal) {
  comps <- competitor_triangle[[focal]]
  data.frame(
    Focal = focal,
    Competitor = comps,
    SingleCond = focal,
    PairCond = paste0(comps, ",", focal),
    stringsAsFactors = FALSE
  )
}))
relevant_conds <- unique(c(contrast_map$SingleCond, contrast_map$PairCond))

# ---- helpers: load observed trials (simplified from plot_perf_by_condition.R) ----
normalize_condition_key <- function(values) {
  values <- as.integer(values)
  values <- sort(values[!is.na(values) & values != 0])
  if (length(values) == 0) return(NA_character_)
  paste(values, collapse = ",")
}

get_condition <- function(cue_condition, cue_values) {
  if (!is.na(cue_condition) && nzchar(trimws(cue_condition))) {
    digits <- gsub("[^0-9,]", "", cue_condition)
    parts <- strsplit(digits, ",", fixed = TRUE)[[1]]
    parts <- parts[nzchar(parts)]
    key <- normalize_condition_key(parts)
    if (!is.na(key)) return(key)
  }
  digits <- gsub("[^0-9]", "", cue_values)
  if (nchar(digits) != 4) return(NA_character_)
  vals <- as.integer(strsplit(digits, "")[[1]])
  normalize_condition_key(vals)
}

is_true_flag <- function(x) {
  tolower(trimws(as.character(x))) %in% c("1", "true", "t", "yes", "y")
}

is_timeout_response <- function(response) {
  r <- tolower(trimws(as.character(response)))
  is.na(r) | r == "" | r %in% c("timeout", "none")
}

trial_file_regex <- paste0(
  "^CCRP_subj(\\d+)_ses(\\d+)",
  "(?:\\(([^)]+)\\))?",
  "_trials",
  "(?:_(a\\d+))?",
  "\\.csv$"
)

load_obs_subject <- function(sid) {
  sub_dir <- file.path(data_dir, paste0("sub", sid))
  if (!dir.exists(sub_dir)) return(NULL)
  csv_paths <- list.files(sub_dir, pattern = "\\.csv$", full.names = TRUE)
  frames <- list()
  for (path in csv_paths) {
    if (is.na(file.info(path)$size) || file.info(path)$size == 0) next
    m <- regexec(trial_file_regex, basename(path), perl = TRUE, ignore.case = TRUE)
    hit <- regmatches(basename(path), m)[[1]]
    if (length(hit) == 0) next
    file_subj <- as.integer(hit[2])
    file_ses <- as.integer(hit[3])
    if (is.na(file_subj) || file_subj != sid) next
    if (!(file_ses %in% session_ids)) next
    df <- read.csv(path, stringsAsFactors = FALSE, colClasses = "character", check.names = FALSE)
    needed <- c(
      "Subject", "Session", "Block", "WarmUpTrial", "Response", "RT", "ACC",
      "CueValues", "CueCondition", "BurnInBlock",
      "CueRankResponse", "CueResponseValue"
    )
    for (col in needed) if (!(col %in% names(df))) df[[col]] <- NA_character_
    frames[[length(frames) + 1]] <- df
  }
  if (length(frames) == 0) return(NULL)
  bind_rows(frames) %>%
    mutate(
      SubjectNumInt = sid,
      SubjectFacet = paste0("sub", sid),
      Condition = mapply(get_condition, CueCondition, CueValues, USE.NAMES = FALSE),
      WarmUpFlag = is_true_flag(WarmUpTrial),
      BurnInFlag = is_true_flag(BurnInBlock) |
        (!is.na(BurnInBlock) & trimws(BurnInBlock) != "" &
           !is.na(suppressWarnings(as.integer(Block))) &
           as.integer(Block) == 0L),
      TimeoutFlag = is_timeout_response(Response),
      RT_num = suppressWarnings(as.numeric(RT)),
      ACC_num = suppressWarnings(as.numeric(ACC)),
      CueRespVal = suppressWarnings(as.integer(trimws(CueResponseValue))),
      CueRank = suppressWarnings(as.integer(trimws(CueRankResponse)))
    ) %>%
    filter(
      !WarmUpFlag,
      !BurnInFlag,
      !TimeoutFlag,
      !is.na(Condition),
      !is.na(RT_num),
      RT_num > 0,
      RT_num <= 4000
    )
}

load_sim_subject <- function(sid, mode_tag) {
  path <- file.path(pred_dir, paste0("sub", sid, "_mis_lba_", mode_tag, "_simtrials.csv"))
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, stringsAsFactors = FALSE)
  df %>%
    mutate(
      SubjectNumInt = sid,
      SubjectFacet = paste0("sub", sid),
      Condition = mapply(get_condition, CueCondition, CueValues, USE.NAMES = FALSE),
      RT_num = suppressWarnings(as.numeric(RT_ms)),
      ACC_num = suppressWarnings(as.numeric(ACC)),
      CueRespVal = suppressWarnings(as.integer(CueResponseValue)),
      CueRank = suppressWarnings(as.integer(CueRankResponse)),
      SimRep = if ("SimRep" %in% names(.)) as.integer(SimRep) else 1L
    ) %>%
    filter(!is.na(Condition), !is.na(RT_num), RT_num > 0)
}

# ---- compute RT-bin summaries for one source (obs or one sim rep) ----
compute_rtbin_delta_rt <- function(df, subject_facet) {
  choose_focal <- df %>%
    mutate(CondChr = as.character(Condition)) %>%
    filter(CondChr %in% relevant_conds) %>%
    mutate(
      IsSingle = CondChr %in% unique(contrast_map$SingleCond),
      IsPair = CondChr %in% unique(contrast_map$PairCond),
      FocalFromCond = dplyr::case_when(
        IsSingle ~ CondChr,
        IsPair ~ sub("^.*,", "", CondChr),
        TRUE ~ NA_character_
      )
    ) %>%
    filter(
      !is.na(FocalFromCond),
      !is.na(CueRespVal),
      CueRespVal == as.integer(FocalFromCond),
      (IsSingle & (is.na(CueRank) | CueRank == 1L | CueRespVal == as.integer(FocalFromCond))) |
        (IsPair & (is.na(CueRank) | CueRank == 1L))
    )

  if (nrow(choose_focal) == 0) return(NULL)

  rtbin <- choose_focal %>%
    group_by(CondChr) %>%
    filter(n() >= min_trials_rtbin) %>%
    mutate(
      RTBin = ntile(RT_num, n_rt_bins),
      RTBin = factor(RTBin, levels = 1:n_rt_bins, labels = rt_bin_labels)
    ) %>%
    ungroup() %>%
    group_by(CondChr, FocalFromCond, IsPair, IsSingle, RTBin) %>%
    summarize(
      MeanRT = mean(RT_num, na.rm = TRUE),
      Trials = n(),
      RTSE = if_else(Trials > 1, sd(RT_num, na.rm = TRUE) / sqrt(Trials), 0),
      .groups = "drop"
    )

  pair <- rtbin %>%
    filter(IsPair) %>%
    mutate(
      Competitor = as.integer(sub(",.*$", "", CondChr)),
      HighReward = FocalFromCond
    ) %>%
    filter(Competitor %in% competitor_levels, HighReward %in% focal_levels)

  single <- rtbin %>%
    filter(IsSingle) %>%
    transmute(
      HighRewardRaw = FocalFromCond,
      RTBin,
      BaselineRT = MeanRT,
      BaselineSE = RTSE
    )

  pair %>%
    mutate(HighRewardRaw = as.character(FocalFromCond)) %>%
    left_join(single, by = c("HighRewardRaw", "RTBin")) %>%
    filter(!is.na(BaselineRT)) %>%
    mutate(
      SubjectFacet = subject_facet,
      DeltaRT = MeanRT - BaselineRT,
      DeltaSE = sqrt(RTSE^2 + BaselineSE^2),
      HighReward = factor(HighReward, levels = focal_levels, labels = unname(high_labels[focal_levels])),
      Competitor = factor(Competitor, levels = competitor_levels)
    )
}

compute_rtbin_p_high <- function(df, subject_facet) {
  acc_df <- df %>%
    mutate(CondChr = as.character(Condition)) %>%
    filter(CondChr %in% relevant_conds) %>%
    mutate(
      IsPair = CondChr %in% unique(contrast_map$PairCond),
      FocalFromCond = if_else(
        IsPair,
        sub("^.*,", "", CondChr),
        CondChr
      ),
      ChoseFocal = as.integer(
        (!is.na(ACC_num) & ACC_num == 1) |
          (!is.na(CueRank) & CueRank == 1L &
             !is.na(CueRespVal) & CueRespVal == as.integer(FocalFromCond))
      )
    ) %>%
    filter(IsPair, !is.na(FocalFromCond), !is.na(ChoseFocal))

  if (nrow(acc_df) == 0) return(NULL)

  acc_df %>%
    group_by(CondChr) %>%
    filter(n() >= min_trials_rtbin) %>%
    mutate(
      RTBin = ntile(RT_num, n_rt_bins),
      RTBin = factor(RTBin, levels = 1:n_rt_bins, labels = rt_bin_labels),
      Competitor = as.integer(sub(",.*$", "", CondChr)),
      HighReward = FocalFromCond
    ) %>%
    filter(Competitor %in% competitor_levels, HighReward %in% focal_levels) %>%
    group_by(CondChr, HighReward, Competitor, RTBin) %>%
    summarize(
      PHigh = mean(ChoseFocal, na.rm = TRUE),
      Trials = n(),
      PSE = if_else(Trials > 1, sqrt(PHigh * (1 - PHigh) / Trials), 0),
      .groups = "drop"
    ) %>%
    mutate(
      SubjectFacet = subject_facet,
      HighReward = factor(HighReward, levels = focal_levels, labels = unname(high_labels[focal_levels])),
      Competitor = factor(Competitor, levels = competitor_levels),
      ymin = pmax(0, PHigh - PSE),
      ymax = pmin(1, PHigh + PSE)
    )
}

# Average sim-rep summaries
avg_delta_rt <- function(sim_df, subject_facet) {
  reps <- sort(unique(sim_df$SimRep))
  parts <- lapply(reps, function(r) {
    compute_rtbin_delta_rt(filter(sim_df, SimRep == r), subject_facet)
  })
  parts <- Filter(Negate(is.null), parts)
  if (length(parts) == 0) return(NULL)
  bind_rows(parts) %>%
    group_by(SubjectFacet, HighReward, Competitor, RTBin) %>%
    summarize(
      DeltaRT = mean(DeltaRT, na.rm = TRUE),
      DeltaSE = sd(DeltaRT, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      ymin = DeltaRT - ifelse(is.na(DeltaSE), 0, DeltaSE),
      ymax = DeltaRT + ifelse(is.na(DeltaSE), 0, DeltaSE),
      DeltaSE = ifelse(is.na(DeltaSE), 0, DeltaSE)
    )
}

avg_p_high <- function(sim_df, subject_facet) {
  reps <- sort(unique(sim_df$SimRep))
  parts <- lapply(reps, function(r) {
    compute_rtbin_p_high(filter(sim_df, SimRep == r), subject_facet)
  })
  parts <- Filter(Negate(is.null), parts)
  if (length(parts) == 0) return(NULL)
  bind_rows(parts) %>%
    group_by(SubjectFacet, HighReward, Competitor, RTBin) %>%
    summarize(
      PHigh = mean(PHigh, na.rm = TRUE),
      PSE = sd(PHigh, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      PSE = ifelse(is.na(PSE), 0, PSE),
      ymin = pmax(0, PHigh - PSE),
      ymax = pmin(1, PHigh + PSE)
    )
}

# ---- gather subjects ----
obs_delta_list <- list()
pred_delta_list <- list()
obs_phigh_list <- list()
pred_phigh_list <- list()
used_subjects <- integer(0)

for (sid in subject_ids) {
  sim_path <- file.path(pred_dir, paste0("sub", sid, "_mis_lba_", mode_tag, "_simtrials.csv"))
  if (!file.exists(sim_path)) {
    message("skip sub", sid, ": missing ", basename(sim_path))
    next
  }
  obs <- load_obs_subject(sid)
  if (is.null(obs) || nrow(obs) == 0) {
    message("skip sub", sid, ": no usable observed trials")
    next
  }
  sim <- load_sim_subject(sid, mode_tag)
  if (is.null(sim) || nrow(sim) == 0) {
    message("skip sub", sid, ": empty simtrials")
    next
  }

  facet <- paste0("sub", sid)
  od <- compute_rtbin_delta_rt(obs, facet)
  pd <- avg_delta_rt(sim, facet)
  op <- compute_rtbin_p_high(obs, facet)
  pp <- avg_p_high(sim, facet)

  if (is.null(od) || is.null(pd) || is.null(op) || is.null(pp)) {
    message("skip sub", sid, ": could not form RT-bin summaries")
    next
  }

  obs_delta_list[[length(obs_delta_list) + 1]] <- od %>% mutate(Source = "Data")
  pred_delta_list[[length(pred_delta_list) + 1]] <- pd %>% mutate(Source = "Prediction")
  obs_phigh_list[[length(obs_phigh_list) + 1]] <- op %>% mutate(Source = "Data")
  pred_phigh_list[[length(pred_phigh_list) + 1]] <- pp %>% mutate(Source = "Prediction")
  used_subjects <- c(used_subjects, sid)
  message("loaded sub", sid, ": observed data + simtrials ready to plot")
}

if (length(used_subjects) == 0) {
  stop(
    "No subjects had both observed data and simtrials for mode_tag=", mode_tag,
    ". Fit with mis_lba_basic_multisub.jl first (writes *_simtrials.csv)."
  )
}

facet_levels <- paste0("sub", used_subjects)

delta_df <- bind_rows(obs_delta_list, pred_delta_list) %>%
  mutate(
    SubjectFacet = factor(SubjectFacet, levels = facet_levels),
    Source = factor(Source, levels = c("Data", "Prediction")),
    ymin = DeltaRT - DeltaSE,
    ymax = DeltaRT + DeltaSE
  )

phigh_df <- bind_rows(obs_phigh_list, pred_phigh_list) %>%
  mutate(
    SubjectFacet = factor(SubjectFacet, levels = facet_levels),
    Source = factor(Source, levels = c("Data", "Prediction"))
  )

# ---- plots ----
# Data = dashed, Prediction = solid; color by Competitor
base_theme <- theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "top",
    axis.text.x = element_text(angle = 35, hjust = 1, size = 9)
  )

p_delta <- ggplot(
  delta_df,
  aes(
    x = RTBin, y = DeltaRT, color = Competitor, fill = Competitor,
    group = interaction(Competitor, Source),
    linetype = Source
  )
) +
  geom_hline(yintercept = 0, color = "gray50", linewidth = 0.6, linetype = "dashed") +
  geom_ribbon(aes(ymin = ymin, ymax = ymax, linetype = NULL), alpha = 0.10, color = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.2) +
  scale_color_manual(values = comp_colors, name = "Competitor") +
  scale_fill_manual(values = comp_colors, name = "Competitor") +
  scale_linetype_manual(values = c(Data = "dashed", Prediction = "solid"), name = "Source") +
  facet_grid(SubjectFacet ~ HighReward, scales = "free_y") +
  labs(
    title = paste0("Competitor cost ΔRT by RT bin  |  ", mode_tag),
    subtitle = "Dashed = data, solid = prediction. Color = competitor (weaker cue).",
    x = "RT bin (equal-n within condition)",
    y = "Competitor cost ΔRT (ms)"
  ) +
  base_theme

sub_tag <- if (length(used_subjects) == 1) {
  paste0("sub", used_subjects[[1]])
} else {
  paste0("sub", min(used_subjects), "-", max(used_subjects))
}
delta_file <- file.path(fig_dir, paste0(sub_tag, "_CompCost_RTbinDeltaRT_", mode_tag, ".png"))
ggsave(delta_file, p_delta, width = 18, height = max(10, 2.4 * length(used_subjects)), dpi = 300, limitsize = FALSE)
message("Saved: ", delta_file)

p_phigh <- ggplot(
  phigh_df,
  aes(
    x = RTBin, y = PHigh, color = Competitor, fill = Competitor,
    group = interaction(Competitor, Source),
    linetype = Source
  )
) +
  geom_ribbon(aes(ymin = ymin, ymax = ymax, linetype = NULL), alpha = 0.10, color = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.2) +
  scale_color_manual(values = comp_colors, name = "Competitor") +
  scale_fill_manual(values = comp_colors, name = "Competitor") +
  scale_linetype_manual(values = c(Data = "dashed", Prediction = "solid"), name = "Source") +
  facet_grid(SubjectFacet ~ HighReward, scales = "free_y") +
  labs(
    title = paste0("P(chose high | RT bin) by competitor  |  ", mode_tag),
    subtitle = "Dashed = data, solid = prediction. Color = competitor (weaker cue).",
    x = "RT bin (equal-n within condition)",
    y = "P(chose higher reward)"
  ) +
  base_theme

phigh_file <- file.path(fig_dir, paste0(sub_tag, "_CompCost_RTbinPChooseHigh_", mode_tag, ".png"))
ggsave(phigh_file, p_phigh, width = 18, height = max(10, 2.4 * length(used_subjects)), dpi = 300, limitsize = FALSE)
message("Saved: ", phigh_file)

message("Done. Subjects plotted: ", paste(used_subjects, collapse = ","))
