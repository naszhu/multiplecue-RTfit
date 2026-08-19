#!/usr/bin/env Rscript
# Overlay model prediction (pooled, flat horizontal line) on observed median RT
# by session, per subject. Style follows aug_14_rt_median_by_session_colorCond
# from multiplecue-responsebox/analysis: 3 condition-group rows per subject panel
# (adjacent green, single blue, non-adjacent orange), dots = observed,
# line = model prediction (flat, pooled across sessions).
#
# Called by: run_plot_overlay.sh
#   --subjects=18:20    subject range or comma list
#   --mode_tag=expW_kDeltaR
#   --pred_dir=.../data_prediction
#   --fig_dir=.../figs/fit_aug3
#   --data_dir=.../multiplecue-responsebox/exp/data_from_lab
#   --sessions=6:17

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(grid)
  library(gridExtra)
})

# ---- parse CLI args ----
args <- commandArgs(trailingOnly = TRUE)
get_opt <- function(name, default = NULL) {
  pat <- paste0("^--", name, "=(.+)$")
  m <- regmatches(args, regexec(pat, args))
  hits <- Filter(function(x) length(x) == 2, m)
  if (length(hits) == 0) {
    if (is.null(default)) stop("Required arg missing: --", name)
    return(default)
  }
  hits[[length(hits)]][[2]]
}

parse_range <- function(s) {
  s <- trimws(s)
  if (grepl(":", s, fixed = TRUE)) {
    parts <- strsplit(s, ":", fixed = TRUE)[[1]]
    seq(as.integer(parts[1]), as.integer(parts[2]))
  } else {
    as.integer(strsplit(s, ",", fixed = TRUE)[[1]])
  }
}

subject_ids <- parse_range(get_opt("subjects", "18:20"))
mode_tag    <- get_opt("mode_tag",  "expW_kDeltaR")
pred_dir    <- get_opt("pred_dir")
fig_dir     <- get_opt("fig_dir")
data_dir    <- get_opt("data_dir")
session_ids <- parse_range(get_opt("sessions", "6:17"))

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

message("subjects: ", paste(subject_ids, collapse = ","))
message("mode_tag: ", mode_tag)
message("sessions: ", paste(range(session_ids), collapse = "-"))

# ---- condition definitions (same as source file) ----
reward_sets <- list(
  "1" = 1L, "2" = 2L, "3" = 3L, "4" = 4L,
  "1,2" = c(1L, 2L), "1,3" = c(1L, 3L), "1,4" = c(1L, 4L),
  "2,3" = c(2L, 3L), "2,4" = c(2L, 4L), "3,4" = c(3L, 4L)
)
condition_levels <- names(reward_sets)

normalize_condition_key <- function(values) {
  values <- as.integer(values)
  values <- sort(values[!is.na(values) & values != 0])
  if (length(values) == 0) return(NA_character_)
  paste(values, collapse = ",")
}

get_condition <- function(cue_condition, cue_values) {
  if (!is.na(cue_condition) && nzchar(trimws(cue_condition))) {
    digits <- gsub("[^0-9,]", "", cue_condition)
    parts  <- strsplit(digits, ",", fixed = TRUE)[[1]]
    parts  <- parts[nzchar(parts)]
    key    <- normalize_condition_key(parts)
    if (!is.na(key) && key %in% names(reward_sets)) return(key)
  }
  digits <- gsub("[^0-9]", "", cue_values)
  if (nchar(digits) != 4) return(NA_character_)
  vals <- as.integer(strsplit(digits, "")[[1]])
  key  <- normalize_condition_key(vals[vals != 0])
  if (is.na(key) || !(key %in% names(reward_sets))) return(NA_character_)
  key
}

get_reward_diff <- function(cond) {
  if (is.na(cond) || !grepl(",", cond, fixed = TRUE)) return(NA_real_)
  parts <- as.integer(strsplit(cond, ",", fixed = TRUE)[[1]])
  if (length(parts) != 2 || any(is.na(parts))) return(NA_real_)
  abs(parts[2] - parts[1])
}

label_response_device <- function(rd) {
  r <- tolower(trimws(rd))
  out <- rep("UnknownDevice", length(r))
  out[grepl("cedrus|response[_ -]?box", r)] <- "RB"
  out[grepl("keyboard", r)] <- "KB"
  out[grepl("self", r)] <- "SRB1"
  out
}

is_true_flag <- function(x) tolower(trimws(as.character(x))) %in% c("1", "true", "t", "yes", "y")
is_timeout   <- function(r) { r <- tolower(trimws(as.character(r))); is.na(r) | r == "" | r %in% c("timeout", "none") }

# ---- condition group definitions & colors ----
condition_group_defs <- list(
  list(
    key        = "adjacent_diff1",
    label      = "Adjacent pairs (reward diff = 1)",
    conditions = c("1,2", "2,3", "3,4"),
    colors     = c("(1,2)" = "#004529", "(2,3)" = "#238B45", "(3,4)" = "#78C679")
  ),
  list(
    key        = "single",
    label      = "Single-cue conditions",
    conditions = c("1", "2", "3", "4"),
    colors     = c("(1)" = "#053061", "(2)" = "#2166AC", "(3)" = "#4393C3", "(4)" = "#92C5DE")
  ),
  list(
    key        = "far_pairs",
    label      = "Non-adjacent pairs",
    conditions = c("1,3", "2,4", "1,4"),
    colors     = c("(1,3)" = "#7F2704", "(2,4)" = "#E6550D", "(1,4)" = "#FDAE6B")
  )
)
condition_group_order  <- vapply(condition_group_defs, `[[`, character(1), "label")
condition_to_group     <- setNames(
  rep(condition_group_order, vapply(condition_group_defs, function(g) length(g$conditions), integer(1))),
  unlist(lapply(condition_group_defs, `[[`, "conditions"))
)
condition_group_colors <- unlist(lapply(condition_group_defs, `[[`, "colors"))
condition_group_colors <- condition_group_colors[!duplicated(names(condition_group_colors))]

assign_group <- function(cond) {
  g <- condition_to_group[[as.character(cond)]]
  if (is.null(g)) NA_character_ else g
}

# ---- load observed lab data ----
trial_file_regex <- "^CCRP_subj(\\d+)_ses(\\d+)(?:\\([^)]+\\))?_trials(?:_a\\d+)?\\.csv$"

load_obs_data <- function(data_dir, subject_ids, session_ids) {
  rows <- list()
  for (sid in subject_ids) {
    sub_dir <- file.path(data_dir, paste0("sub", sid))
    if (!dir.exists(sub_dir)) { message("skip sub", sid, ": no folder"); next }
    csvs <- list.files(sub_dir, pattern = "\\.csv$", full.names = TRUE)
    for (path in csvs) {
      m   <- regexec(trial_file_regex, basename(path), perl = TRUE, ignore.case = TRUE)
      hit <- regmatches(basename(path), m)[[1]]
      if (length(hit) == 0) next
      file_sub <- as.integer(hit[2])
      file_ses <- as.integer(hit[3])
      if (is.na(file_sub) || file_sub != sid) next
      if (!(file_ses %in% session_ids)) next
      if (file.info(path)$size == 0) next
      rows[[length(rows) + 1]] <- data.frame(
        path = path, subject_id = sid, session = file_ses,
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows) == 0) return(NULL)
  file_df <- bind_rows(rows) %>%
    arrange(subject_id, session) %>%
    filter(!duplicated(paste(subject_id, session)))

  dfs <- lapply(seq_len(nrow(file_df)), function(i) {
    path <- file_df$path[i]
    df   <- tryCatch(
      read.csv(path, stringsAsFactors = FALSE, colClasses = "character", check.names = FALSE),
      error = function(e) NULL
    )
    if (is.null(df) || nrow(df) == 0) return(NULL)
    needed <- c("Subject", "Session", "WarmUpTrial", "Response", "RT",
                "CueValues", "CueCondition", "ResponseDevice", "BurnInBlock",
                "TrialWallClockTime")
    for (col in needed) if (!(col %in% names(df))) df[[col]] <- NA_character_
    df
  })
  bind_rows(Filter(Negate(is.null), dfs))
}

obs_raw <- load_obs_data(data_dir, subject_ids, session_ids)
if (is.null(obs_raw) || nrow(obs_raw) == 0) stop("No observed trial data found.")

obs_df <- obs_raw %>%
  mutate(
    SubjectNumInt = suppressWarnings(as.integer(gsub("[^0-9]", "", Subject))),
    SessionNumInt = suppressWarnings(as.integer(Session)),
    DeviceLabel   = label_response_device(ResponseDevice),
    Condition     = mapply(get_condition, CueCondition, CueValues, USE.NAMES = FALSE),
    WarmUpFlag    = is_true_flag(WarmUpTrial),
    BurnInFlag    = is_true_flag(BurnInBlock),
    TimeoutFlag   = is_timeout(Response),
    RT_num        = suppressWarnings(as.numeric(RT))
  ) %>%
  filter(
    SubjectNumInt %in% subject_ids,
    SessionNumInt %in% session_ids,
    !WarmUpFlag, !BurnInFlag, !TimeoutFlag,
    !is.na(RT_num), RT_num > 0, RT_num <= 4000,
    !is.na(Condition)
  )

if (nrow(obs_df) == 0) stop("No usable observed trials after filtering.")

obs_summary <- obs_df %>%
  group_by(SubjectNumInt, DeviceLabel, SessionNumInt, Condition) %>%
  summarize(MedianRT = median(RT_num, na.rm = TRUE), Trials = n(), .groups = "drop") %>%
  mutate(
    ConditionLabel = paste0("(", Condition, ")"),
    Session        = factor(SessionNumInt, levels = session_ids),
    SubjectFacet   = paste0("sub", SubjectNumInt, "_", DeviceLabel),
    ConditionGroup = factor(
      vapply(Condition, assign_group, character(1)),
      levels = condition_group_order
    )
  ) %>%
  filter(!is.na(ConditionGroup))

# ---- day-start vertical lines from TrialWallClockTime ----
day_vlines <- tryCatch({
  obs_raw %>%
    mutate(
      SubjectNumInt = suppressWarnings(as.integer(gsub("[^0-9]", "", Subject))),
      SessionNumInt = suppressWarnings(as.integer(Session)),
      DeviceLabel   = label_response_device(ResponseDevice)
    ) %>%
    filter(!is.na(TrialWallClockTime), nzchar(trimws(TrialWallClockTime))) %>%
    mutate(TrialDate = as.Date(substr(TrialWallClockTime, 1, 10), format = "%Y-%m-%d")) %>%
    filter(!is.na(TrialDate)) %>%
    group_by(SubjectNumInt, DeviceLabel, SessionNumInt) %>%
    summarize(SessionDate = min(TrialDate, na.rm = TRUE), .groups = "drop") %>%
    group_by(SubjectNumInt, DeviceLabel, SessionDate) %>%
    slice_min(order_by = SessionNumInt, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
      SubjectFacet = paste0("sub", SubjectNumInt, "_", DeviceLabel),
      VlineX       = match(SessionNumInt, session_ids)
    )
}, error = function(e) data.frame(SubjectFacet = character(), VlineX = numeric()))

# ---- load prediction CSVs (_pred.csv, pooled across sessions) ----
load_pred <- function(sid, pred_dir, mode_tag) {
  path <- file.path(pred_dir, paste0("sub", sid, "_mis_lba_", mode_tag, "_pred.csv"))
  if (!file.exists(path)) return(NULL)
  tryCatch(
    read.csv(path, stringsAsFactors = FALSE),
    error = function(e) NULL
  )
}

pred_all <- bind_rows(lapply(subject_ids, function(sid) {
  df <- load_pred(sid, pred_dir, mode_tag)
  if (is.null(df) || nrow(df) == 0) {
    message("skip sub", sid, ": no pred CSV for mode_tag=", mode_tag)
    return(NULL)
  }
  df
}))

if (nrow(pred_all) == 0) {
  stop("No prediction CSVs found for mode_tag=", mode_tag,
       ". Run Julia fitter first (writes *_pred.csv).")
}

# Normalize condition column from pred CSV  (e.g. "(2,1)" -> "1,2")
normalize_pred_cond <- function(x) {
  digits <- gsub("[^0-9]", "", as.character(x))
  if (nchar(digits) == 0) return(NA_character_)
  vals <- as.integer(strsplit(digits, "")[[1]])
  paste(sort(vals[vals != 0]), collapse = ",")
}

pred_summary <- pred_all %>%
  mutate(
    ConditionNorm  = vapply(condition, normalize_pred_cond, character(1)),
    ConditionLabel = paste0("(", ConditionNorm, ")"),
    # pred_rt50 is in seconds -> convert to ms
    PredMedianRT   = pred_rt50 * 1000,
    ConditionGroup = factor(
      vapply(ConditionNorm, assign_group, character(1)),
      levels = condition_group_order
    )
  ) %>%
  filter(!is.na(ConditionGroup), !is.na(PredMedianRT)) %>%
  rename(SubjectNumInt = subject)

# ---- auto grid layout ----
all_facets <- sort(unique(obs_summary$SubjectFacet))
n_subs     <- length(all_facets)
# choose ncol so subplot is roughly square (each sub panel = 3 condition rows tall)
facet_ncol <- if (n_subs <= 3) n_subs else if (n_subs <= 6) 3L else 4L
facet_nrow <- ceiling(n_subs / facet_ncol)

# ---- per-subject mini-plot (3 condition-group rows) ----
plot_base_size  <- 12
strip_text_size <- 11
axis_text_size  <- 9
line_width_obs  <- 0.5
point_size_obs  <- 3.0    # observed dots bigger for visibility
line_width_pred <- 1.1    # prediction horizontal line
pred_alpha      <- 0.9

y_lim <- range(c(obs_summary$MedianRT, pred_summary$PredMedianRT), na.rm = TRUE)
y_pad <- diff(y_lim) * 0.05
y_lim <- y_lim + c(-y_pad, y_pad)

extract_legend_grob <- function(p) {
  gt  <- ggplotGrob(p)
  idx <- which(gt$layout$name == "guide-box")
  if (length(idx) == 0) return(grid::nullGrob())
  gt$grobs[[idx]]
}

make_sub_plot <- function(subject_facet, show_x_title = FALSE, show_y_title = FALSE,
                          show_legend = FALSE) {
  sid <- as.integer(sub("^sub(\\d+)_.*$", "\\1", subject_facet))

  obs_s  <- obs_summary %>% filter(SubjectFacet == subject_facet)
  pred_s <- pred_summary %>% filter(SubjectNumInt == sid)

  if (nrow(obs_s) == 0) return(ggplot() + theme_void())

  vlines_s <- day_vlines %>%
    filter(SubjectFacet == subject_facet) %>%
    tidyr::crossing(ConditionGroup = factor(condition_group_order, levels = condition_group_order))

  # Prediction: flat line spanning full session range
  pred_flat <- pred_s %>%
    filter(!is.na(PredMedianRT)) %>%
    mutate(xmin = 1L, xmax = length(session_ids))

  p <- ggplot(obs_s, aes(x = as.integer(Session), y = MedianRT,
                         color = ConditionLabel, group = ConditionLabel)) +
    # day-start vertical lines
    {
      if (nrow(vlines_s) > 0)
        geom_vline(data = vlines_s, aes(xintercept = VlineX),
                   inherit.aes = FALSE, color = "black", linewidth = 0.4)
    } +
    # observed: dots connected by thin line
    geom_line(linewidth = line_width_obs, alpha = 0.6) +
    geom_point(size = point_size_obs, alpha = 0.9) +
    # prediction: bold flat horizontal segment per condition
    {
      if (nrow(pred_flat) > 0)
        geom_segment(
          data = pred_flat,
          aes(x = xmin, xend = xmax, y = PredMedianRT, yend = PredMedianRT,
              color = ConditionLabel),
          inherit.aes = FALSE,
          linewidth = line_width_pred,
          alpha = pred_alpha,
          linetype = "solid"
        )
    } +
    scale_color_manual(values = condition_group_colors, name = "Condition") +
    scale_x_continuous(breaks = seq_along(session_ids), labels = session_ids) +
    coord_cartesian(ylim = y_lim) +
    facet_wrap(~ConditionGroup, ncol = 1, scales = "fixed") +
    labs(
      title = subject_facet,
      x     = if (show_x_title) "Session" else NULL,
      y     = if (show_y_title) "Median RT (ms)" else NULL
    ) +
    theme_minimal(base_size = plot_base_size) +
    theme(
      plot.title    = element_text(size = strip_text_size, face = "bold", hjust = 0.5),
      plot.margin   = margin(2, 4, 2, 4),
      axis.title    = element_text(size = axis_text_size + 1, face = "bold"),
      axis.text     = element_text(size = axis_text_size,     face = "bold"),
      axis.text.x   = element_text(angle = 45, hjust = 1),
      strip.text    = element_text(size = strip_text_size - 1, face = "bold"),
      legend.position  = if (show_legend) "bottom" else "none",
      legend.key.width = unit(0.7, "cm"),
      panel.spacing    = unit(0.35, "lines")
    )
  p
}

# ---- assemble all subjects into grid ----
sub_grobs <- lapply(seq_along(all_facets), function(i) {
  sf      <- all_facets[[i]]
  row_idx <- ceiling(i / facet_ncol)
  col_idx <- (i - 1L) %% facet_ncol + 1L
  p <- make_sub_plot(
    sf,
    show_x_title = (row_idx == facet_nrow),
    show_y_title = (col_idx == 1L)
  )
  ggplotGrob(p)
})

# pad to full grid
n_cells <- facet_nrow * facet_ncol
while (length(sub_grobs) < n_cells) {
  sub_grobs[[length(sub_grobs) + 1L]] <- grid::nullGrob()
}

layout_mat <- matrix(seq_len(n_cells), nrow = facet_nrow, ncol = facet_ncol, byrow = TRUE)

# legend from first subject
legend_p    <- make_sub_plot(all_facets[[1]], show_legend = TRUE)
legend_grob <- extract_legend_grob(legend_p)

ses_tag <- paste0("ses", min(session_ids), "-", max(session_ids))
sub_tag <- if (length(subject_ids) == 1) {
  paste0("sub", subject_ids[[1]])
} else {
  paste0("sub", min(subject_ids), "-", max(subject_ids))
}

title_grob <- grid::textGrob(
  paste0("Median RT by session: observed (dots+line) vs model prediction (flat line)  |  ",
         mode_tag, "  |  ", sub_tag, "_", ses_tag),
  gp = grid::gpar(fontsize = 14, fontface = "bold")
)
subtitle_grob <- grid::textGrob(
  paste0(
    "Dots + thin connecting line = observed median RT per session. ",
    "Bold flat line = model prediction (pooled across sessions). ",
    "Colors: green = adjacent pairs, blue = single cue, orange = non-adjacent pairs. ",
    "Black vertical lines = first session of each calendar day."
  ),
  gp = grid::gpar(fontsize = 10)
)

page_grob <- gridExtra::arrangeGrob(
  grobs         = c(sub_grobs, list(legend_grob)),
  layout_matrix = rbind(layout_mat, rep(n_cells + 1L, facet_ncol)),
  heights       = c(rep(1, facet_nrow), 0.15),
  top           = gridExtra::arrangeGrob(title_grob, subtitle_grob, heights = c(0.55, 0.45))
)

# ---- save ----
out_file <- file.path(fig_dir,
  paste0(sub_tag, "_RTbySession_PredOverlay_", mode_tag, "_", ses_tag, ".png"))

width_in  <- max(16, 5.5 * facet_ncol)
height_in <- max(10, 6.5 * facet_nrow + 2.0)

ggsave(
  filename  = out_file,
  plot      = page_grob,
  width     = width_in,
  height    = height_in,
  dpi       = 300,
  limitsize = FALSE
)
message("Saved: ", out_file)
message("Done.")
