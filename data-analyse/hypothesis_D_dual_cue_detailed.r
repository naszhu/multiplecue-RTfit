# Hypothesis D: Detailed analysis for dual-cue conditions (5-10)
# Classify by: Speed + Optimality + Mismatch status
# - Optimal: Chose max reward
# - Second Best: Chose second highest reward
# - Mismatch: Chose location with no cue (0 reward)

library(tidyverse)
library(data.table)
library(gridExtra)

# Set working directory to the data folder
data_folder <- "../data/ParticipantCPP002-003/ParticipantCPP002-003"

# Function to read a single .dat file
read_dat_file <- function(file_path) {
  lines <- readLines(file_path)
  header_idx <- which(grepl("^ExperimentName", lines))
  
  if (length(header_idx) == 0) {
    warning(paste("No header found in", basename(file_path)))
    return(NULL)
  }
  
  data <- fread(
    file_path,
    skip = header_idx[1] - 1,
    sep = "\t",
    header = TRUE,
    fill = TRUE,
    na.strings = c("", "NA", "N/A")
  )
  
  data$filename <- basename(file_path)
  return(data)
}

# Get all .dat files
dat_files <- list.files(
  path = data_folder,
  pattern = "\\.dat$",
  full.names = TRUE
)

cat("Found", length(dat_files), "data files\n")

# Read all files and combine
all_data_list <- lapply(dat_files, read_dat_file)
all_data_list <- all_data_list[!sapply(all_data_list, is.null)]
combined_data <- rbindlist(all_data_list, fill = TRUE)

# Convert RT to numeric and filter
combined_data$RT <- as.numeric(as.character(combined_data$RT))
combined_data <- combined_data[!is.na(combined_data$RT), ]
combined_data <- combined_data[combined_data$RT > 0 & combined_data$RT <= 10, ]

# RT is already in seconds
combined_data$RT_sec <- combined_data$RT

cat("After filtering, data has", nrow(combined_data), "rows\n")

# Filter to dual-cue conditions only (5-10)
dual_cue_conditions <- c(5, 6, 7, 8, 9, 10)
combined_data <- combined_data[combined_data$CueCondition %in% dual_cue_conditions, ]

cat("After filtering to dual-cue conditions (5-10), data has", nrow(combined_data), "rows\n")

# Parse CueValues
parse_cue_values <- function(cue_values_str) {
  if (length(cue_values_str) == 0 || is.na(cue_values_str) || cue_values_str == "") return(NA)
  clean_str <- gsub("[\\[\\]\\s]", "", as.character(cue_values_str))
  if (clean_str == "" || is.na(clean_str)) return(NA)
  if (grepl(",", clean_str)) {
    values <- as.numeric(strsplit(clean_str, ",")[[1]])
  } else {
    values <- as.numeric(strsplit(clean_str, "")[[1]])
  }
  values <- values[!is.na(values)]
  if (length(values) == 0) return(NA)
  return(values)
}

combined_data$ParsedCueValues <- lapply(combined_data$CueValues, parse_cue_values)

# Get response value and reward structure
combined_data$CueResponseValue_num <- as.numeric(as.character(combined_data$CueResponseValue))

# Calculate max and second max rewards
combined_data$MaxReward <- sapply(combined_data$ParsedCueValues, function(x) {
  if (any(is.na(x)) || length(x) == 0) return(NA)
  max(x, na.rm = TRUE)
})

combined_data$SecondMaxReward <- sapply(combined_data$ParsedCueValues, function(x) {
  if (any(is.na(x)) || length(x) < 2) return(NA)
  sorted <- sort(x, decreasing = TRUE)
  sorted[2]
})

# Check if response value is in the cue values (not a mismatch)
combined_data$ResponseInCues <- sapply(1:nrow(combined_data), function(i) {
  if (is.na(combined_data$CueResponseValue_num[i])) return(FALSE)
  cue_vals <- combined_data$ParsedCueValues[[i]]
  if (is.null(cue_vals) || length(cue_vals) == 0 || any(is.na(cue_vals))) return(FALSE)
  combined_data$CueResponseValue_num[i] %in% cue_vals
})

# Classify choice type: Optimal, Second Best, or Mismatch (no cue)
combined_data$ChoiceType <- ifelse(
  is.na(combined_data$CueResponseValue_num) | combined_data$CueResponseValue_num == 0,
  "Mismatch (No Cue)",
  ifelse(
    !combined_data$ResponseInCues,
    "Mismatch (No Cue)",
    ifelse(
      combined_data$CueResponseValue_num == combined_data$MaxReward,
      "Optimal",
      ifelse(
        !is.na(combined_data$SecondMaxReward) & 
        combined_data$CueResponseValue_num == combined_data$SecondMaxReward,
        "Second Best",
        "Other Non-Optimal"  # Chose a cue but not max or second max (shouldn't happen in dual-cue, but just in case)
      )
    )
  )
)

# Classify trials based on RT speed, optimality, and mismatch
combined_data$TrialType <- ifelse(
  combined_data$RT_sec < 0.25,
  paste("Fast", combined_data$ChoiceType),
  ifelse(
    combined_data$RT_sec < 0.4,
    paste("Medium", combined_data$ChoiceType),
    paste("Slow", combined_data$ChoiceType)
  )
)

cat("\nTrial classification by RT, optimality, and mismatch:\n")
print(table(combined_data$TrialType, useNA = "always"))

# Create plots
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Overall plot: RT density by trial type (with weighted densities)
# ============================================================================
data_for_plot <- combined_data[!is.na(combined_data$TrialType), ]

# Calculate proportions for each trial type
trial_type_counts <- table(data_for_plot$TrialType)
total_trials <- sum(trial_type_counts)
trial_type_props <- trial_type_counts / total_trials

# Calculate overall density
overall_density <- density(data_for_plot$RT_sec, na.rm = TRUE)
overall_df <- data.frame(x = overall_density$x, y = overall_density$y)

# Calculate weighted densities for each trial type
weighted_densities <- list()
for (tt in names(trial_type_props)) {
  tt_data <- data_for_plot[data_for_plot$TrialType == tt, ]
  if (nrow(tt_data) > 1) {
    tt_density <- density(tt_data$RT_sec, na.rm = TRUE)
    # Weight by proportion
    weighted_densities[[tt]] <- data.frame(
      x = tt_density$x,
      y = tt_density$y * trial_type_props[tt],
      TrialType = tt
    )
  }
}

# Combine all weighted densities
weighted_df <- do.call(rbind, weighted_densities)

p_overall <- ggplot() +
  # Overall density line (black, thicker) - drawn first as base layer
  geom_line(data = overall_df, aes(x = x, y = y), 
           color = "black", linewidth = 2.5, alpha = 0.9, linetype = "solid") +
  # Weighted trial type-specific density lines - drawn on top
  geom_line(data = weighted_df, aes(x = x, y = y, color = TrialType), 
           alpha = 0.7, linewidth = 1.2) +
  labs(
    title = "Hypothesis D: RT Density by Speed + Optimality + Mismatch (Dual-Cue Conditions 5-10)",
    subtitle = "Weighted densities sum to overall (black line) - Optimal vs Second Best vs Mismatch",
    x = "Reaction Time (seconds)",
    y = "Density",
    color = "Trial Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  geom_vline(xintercept = c(0.25, 0.4), 
             linetype = "dashed", alpha = 0.5, color = "gray")

ggsave(
  filename = file.path(output_dir, "hypothesis_D_overall.png"),
  plot = p_overall,
  width = 14,
  height = 8,
  dpi = 300,
  bg = "white"
)

# ============================================================================
# Condition-specific plots: Conditions 5, 6, 7, 8, 9, 10
# ============================================================================
target_conditions <- c(5, 6, 7, 8, 9, 10)
combined_data_conditions <- combined_data[
  combined_data$CueCondition %in% target_conditions & 
  !is.na(combined_data$TrialType),
]

cat("\n=== Creating condition-specific plots for conditions:", paste(target_conditions, collapse = ", "), "===\n")
cat("Number of trials:", nrow(combined_data_conditions), "\n")

if (nrow(combined_data_conditions) > 0) {
  
  plot_list_trialtypes <- list()
  
  for (cond in target_conditions) {
    data_cond <- combined_data_conditions[combined_data_conditions$CueCondition == cond, ]
    
    if (nrow(data_cond) > 0) {
      # Calculate proportions for each trial type
      trial_type_counts_cond <- table(data_cond$TrialType)
      total_trials_cond <- sum(trial_type_counts_cond)
      trial_type_props_cond <- trial_type_counts_cond / total_trials_cond
      
      # Calculate overall density
      overall_density_cond <- density(data_cond$RT_sec, na.rm = TRUE)
      overall_df_cond <- data.frame(x = overall_density_cond$x, y = overall_density_cond$y)
      
      # Calculate weighted densities for each trial type
      weighted_densities_cond <- list()
      for (tt in names(trial_type_props_cond)) {
        tt_data <- data_cond[data_cond$TrialType == tt, ]
        if (nrow(tt_data) > 1) {
          tt_density <- density(tt_data$RT_sec, na.rm = TRUE)
          # Weight by proportion
          weighted_densities_cond[[tt]] <- data.frame(
            x = tt_density$x,
            y = tt_density$y * trial_type_props_cond[tt],
            TrialType = tt
          )
        }
      }
      
      # Combine all weighted densities
      if (length(weighted_densities_cond) > 0) {
        weighted_df_cond <- do.call(rbind, weighted_densities_cond)
        
        p <- ggplot() +
          # Overall density line (black, thicker) - drawn first as base layer
          geom_line(data = overall_df_cond, aes(x = x, y = y), 
                   color = "black", linewidth = 2.5, alpha = 0.9, linetype = "solid") +
          # Weighted trial type-specific density lines - drawn on top
          geom_line(data = weighted_df_cond, aes(x = x, y = y, color = TrialType), 
                   alpha = 0.7, linewidth = 1.2) +
          labs(
            title = paste("Condition", cond, "- Overall (black) vs Trial Types"),
            subtitle = "Weighted densities: Optimal vs Second Best vs Mismatch",
            x = "Reaction Time (seconds)",
            y = "Density",
            color = "Trial Type"
          ) +
          theme_minimal() +
          theme(
            plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 10),
            legend.position = "bottom",
            legend.text = element_text(size = 8),
            plot.background = element_rect(fill = "white", color = NA),
            panel.background = element_rect(fill = "white", color = NA)
          ) +
          geom_vline(xintercept = c(0.25, 0.4), 
                     linetype = "dashed", alpha = 0.5, color = "gray")
        
        plot_list_trialtypes[[as.character(cond)]] <- p
      }
    }
  }
  
  if (length(plot_list_trialtypes) > 0) {
    combined_plot_trialtypes <- do.call(grid.arrange, c(plot_list_trialtypes, ncol = 3, nrow = 2))
    
    ggsave(
      filename = file.path(output_dir, "hypothesis_D_by_condition.png"),
      plot = combined_plot_trialtypes,
      width = 20,
      height = 14,
      dpi = 300,
      bg = "white"
    )
    cat("Saved: hypothesis_D_by_condition.png\n")
  }
}

# Summary statistics
cat("\n=== Summary Statistics by Trial Type ===\n")
summary_stats <- combined_data[!is.na(combined_data$TrialType), ] %>%
  group_by(TrialType) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    median_RT = median(RT_sec, na.rm = TRUE),
    sd_RT = sd(RT_sec, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

# Save summary statistics
write.csv(
  summary_stats,
  file = file.path(output_dir, "hypothesis_D_summary.csv"),
  row.names = FALSE
)

cat("\n=== Hypothesis D Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")

