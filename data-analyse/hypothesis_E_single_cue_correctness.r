# Hypothesis E: Analysis for single-cue conditions (1-4)
# Classify by: Speed + Correctness
# - Correct: Chose the cued location
# - Incorrect: Chose wrong/empty location

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

# Filter to single-cue conditions only (1-4)
single_cue_conditions <- c(1, 2, 3, 4)
combined_data <- combined_data[combined_data$CueCondition %in% single_cue_conditions, ]

cat("After filtering to single-cue conditions (1-4), data has", nrow(combined_data), "rows\n")

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

# Get response value and max reward
combined_data$CueResponseValue_num <- as.numeric(as.character(combined_data$CueResponseValue))
combined_data$MaxReward <- sapply(combined_data$ParsedCueValues, function(x) {
  if (any(is.na(x)) || length(x) == 0) return(NA)
  max(x, na.rm = TRUE)
})

# Check if participant chose correctly (the cued location)
# In single-cue conditions, correct = chose the location with the cue (MaxReward)
combined_data$ChoseCorrect <- combined_data$CueResponseValue_num == combined_data$MaxReward

# Classify trials based on RT speed and correctness
combined_data$TrialType <- ifelse(
  combined_data$RT_sec < 0.25,
  ifelse(combined_data$ChoseCorrect, "Fast Correct", "Fast Incorrect"),
  ifelse(
    combined_data$RT_sec < 0.4,
    ifelse(combined_data$ChoseCorrect, "Medium Correct", "Medium Incorrect"),
    ifelse(combined_data$ChoseCorrect, "Slow Correct", "Slow Incorrect")
  )
)

cat("\nTrial classification by RT and correctness:\n")
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
    title = "Hypothesis E: RT Density by Speed + Correctness (Single-Cue Conditions 1-4)",
    subtitle = "Weighted densities sum to overall (black line) - Correct vs Incorrect",
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
  filename = file.path(output_dir, "hypothesis_E_overall.png"),
  plot = p_overall,
  width = 14,
  height = 8,
  dpi = 300,
  bg = "white"
)

# ============================================================================
# Condition-specific plots: Conditions 1, 2, 3, 4
# ============================================================================
target_conditions <- c(1, 2, 3, 4)
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
            subtitle = "Weighted densities: Correct vs Incorrect",
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
    combined_plot_trialtypes <- do.call(grid.arrange, c(plot_list_trialtypes, ncol = 2, nrow = 2))
    
    ggsave(
      filename = file.path(output_dir, "hypothesis_E_by_condition.png"),
      plot = combined_plot_trialtypes,
      width = 16,
      height = 12,
      dpi = 300,
      bg = "white"
    )
    cat("Saved: hypothesis_E_by_condition.png\n")
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
    correct_rate = mean(ChoseCorrect, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

# Save summary statistics
write.csv(
  summary_stats,
  file = file.path(output_dir, "hypothesis_E_summary.csv"),
  row.names = FALSE
)

cat("\n=== Hypothesis E Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")


