# Hypothesis C: Eye-Tracking Artifacts (Re-fixations)
# Test if slow peak comes from trials where eye started to move, missed target,
# and then corrected to land on correct target

library(tidyverse)
library(data.table)

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

# Extract eye tracking data
# We need to check if there are multiple eye movements before landing on target
# EyeT1, EyeT2, ... EyeT36 are timestamps
# EyeX1, EyeX2, ... EyeX36 are x-coordinates
# EyeY1, EyeY2, ... EyeY36 are y-coordinates

# Function to extract eye movement sequence
extract_eye_sequence <- function(row) {
  # Get all eye tracking samples
  eye_times <- c()
  eye_x <- c()
  eye_y <- c()
  
  for (i in 1:36) {
    time_col <- paste0("EyeT", i)
    x_col <- paste0("EyeX", i)
    y_col <- paste0("EyeY", i)
    
    if (time_col %in% names(row)) {
      time_val <- as.numeric(as.character(row[[time_col]]))
      x_val <- as.numeric(as.character(row[[x_col]]))
      y_val <- as.numeric(as.character(row[[y_col]]))
      
      if (!is.na(time_val) && !is.na(x_val) && !is.na(y_val)) {
        eye_times <- c(eye_times, time_val)
        eye_x <- c(eye_x, x_val)
        eye_y <- c(eye_y, y_val)
      }
    }
  }
  
  return(list(times = eye_times, x = eye_x, y = eye_y))
}

# Get CueTime and PointTargetTime for reference
combined_data$CueTime_num <- as.numeric(as.character(combined_data$CueTime))
combined_data$PointTargetTime_num <- as.numeric(as.character(combined_data$PointTargetTime))
combined_data$PointTargetResponse_num <- as.numeric(as.character(combined_data$PointTargetResponse))

# Calculate number of eye samples
combined_data$NumberEyeSamples_num <- as.numeric(as.character(combined_data$NumberEyeSamples))

# Simple heuristic: If there are many eye samples relative to RT, might indicate multiple movements
# Or if RT is long but number of samples is high, might indicate corrections

# Calculate eye movement metrics
combined_data$EyeSamplesPerSecond <- ifelse(
  combined_data$RT_sec > 0,
  combined_data$NumberEyeSamples_num / combined_data$RT_sec,
  NA
)

# Classify based on RT and eye sample density
# High sample density with long RT might indicate corrections
combined_data$TrialType <- ifelse(
  combined_data$RT_sec < 0.25,
  "Fast (Single Movement?)",
  ifelse(
    combined_data$RT_sec < 0.4,
    "Medium",
    ifelse(
      !is.na(combined_data$EyeSamplesPerSecond) & combined_data$EyeSamplesPerSecond > 50,
      "Slow with High Eye Activity (Possible Correction)",
      "Slow (Other)"
    )
  )
)

cat("\nTrial classification by RT and eye activity:\n")
print(table(combined_data$TrialType, useNA = "always"))

# Alternative: Look at eye sample count directly
combined_data$TrialTypeBySamples <- ifelse(
  combined_data$RT_sec < 0.25,
  "Fast",
  ifelse(
    !is.na(combined_data$NumberEyeSamples_num) & combined_data$NumberEyeSamples_num > 20,
    "Slow with Many Samples (Possible Correction)",
    "Slow with Few Samples"
  )
)

cat("\nTrial classification by RT and sample count:\n")
print(table(combined_data$TrialTypeBySamples, useNA = "always"))

# Create plots
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Plot 1: RT distribution by eye activity type
p1 <- ggplot(combined_data[!is.na(combined_data$TrialType), ], 
             aes(x = RT_sec, fill = TrialType)) +
  geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
  facet_wrap(~ TrialType, ncol = 1, scales = "free_y") +
  labs(
    title = "Hypothesis C: RT Distribution by Eye Activity Type",
    subtitle = "Slow RTs with high eye activity might indicate re-fixations/corrections",
    x = "Reaction Time (seconds)",
    y = "Frequency",
    fill = "Trial Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(size = 10, face = "bold")
  ) +
  geom_vline(xintercept = c(0.25, 0.4, 0.6), 
             linetype = "dashed", alpha = 0.3, color = "red")

ggsave(
  filename = file.path(output_dir, "hypothesis_C_eye_activity.png"),
  plot = p1,
  width = 12,
  height = 10,
  dpi = 300
)

# Plot 2: RT vs Number of Eye Samples
p2 <- ggplot(combined_data[!is.na(combined_data$NumberEyeSamples_num), ], 
             aes(x = RT_sec, y = NumberEyeSamples_num)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 1.5) +
  labs(
    title = "Hypothesis C: RT vs Number of Eye Samples",
    subtitle = "Re-fixations should show more eye samples for given RT",
    x = "Reaction Time (seconds)",
    y = "Number of Eye Samples"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave(
  filename = file.path(output_dir, "hypothesis_C_RT_vs_samples.png"),
  plot = p2,
  width = 10,
  height = 6,
  dpi = 300
)

# Plot 3: Eye samples per second vs RT
p3 <- ggplot(combined_data[!is.na(combined_data$EyeSamplesPerSecond) & 
                           combined_data$EyeSamplesPerSecond < 200, ], 
             aes(x = RT_sec, y = EyeSamplesPerSecond)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 1.5) +
  labs(
    title = "Hypothesis C: Eye Sample Density vs RT",
    subtitle = "High density with long RT might indicate corrections",
    x = "Reaction Time (seconds)",
    y = "Eye Samples per Second"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave(
  filename = file.path(output_dir, "hypothesis_C_sample_density.png"),
  plot = p3,
  width = 10,
  height = 6,
  dpi = 300
)

# Plot 4: RT distribution by CueCondition, colored by eye activity
# Focus on bimodal conditions
bimodal_conditions <- c(8, 9, 10)
combined_data_subset <- combined_data[
  combined_data$CueCondition %in% bimodal_conditions & 
  !is.na(combined_data$TrialTypeBySamples),
]

if (nrow(combined_data_subset) > 0) {
  p4 <- ggplot(combined_data_subset, 
               aes(x = RT_sec, fill = TrialTypeBySamples)) +
    geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
    facet_wrap(~ CueCondition, scales = "free_y") +
    labs(
      title = "Hypothesis C: RT Distribution by CueCondition (Bimodal Conditions)",
      subtitle = "Colored by eye sample count (indicator of re-fixations)",
      x = "Reaction Time (seconds)",
      y = "Frequency",
      fill = "Eye Activity"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      strip.text = element_text(size = 10, face = "bold")
    )
  
  ggsave(
    filename = file.path(output_dir, "hypothesis_C_bimodal_conditions.png"),
    plot = p4,
    width = 14,
    height = 6,
    dpi = 300
  )
}

# Plot 5: Density plot comparing fast vs slow with high eye activity
p5 <- ggplot(combined_data[!is.na(combined_data$TrialTypeBySamples), ], 
             aes(x = RT_sec, color = TrialTypeBySamples)) +
  geom_density(alpha = 0.7, linewidth = 1.5) +
  labs(
    title = "Hypothesis C: RT Density by Eye Sample Count",
    subtitle = "Slow trials with many samples might indicate re-fixations",
    x = "Reaction Time (seconds)",
    y = "Density",
    color = "Eye Activity"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "right"
  ) +
  geom_vline(xintercept = c(0.25, 0.4, 0.6), 
             linetype = "dashed", alpha = 0.3, color = "gray")

ggsave(
  filename = file.path(output_dir, "hypothesis_C_density_comparison.png"),
  plot = p5,
  width = 12,
  height = 6,
  dpi = 300
)

# Summary statistics
cat("\n=== Summary Statistics by Trial Type (Eye Activity) ===\n")
summary_stats <- combined_data[!is.na(combined_data$TrialType), ] %>%
  group_by(TrialType) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    median_RT = median(RT_sec, na.rm = TRUE),
    sd_RT = sd(RT_sec, na.rm = TRUE),
    mean_samples = mean(NumberEyeSamples_num, na.rm = TRUE),
    mean_samples_per_sec = mean(EyeSamplesPerSecond, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

cat("\n=== Summary Statistics by Sample Count ===\n")
summary_samples <- combined_data[!is.na(combined_data$TrialTypeBySamples), ] %>%
  group_by(TrialTypeBySamples) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    median_RT = median(RT_sec, na.rm = TRUE),
    sd_RT = sd(RT_sec, na.rm = TRUE),
    mean_samples = mean(NumberEyeSamples_num, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_samples)

# Save summary statistics
write.csv(
  summary_stats,
  file = file.path(output_dir, "hypothesis_C_summary_activity.csv"),
  row.names = FALSE
)

write.csv(
  summary_samples,
  file = file.path(output_dir, "hypothesis_C_summary_samples.csv"),
  row.names = FALSE
)

cat("\n=== Hypothesis C Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")

