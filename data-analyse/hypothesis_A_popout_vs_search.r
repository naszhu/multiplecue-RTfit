# Hypothesis A: "Pop-out" vs. "Search"
# Test if fast peak corresponds to single high-value cue (pop-out)
# and slow peak corresponds to competing high-value cues (search)

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

# Parse CueValues string (format: "[1,2,3,4]" or "1234" or similar)
parse_cue_values <- function(cue_values_str) {
  if (length(cue_values_str) == 0 || is.na(cue_values_str) || cue_values_str == "") return(NA)
  
  # Remove brackets and whitespace
  clean_str <- gsub("[\\[\\]\\s]", "", as.character(cue_values_str))
  
  if (clean_str == "" || is.na(clean_str)) return(NA)
  
  # Check if comma-separated
  if (grepl(",", clean_str)) {
    values <- as.numeric(strsplit(clean_str, ",")[[1]])
  } else {
    # Parse as individual digits
    values <- as.numeric(strsplit(clean_str, "")[[1]])
  }
  
  values <- values[!is.na(values)]
  if (length(values) == 0) return(NA)
  return(values)
}

# Parse CueValues for each row
combined_data$ParsedCueValues <- lapply(combined_data$CueValues, parse_cue_values)

# Calculate metrics for each trial
combined_data$MaxReward <- sapply(combined_data$ParsedCueValues, function(x) {
  if (any(is.na(x)) || length(x) == 0) return(NA)
  max(x, na.rm = TRUE)
})

combined_data$SecondMaxReward <- sapply(combined_data$ParsedCueValues, function(x) {
  if (any(is.na(x)) || length(x) < 2) return(NA)
  sorted <- sort(x, decreasing = TRUE)
  sorted[2]
})

# Calculate reward difference (gap between max and second max)
combined_data$RewardGap <- combined_data$MaxReward - combined_data$SecondMaxReward

# Classify trials: "Pop-out" (large gap) vs "Search" (small gap)
# Using median split or a threshold
median_gap <- median(combined_data$RewardGap, na.rm = TRUE)
combined_data$TrialType <- ifelse(
  combined_data$RewardGap >= median_gap,
  "Pop-out (Large Gap)",
  "Search (Competing Cues)"
)

# Alternative: Use a more strict threshold (e.g., gap >= 2)
combined_data$TrialTypeStrict <- ifelse(
  combined_data$RewardGap >= 2,
  "Pop-out (Gap >= 2)",
  ifelse(
    combined_data$RewardGap >= 1,
    "Moderate Competition",
    "High Competition (Gap < 1)"
  )
)

cat("\nTrial classification (median split):\n")
print(table(combined_data$TrialType, useNA = "always"))

cat("\nTrial classification (strict):\n")
print(table(combined_data$TrialTypeStrict, useNA = "always"))

# Create plots
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Plot 1: RT distribution by trial type (median split)
p1 <- ggplot(combined_data[!is.na(combined_data$TrialType), ], 
             aes(x = RT_sec, fill = TrialType)) +
  geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
  facet_wrap(~ TrialType, ncol = 1, scales = "free_y") +
  labs(
    title = "Hypothesis A: RT Distribution by Pop-out vs Search (Median Split)",
    subtitle = "Pop-out = Large reward gap, Search = Competing high-value cues",
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
  geom_vline(xintercept = c(0.2, 0.3, 0.4, 0.6), 
             linetype = "dashed", alpha = 0.3, color = "red")

ggsave(
  filename = file.path(output_dir, "hypothesis_A_median_split.png"),
  plot = p1,
  width = 12,
  height = 8,
  dpi = 300
)

# Plot 2: RT density by trial type (strict classification)
p2 <- ggplot(combined_data[!is.na(combined_data$TrialTypeStrict), ], 
             aes(x = RT_sec, color = TrialTypeStrict)) +
  geom_density(alpha = 0.7, linewidth = 1.5) +
  labs(
    title = "Hypothesis A: RT Density by Competition Level",
    subtitle = "Strict classification based on reward gap",
    x = "Reaction Time (seconds)",
    y = "Density",
    color = "Competition Level"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "right"
  ) +
  geom_vline(xintercept = c(0.2, 0.3, 0.4, 0.6), 
             linetype = "dashed", alpha = 0.3, color = "red")

ggsave(
  filename = file.path(output_dir, "hypothesis_A_strict_classification.png"),
  plot = p2,
  width = 12,
  height = 6,
  dpi = 300
)

# Plot 3: RT vs Reward Gap scatter plot
p3 <- ggplot(combined_data[!is.na(combined_data$RewardGap), ], 
             aes(x = RewardGap, y = RT_sec)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 1.5) +
  labs(
    title = "Hypothesis A: RT vs Reward Gap",
    subtitle = "Larger gap (pop-out) should show faster RTs",
    x = "Reward Gap (Max - Second Max)",
    y = "Reaction Time (seconds)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  )

ggsave(
  filename = file.path(output_dir, "hypothesis_A_RT_vs_gap.png"),
  plot = p3,
  width = 10,
  height = 6,
  dpi = 300
)

# Plot 4: RT distribution by CueCondition, colored by trial type
# Focus on conditions that show bimodality
bimodal_conditions <- c(8, 9, 10)  # Based on summary stats showing higher means
combined_data_subset <- combined_data[
  combined_data$CueCondition %in% bimodal_conditions & 
  !is.na(combined_data$TrialTypeStrict),
]

if (nrow(combined_data_subset) > 0) {
  p4 <- ggplot(combined_data_subset, 
               aes(x = RT_sec, fill = TrialTypeStrict)) +
    geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
    facet_wrap(~ CueCondition, scales = "free_y") +
    labs(
      title = "Hypothesis A: RT Distribution by CueCondition (Bimodal Conditions)",
      subtitle = "Colored by competition level",
      x = "Reaction Time (seconds)",
      y = "Frequency",
      fill = "Competition Level"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      strip.text = element_text(size = 10, face = "bold")
    )
  
  ggsave(
    filename = file.path(output_dir, "hypothesis_A_bimodal_conditions.png"),
    plot = p4,
    width = 14,
    height = 6,
    dpi = 300
  )
}

# Summary statistics
cat("\n=== Summary Statistics by Trial Type (Median Split) ===\n")
summary_median <- combined_data[!is.na(combined_data$TrialType), ] %>%
  group_by(TrialType) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    median_RT = median(RT_sec, na.rm = TRUE),
    sd_RT = sd(RT_sec, na.rm = TRUE),
    q25 = quantile(RT_sec, 0.25, na.rm = TRUE),
    q75 = quantile(RT_sec, 0.75, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_median)

cat("\n=== Summary Statistics by Competition Level (Strict) ===\n")
summary_strict <- combined_data[!is.na(combined_data$TrialTypeStrict), ] %>%
  group_by(TrialTypeStrict) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    median_RT = median(RT_sec, na.rm = TRUE),
    sd_RT = sd(RT_sec, na.rm = TRUE),
    q25 = quantile(RT_sec, 0.25, na.rm = TRUE),
    q75 = quantile(RT_sec, 0.75, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_strict)

# Save summary statistics
write.csv(
  summary_median,
  file = file.path(output_dir, "hypothesis_A_summary_median.csv"),
  row.names = FALSE
)

write.csv(
  summary_strict,
  file = file.path(output_dir, "hypothesis_A_summary_strict.csv"),
  row.names = FALSE
)

cat("\n=== Hypothesis A Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")


