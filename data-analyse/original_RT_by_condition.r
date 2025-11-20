# Original RT Distribution by Cue Condition
# Plot RT distribution for each cue condition using the original RT data
# Similar to Hypothesis C2 faceted plot but with original RT

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

# Filter to valid data with cue conditions
valid_data <- combined_data[
  !is.na(combined_data$RT_sec) & 
  !is.na(combined_data$CueCondition) &
  is.finite(combined_data$RT_sec),
]

cat("Valid data with cue conditions:", nrow(valid_data), "trials\n")

# Create plots
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Convert CueCondition to factor for better plotting
valid_data$CueCondition_fac <- as.factor(valid_data$CueCondition)

# Plot: RT distribution by cue condition (facetted)
p_faceted <- ggplot(valid_data, aes(x = RT_sec)) +
  geom_density(linewidth = 1.2, fill = "steelblue", alpha = 0.5) +
  facet_wrap(~ CueCondition_fac, scales = "free_y") +
  labs(
    title = "Original RT Distribution by Cue Condition (Individual)",
    subtitle = "Each panel shows one cue condition - Check for bimodality",
    x = "Reaction Time (seconds)",
    y = "Density"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  geom_vline(xintercept = c(0.25, 0.4), 
             linetype = "dashed", alpha = 0.5, color = "gray")

ggsave(
  filename = file.path(output_dir, "original_RT_by_condition_faceted.png"),
  plot = p_faceted,
  width = 16,
  height = 12,
  dpi = 300,
  bg = "white"
)

# Plot: RT distribution by cue condition (overlaid)
p_overlaid <- ggplot(valid_data, aes(x = RT_sec, color = CueCondition_fac)) +
  geom_density(linewidth = 1.2, alpha = 0.7) +
  labs(
    title = "Original RT Distribution by Cue Condition",
    subtitle = "All conditions overlaid - Testing for bimodality",
    x = "Reaction Time (seconds)",
    y = "Density",
    color = "Cue Condition"
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
  filename = file.path(output_dir, "original_RT_by_condition.png"),
  plot = p_overlaid,
  width = 14,
  height = 8,
  dpi = 300,
  bg = "white"
)

# Summary statistics
cat("\n=== Summary Statistics by Cue Condition (Original RT) ===\n")
summary_stats <- valid_data %>%
  group_by(CueCondition) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    median_RT = median(RT_sec, na.rm = TRUE),
    sd_RT = sd(RT_sec, na.rm = TRUE),
    min_RT = min(RT_sec, na.rm = TRUE),
    max_RT = max(RT_sec, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

# Save summary statistics
write.csv(
  summary_stats,
  file = file.path(output_dir, "original_RT_summary.csv"),
  row.names = FALSE
)

cat("\n=== Original RT Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")

