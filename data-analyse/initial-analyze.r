# Load required libraries
library(tidyverse)
library(data.table)

# Set working directory to the data folder
data_folder <- "../data/ParticipantCPP002-003/ParticipantCPP002-003"

# Function to read a single .dat file
read_dat_file <- function(file_path) {
  # Read all lines
  lines <- readLines(file_path)
  
  # Find the header row (starts with "ExperimentName")
  header_idx <- which(grepl("^ExperimentName", lines))
  
  if (length(header_idx) == 0) {
    warning(paste("No header found in", basename(file_path)))
    return(NULL)
  }
  
  # Read data starting from header row
  data <- fread(
    file_path,
    skip = header_idx[1] - 1,
    sep = "\t",
    header = TRUE,
    fill = TRUE,
    na.strings = c("", "NA", "N/A")
  )
  
  # Add filename for tracking
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

# Remove NULL entries
all_data_list <- all_data_list[!sapply(all_data_list, is.null)]

# Combine all dataframes
combined_data <- rbindlist(all_data_list, fill = TRUE)

cat("Combined data has", nrow(combined_data), "rows and", ncol(combined_data), "columns\n")

# Convert RT to numeric (remove any non-numeric values)
combined_data$RT <- as.numeric(as.character(combined_data$RT))

# Remove rows with missing RT
combined_data <- combined_data[!is.na(combined_data$RT), ]

# Remove rows with RT <= 0 or unreasonably high RT (e.g., > 10 seconds)
combined_data <- combined_data[combined_data$RT > 0 & combined_data$RT <= 10000, ]

cat("After filtering, data has", nrow(combined_data), "rows\n")

# Check CueCondition values
cat("\nCueCondition values:\n")
print(table(combined_data$CueCondition, useNA = "always"))

# Create RT distribution plot by CueCondition
p <- ggplot(combined_data, aes(x = RT, fill = as.factor(CueCondition))) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  facet_wrap(~ CueCondition, scales = "free_y") +
  labs(
    title = "RT Distribution by Cue Condition - Participant CPP002-003",
    x = "Reaction Time (ms)",
    y = "Frequency",
    fill = "Cue Condition"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    strip.text = element_text(size = 10, face = "bold")
  )

# Save the plot
ggsave(
  filename = "RT_distribution_by_cuecondition.png",
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

cat("\nPlot saved to: ../data/ParticipantCPP002-003/RT_distribution_by_cuecondition.png\n")

# Also create a density plot
p_density <- ggplot(combined_data, aes(x = RT, color = as.factor(CueCondition))) +
  geom_density(alpha = 0.7, linewidth = 1) +
  labs(
    title = "RT Density Distribution by Cue Condition - Participant CPP002-003",
    x = "Reaction Time (ms)",
    y = "Density",
    color = "Cue Condition"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "right"
  )

ggsave(
  filename = "RT_density_by_cuecondition.png",
  plot = p_density,
  width = 12,
  height = 6,
  dpi = 300
)

cat("Density plot saved to: ../data/ParticipantCPP002-003/RT_density_by_cuecondition.png\n")

# Summary statistics by CueCondition
cat("\nSummary statistics by CueCondition:\n")
summary_stats <- combined_data %>%
  group_by(CueCondition) %>%
  summarise(
    n = n(),
    mean_RT = mean(RT, na.rm = TRUE),
    median_RT = median(RT, na.rm = TRUE),
    sd_RT = sd(RT, na.rm = TRUE),
    min_RT = min(RT, na.rm = TRUE),
    max_RT = max(RT, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_stats)

# Save summary statistics
write.csv(
  summary_stats,
  file = "../data/ParticipantCPP002-003/RT_summary_by_cuecondition.csv",
  row.names = FALSE
)

cat("\nSummary statistics saved to: ../data/ParticipantCPP002-003/RT_summary_by_cuecondition.csv\n")

cat("\nAnalysis complete!\n")

