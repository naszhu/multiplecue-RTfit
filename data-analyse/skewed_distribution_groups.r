# Simulate Skewed Distribution and Plot Density by Groups
# This script simulates a skewed distribution, separates values into three groups
# (big, medium, smaller), and plots overlapping density curves

library(tidyverse)

# Set seed for reproducibility
set.seed(123)

# Simulate a skewed distribution (using gamma distribution for right-skewed data)
# This is similar to reaction time data which is typically right-skewed
n <- 10000
skewed_data <- rgamma(n, shape = 2, rate = 0.5)  # Creates right-skewed distribution

# Create a data frame with the original data
data_df <- data.frame(
  value = skewed_data,
  group = "Original"
)

# Separate values into three groups: big, medium, smaller
# Using tertiles (33rd and 67th percentiles)
quantiles <- quantile(skewed_data, probs = c(0.33, 0.67))

# Assign groups based on tertiles
data_df$group <- case_when(
  skewed_data <= quantiles[1] ~ "Smaller",
  skewed_data > quantiles[1] & skewed_data <= quantiles[2] ~ "Medium",
  skewed_data > quantiles[2] ~ "Big"
)

# Create separate data frames for each group
smaller_group <- data.frame(
  value = skewed_data[skewed_data <= quantiles[1]],
  group = "Smaller"
)

medium_group <- data.frame(
  value = skewed_data[skewed_data > quantiles[1] & skewed_data <= quantiles[2]],
  group = "Medium"
)

big_group <- data.frame(
  value = skewed_data[skewed_data > quantiles[2]],
  group = "Big"
)

# Combine all groups for plotting
plot_data <- rbind(
  data.frame(value = skewed_data, group = "Original"),
  smaller_group,
  medium_group,
  big_group
)

# Create the overlapping density plot
p <- ggplot(plot_data, aes(x = value, color = group, fill = group)) +
  geom_density(alpha = 0.3, linewidth = 1.2) +
  labs(
    title = "Density Plot: Skewed Distribution and Separated Groups",
    subtitle = "Overlapping density curves for Original data and three groups (Big, Medium, Smaller)",
    x = "Value",
    y = "Density",
    color = "Group",
    fill = "Group"
  ) +
  scale_color_manual(
    values = c(
      "Original" = "black",
      "Big" = "red",
      "Medium" = "blue",
      "Smaller" = "green"
    )
  ) +
  scale_fill_manual(
    values = c(
      "Original" = "black",
      "Big" = "red",
      "Medium" = "blue",
      "Smaller" = "green"
    )
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )

# Create output directory if it doesn't exist
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Save the plot using ggsave
ggsave(
  filename = file.path(output_dir, "skewed_distribution_groups_overlay.png"),
  plot = p,
  width = 14,
  height = 8,
  dpi = 300,
  bg = "white"
)

# Print summary statistics
cat("\n=== Summary Statistics ===\n")
cat("Original Data:\n")
cat("  Mean:", mean(skewed_data), "\n")
cat("  Median:", median(skewed_data), "\n")
cat("  SD:", sd(skewed_data), "\n")
cat("  Min:", min(skewed_data), "\n")
cat("  Max:", max(skewed_data), "\n")
cat("  Quantiles (33%, 67%):", quantiles[1], ",", quantiles[2], "\n")

cat("\nGroup Statistics:\n")
summary_stats <- plot_data %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

# Save summary statistics
write.csv(
  summary_stats,
  file = file.path(output_dir, "skewed_distribution_groups_summary.csv"),
  row.names = FALSE
)

cat("\n=== Plot saved to:", file.path(output_dir, "skewed_distribution_groups_overlay.png"), "===\n")
cat("Summary statistics saved to:", file.path(output_dir, "skewed_distribution_groups_summary.csv"), "\n")

