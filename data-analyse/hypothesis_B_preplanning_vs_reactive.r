# Hypothesis B: Pre-planning vs. Reactive Saccades
# Test if fast peak corresponds to pre-planned/guessed movements
# and slow peak corresponds to value-based decisions

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

# Calculate if participant chose the optimal (max reward) location
combined_data$ChoseOptimal <- combined_data$CueResponseValue_num == combined_data$MaxReward

# Hypothesis B indicators:
# 1. Very fast RTs (< 0.25s) might be pre-planned/guessed
# 2. Fast RTs that don't choose optimal might be guesses
# 3. RT consistency across trials (pre-planning might show more consistent RTs)
# 4. Relationship between RT and choice optimality

# Classify trials based on RT and choice optimality
combined_data$TrialType <- ifelse(
  combined_data$RT_sec < 0.25,
  ifelse(combined_data$ChoseOptimal, "Fast Optimal", "Fast Non-Optimal (Guess?)"),
  ifelse(
    combined_data$RT_sec < 0.4,
    ifelse(combined_data$ChoseOptimal, "Medium Optimal", "Medium Non-Optimal"),
    ifelse(combined_data$ChoseOptimal, "Slow Optimal", "Slow Non-Optimal")
  )
)

# Alternative: Focus on very fast responses
combined_data$IsVeryFast <- combined_data$RT_sec < 0.25
combined_data$IsFast <- combined_data$RT_sec >= 0.25 & combined_data$RT_sec < 0.4
combined_data$IsSlow <- combined_data$RT_sec >= 0.4

cat("\nTrial classification by RT and optimality:\n")
print(table(combined_data$TrialType, useNA = "always"))

cat("\nOptimality rate by RT category:\n")
optimality_by_rt <- combined_data %>%
  mutate(RTCategory = case_when(
    RT_sec < 0.25 ~ "Very Fast (<0.25s)",
    RT_sec < 0.4 ~ "Fast (0.25-0.4s)",
    TRUE ~ "Slow (>0.4s)"
  )) %>%
  group_by(RTCategory) %>%
  summarise(
    n = n(),
    optimal_rate = mean(ChoseOptimal, na.rm = TRUE),
    .groups = "drop"
  )
print(optimality_by_rt)

# Create plots
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Plot 1: RT distribution by choice optimality
p1 <- ggplot(combined_data[!is.na(combined_data$ChoseOptimal), ], 
             aes(x = RT_sec, fill = ChoseOptimal)) +
  geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
  facet_wrap(~ ChoseOptimal, ncol = 1, scales = "free_y", 
             labeller = labeller(ChoseOptimal = c("FALSE" = "Non-Optimal Choice", 
                                                  "TRUE" = "Optimal Choice"))) +
  labs(
    title = "Hypothesis B: RT Distribution by Choice Optimality",
    subtitle = "Pre-planned/guessed responses might be fast and non-optimal",
    x = "Reaction Time (seconds)",
    y = "Frequency",
    fill = "Chose Optimal"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    strip.text = element_text(size = 10, face = "bold")
  ) +
  geom_vline(xintercept = c(0.2, 0.25, 0.3, 0.4, 0.6), 
             linetype = "dashed", alpha = 0.3, color = "red")

ggsave(
  filename = file.path(output_dir, "hypothesis_B_optimality.png"),
  plot = p1,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)

# Plot 2: RT density by trial type (with weighted densities)
data_for_plot2 <- combined_data[!is.na(combined_data$TrialType), ]

# Calculate proportions for each trial type
trial_type_counts_all <- table(data_for_plot2$TrialType)
total_trials_all <- sum(trial_type_counts_all)
trial_type_props_all <- trial_type_counts_all / total_trials_all

# Calculate overall density
overall_density_all <- density(data_for_plot2$RT_sec, na.rm = TRUE)
overall_df_all <- data.frame(x = overall_density_all$x, y = overall_density_all$y)

# Calculate weighted densities for each trial type
weighted_densities_all <- list()
for (tt in names(trial_type_props_all)) {
  tt_data <- data_for_plot2[data_for_plot2$TrialType == tt, ]
  if (nrow(tt_data) > 1) {
    tt_density <- density(tt_data$RT_sec, na.rm = TRUE)
    # Weight by proportion
    weighted_densities_all[[tt]] <- data.frame(
      x = tt_density$x,
      y = tt_density$y * trial_type_props_all[tt],
      TrialType = tt
    )
  }
}

# Combine all weighted densities
weighted_df_all <- do.call(rbind, weighted_densities_all)

p2 <- ggplot() +
  # Overall density line (black, thicker) - drawn first as base layer
  geom_line(data = overall_df_all, aes(x = x, y = y), 
           color = "black", linewidth = 2.5, alpha = 0.9, linetype = "solid") +
  # Weighted trial type-specific density lines - drawn on top
  geom_line(data = weighted_df_all, aes(x = x, y = y, color = TrialType), 
           alpha = 0.7, linewidth = 1.2) +
  labs(
    title = "Hypothesis B: RT Density by RT Speed and Choice Optimality",
    subtitle = "Weighted densities sum to overall (black line) - check if curves align",
    x = "Reaction Time (seconds)",
    y = "Density",
    color = "Trial Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "right"
  ) +
  geom_vline(xintercept = c(0.25, 0.4), 
             linetype = "dashed", alpha = 0.5, color = "gray")

ggsave(
  filename = file.path(output_dir, "hypothesis_B_trial_types.png"),
  plot = p2,
  width = 12,
  height = 6,
  dpi = 300,
  bg = "white"
)

# Plot 3: Optimality rate vs RT (binned)
max_rt <- max(combined_data$RT_sec, na.rm = TRUE)
if (is.finite(max_rt) && max_rt > 0) {
  breaks_seq <- seq(0, max_rt, by = 0.05)
  if (length(breaks_seq) < 2) {
    breaks_seq <- seq(0, max_rt, length.out = 20)
  }
  combined_data$RT_bin <- cut(combined_data$RT_sec, 
                              breaks = breaks_seq,
                              include.lowest = TRUE)
} else {
  combined_data$RT_bin <- NA
}

optimality_by_bin <- combined_data %>%
  filter(!is.na(RT_bin) & !is.na(ChoseOptimal)) %>%
  group_by(RT_bin) %>%
  summarise(
    n = n(),
    optimal_rate = mean(ChoseOptimal, na.rm = TRUE),
    mean_RT = mean(RT_sec, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n >= 10)  # Only show bins with at least 10 trials

p3 <- ggplot(optimality_by_bin, aes(x = mean_RT, y = optimal_rate, size = n)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 1.5) +
  labs(
    title = "Hypothesis B: Choice Optimality Rate vs RT",
    subtitle = "If pre-planning exists, very fast RTs should show lower optimality",
    x = "Reaction Time (seconds)",
    y = "Proportion Choosing Optimal",
    size = "Number of Trials"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11)
  ) +
  ylim(0, 1)

ggsave(
  filename = file.path(output_dir, "hypothesis_B_optimality_vs_RT.png"),
  plot = p3,
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"
)

# Plot 4: RT distribution by CueCondition, colored by optimality
# Focus on bimodal conditions
bimodal_conditions <- c(8, 9, 10)
combined_data_subset <- combined_data[
  combined_data$CueCondition %in% bimodal_conditions & 
  !is.na(combined_data$ChoseOptimal),
]

if (nrow(combined_data_subset) > 0) {
  p4 <- ggplot(combined_data_subset, 
               aes(x = RT_sec, fill = ChoseOptimal)) +
    geom_histogram(bins = 60, alpha = 0.7, position = "identity") +
    facet_wrap(~ CueCondition, scales = "free_y") +
    labs(
      title = "Hypothesis B: RT Distribution by CueCondition (Bimodal Conditions)",
      subtitle = "Colored by choice optimality",
      x = "Reaction Time (seconds)",
      y = "Frequency",
      fill = "Chose Optimal"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      strip.text = element_text(size = 10, face = "bold")
    )
  
  ggsave(
    filename = file.path(output_dir, "hypothesis_B_bimodal_conditions.png"),
    plot = p4,
    width = 14,
    height = 6,
    dpi = 300,
    bg = "white"
  )
}

# ============================================================================
# Condition-Specific Analysis: Conditions 3, 4, 5, 10
# ============================================================================

target_conditions <- c(3, 4, 5, 10)
combined_data_conditions <- combined_data[
  combined_data$CueCondition %in% target_conditions & 
  !is.na(combined_data$ChoseOptimal),
]

cat("\n=== Creating condition-specific plots for conditions:", paste(target_conditions, collapse = ", "), "===\n")
cat("Number of trials:", nrow(combined_data_conditions), "\n")

if (nrow(combined_data_conditions) > 0) {
  
  # ========================================================================
  # Plot 1: RT distribution by choice optimality (one plot per condition)
  # ========================================================================
  plot_list_optimality <- list()
  
  for (cond in target_conditions) {
    data_cond <- combined_data_conditions[combined_data_conditions$CueCondition == cond, ]
    
    if (nrow(data_cond) > 0) {
      p <- ggplot(data_cond, aes(x = RT_sec, fill = ChoseOptimal)) +
        geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
        labs(
          title = paste("Condition", cond),
          x = "Reaction Time (seconds)",
          y = "Frequency",
          fill = "Optimal"
        ) +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          legend.position = "bottom",
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA)
        ) +
        geom_vline(xintercept = c(0.2, 0.25, 0.3, 0.4, 0.6), 
                   linetype = "dashed", alpha = 0.3, color = "red")
      
      plot_list_optimality[[as.character(cond)]] <- p
    }
  }
  
  if (length(plot_list_optimality) > 0) {
    combined_plot_optimality <- do.call(grid.arrange, c(plot_list_optimality, ncol = 2, nrow = 2))
    
    ggsave(
      filename = file.path(output_dir, "hypothesis_B_optimality_by_condition.png"),
      plot = combined_plot_optimality,
      width = 16,
      height = 12,
      dpi = 300,
      bg = "white"
    )
    cat("Saved: hypothesis_B_optimality_by_condition.png\n")
  }
  
  # ========================================================================
  # Plot 2: RT density by trial type (one plot per condition)
  # ========================================================================
  plot_list_trialtypes <- list()
  
  for (cond in target_conditions) {
    data_cond <- combined_data_conditions[
      combined_data_conditions$CueCondition == cond & 
      !is.na(combined_data_conditions$TrialType),
    ]
    
    if (nrow(data_cond) > 0) {
      # Calculate proportions for each trial type
      trial_type_counts <- table(data_cond$TrialType)
      total_trials <- sum(trial_type_counts)
      trial_type_props <- trial_type_counts / total_trials
      
      # Add proportion as a weight column
      data_cond$weight <- trial_type_props[as.character(data_cond$TrialType)]
      
      # Calculate overall density manually for comparison
      overall_density <- density(data_cond$RT_sec, na.rm = TRUE)
      overall_df <- data.frame(x = overall_density$x, y = overall_density$y)
      
      # Calculate weighted densities for each trial type
      weighted_densities <- list()
      for (tt in names(trial_type_props)) {
        tt_data <- data_cond[data_cond$TrialType == tt, ]
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
      if (length(weighted_densities) > 0) {
        weighted_df <- do.call(rbind, weighted_densities)
        
        p <- ggplot() +
          # Overall density line (black, thicker) - drawn first as base layer
          geom_line(data = overall_df, aes(x = x, y = y), 
                   color = "black", linewidth = 2.5, alpha = 0.9, linetype = "solid") +
          # Weighted trial type-specific density lines - drawn on top
          geom_line(data = weighted_df, aes(x = x, y = y, color = TrialType), 
                   alpha = 0.7, linewidth = 1.2) +
          labs(
            title = paste("Condition", cond, "- Overall (black) vs Weighted Trial Types"),
            subtitle = "Weighted densities sum to overall (check if curves align)",
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
      filename = file.path(output_dir, "hypothesis_B_trial_types_by_condition.png"),
      plot = combined_plot_trialtypes,
      width = 16,
      height = 12,
      dpi = 300,
      bg = "white"
    )
    cat("Saved: hypothesis_B_trial_types_by_condition.png\n")
  }
  
  # ========================================================================
  # Plot 3: Optimality rate vs RT (one plot per condition)
  # ========================================================================
  plot_list_optimality_vs_rt <- list()
  
  for (cond in target_conditions) {
    data_cond <- combined_data_conditions[combined_data_conditions$CueCondition == cond, ]
    
    if (nrow(data_cond) > 0) {
      # Create RT bins for this condition
      max_rt_cond <- max(data_cond$RT_sec, na.rm = TRUE)
      if (is.finite(max_rt_cond) && max_rt_cond > 0) {
        breaks_seq <- seq(0, max_rt_cond, by = 0.05)
        if (length(breaks_seq) < 2) {
          breaks_seq <- seq(0, max_rt_cond, length.out = 20)
        }
        data_cond$RT_bin <- cut(data_cond$RT_sec, 
                                breaks = breaks_seq,
                                include.lowest = TRUE)
      } else {
        data_cond$RT_bin <- NA
      }
      
      optimality_by_bin_cond <- data_cond %>%
        filter(!is.na(RT_bin) & !is.na(ChoseOptimal)) %>%
        group_by(RT_bin) %>%
        summarise(
          n = n(),
          optimal_rate = mean(ChoseOptimal, na.rm = TRUE),
          mean_RT = mean(RT_sec, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        filter(n >= 5)  # Lower threshold for condition-specific plots
      
      if (nrow(optimality_by_bin_cond) > 0) {
        p <- ggplot(optimality_by_bin_cond, aes(x = mean_RT, y = optimal_rate, size = n)) +
          geom_point(alpha = 0.6, color = "steelblue") +
          geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 1.5) +
          labs(
            title = paste("Condition", cond),
            x = "Reaction Time (seconds)",
            y = "Proportion Optimal",
            size = "N"
          ) +
          theme_minimal() +
          theme(
            plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            legend.position = "bottom",
            plot.background = element_rect(fill = "white", color = NA),
            panel.background = element_rect(fill = "white", color = NA)
          ) +
          ylim(0, 1)
        
        plot_list_optimality_vs_rt[[as.character(cond)]] <- p
      }
    }
  }
  
  if (length(plot_list_optimality_vs_rt) > 0) {
    combined_plot_optimality_vs_rt <- do.call(grid.arrange, c(plot_list_optimality_vs_rt, ncol = 2, nrow = 2))
    
    ggsave(
      filename = file.path(output_dir, "hypothesis_B_optimality_vs_RT_by_condition.png"),
      plot = combined_plot_optimality_vs_rt,
      width = 16,
      height = 12,
      dpi = 300,
      bg = "white"
    )
    cat("Saved: hypothesis_B_optimality_vs_RT_by_condition.png\n")
  }
  
  # ========================================================================
  # Plot 4: RT distribution by condition, colored by optimality
  # ========================================================================
  p4_conditions <- ggplot(combined_data_conditions, 
                         aes(x = RT_sec, fill = ChoseOptimal)) +
    geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
    facet_wrap(~ CueCondition, ncol = 2, scales = "free_y") +
    labs(
      title = "Hypothesis B: RT Distribution by CueCondition (Conditions 3, 4, 5, 10)",
      subtitle = "Colored by choice optimality",
      x = "Reaction Time (seconds)",
      y = "Frequency",
      fill = "Chose Optimal"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      strip.text = element_text(size = 10, face = "bold"),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
  
  ggsave(
    filename = file.path(output_dir, "hypothesis_B_bimodal_conditions_3_4_5_10.png"),
    plot = p4_conditions,
    width = 14,
    height = 10,
    dpi = 300,
    bg = "white"
  )
  cat("Saved: hypothesis_B_bimodal_conditions_3_4_5_10.png\n")
  
  cat("\n=== Condition-specific plots complete ===\n")
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
    optimal_rate = mean(ChoseOptimal, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_stats)

# Save summary statistics
write.csv(
  summary_stats,
  file = file.path(output_dir, "hypothesis_B_summary.csv"),
  row.names = FALSE
)

write.csv(
  optimality_by_rt,
  file = file.path(output_dir, "hypothesis_B_optimality_by_RT.csv"),
  row.names = FALSE
)

cat("\n=== Hypothesis B Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")

