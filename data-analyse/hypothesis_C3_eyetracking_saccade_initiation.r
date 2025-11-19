# Hypothesis C3: Redefine RT based on saccade initiation
# RT is redefined as the time from cue onset to when the eye first moves
# away from the initial fixation point (center). This measures when the
# saccade is initiated, not when it reaches the target.

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

# ============================================================================
# Step 1: Determine initial fixation point for each trial
# ============================================================================
cat("\n=== Determining initial fixation points ===\n")

# Get CueGazeX and CueGazeY (eye position at cue presentation)
combined_data$CueGazeX_num <- as.numeric(as.character(combined_data$CueGazeX))
combined_data$CueGazeY_num <- as.numeric(as.character(combined_data$CueGazeY))

# Also get CueTime and EyeOffsetTime
combined_data$CueTime_num <- as.numeric(as.character(combined_data$CueTime))
combined_data$EyeOffsetTime_num <- as.numeric(as.character(combined_data$EyeOffsetTime))

# Find max eye samples for efficiency
max_eye_samples <- max(as.numeric(as.character(combined_data$NumberEyeSamples)), na.rm = TRUE)
if (is.na(max_eye_samples) || max_eye_samples > 200) {
  max_eye_samples <- 200
}
cat(sprintf("Maximum eye samples per trial: %d\n", max_eye_samples))

# Function to extract eye sequence
extract_eye_sequence <- function(row) {
  eye_times <- c()
  eye_x <- c()
  eye_y <- c()
  
  n_samples <- as.numeric(as.character(row$NumberEyeSamples))
  if (!is.na(n_samples) && n_samples > 0) {
    max_i <- min(n_samples, max_eye_samples)
  } else {
    max_i <- max_eye_samples
  }
  
  for (i in 1:max_i) {
    time_col <- paste0("EyeT", i)
    x_col <- paste0("EyeX", i)
    y_col <- paste0("EyeY", i)
    
    if (time_col %in% names(row)) {
      time_val <- as.numeric(as.character(row[[time_col]]))
      x_val <- as.numeric(as.character(row[[x_col]]))
      y_val <- as.numeric(as.character(row[[y_col]]))
      
      if (!is.na(time_val) && !is.na(x_val) && !is.na(y_val) &&
          is.finite(time_val) && is.finite(x_val) && is.finite(y_val)) {
        eye_times <- c(eye_times, time_val)
        eye_x <- c(eye_x, x_val)
        eye_y <- c(eye_y, y_val)
      }
    } else {
      break
    }
  }
  
  # Sort by time
  if (length(eye_times) > 0) {
    ord <- order(eye_times)
    return(list(
      times = eye_times[ord],
      x = eye_x[ord],
      y = eye_y[ord]
    ))
  } else {
    return(list(times = numeric(0), x = numeric(0), y = numeric(0)))
  }
}

# Determine initial fixation point for each trial
# Strategy: Use CueGazeX/Y if available, otherwise use first few eye samples
get_initial_fixation <- function(row) {
  # First try CueGazeX/Y
  if (!is.na(row$CueGazeX_num) && !is.na(row$CueGazeY_num) &&
      is.finite(row$CueGazeX_num) && is.finite(row$CueGazeY_num)) {
    return(list(x = row$CueGazeX_num, y = row$CueGazeY_num, method = "CueGaze"))
  }
  
  # Otherwise, use first few eye samples (average of first 5 samples)
  eye_seq <- extract_eye_sequence(row)
  if (length(eye_seq$times) >= 3) {
    # Use first 5 samples or all if fewer
    n_samples <- min(5, length(eye_seq$times))
    init_x <- mean(eye_seq$x[1:n_samples], na.rm = TRUE)
    init_y <- mean(eye_seq$y[1:n_samples], na.rm = TRUE)
    if (is.finite(init_x) && is.finite(init_y)) {
      return(list(x = init_x, y = init_y, method = "FirstSamples"))
    }
  }
  
  return(list(x = NA, y = NA, method = "None"))
}

# Extract initial fixations for all trials
cat("Extracting initial fixation points...\n")
initial_fixations <- lapply(1:nrow(combined_data), function(i) {
  if (i %% 1000 == 0) cat(sprintf("  Processing trial %d/%d\n", i, nrow(combined_data)))
  get_initial_fixation(combined_data[i, ])
})

combined_data$InitialFixX <- sapply(initial_fixations, function(f) f$x)
combined_data$InitialFixY <- sapply(initial_fixations, function(f) f$y)
combined_data$InitialFixMethod <- sapply(initial_fixations, function(f) f$method)

cat(sprintf("\nInitial fixation determination:\n"))
cat(sprintf("  CueGaze: %d trials\n", sum(combined_data$InitialFixMethod == "CueGaze", na.rm = TRUE)))
cat(sprintf("  FirstSamples: %d trials\n", sum(combined_data$InitialFixMethod == "FirstSamples", na.rm = TRUE)))
cat(sprintf("  None: %d trials\n", sum(combined_data$InitialFixMethod == "None" | is.na(combined_data$InitialFixMethod))))

# ============================================================================
# Step 2: Define fixation circle radius
# ============================================================================
# ============================================================================
# ADJUSTABLE PARAMETER: Fixation Circle Radius
# ============================================================================
# Set FIXATION_RADIUS_FACTOR to control the size of the fixation circle:
# - This is a multiplier of the typical saccade distance
# - Smaller values (0.3-0.5): More sensitive, detects smaller movements away from fixation
# - Larger values (0.6-1.0): Less sensitive, requires larger movements before detecting saccade
# - Set to NULL to use automatic calculation based on data
# 
# RECOMMENDED VALUES:
#   - 0.3-0.4: Strict (detects very small movements, may give faster RTs)
#   - 0.5-0.7: Moderate (default, balanced - currently 0.7)
#   - 0.8-1.0: Lenient (only detects larger movements, may give slower RTs)
#
# LINE TO EDIT: Change the value on line 188 below (currently 0.7)
FIXATION_RADIUS_FACTOR <- 1.0  # <--- EDIT THIS VALUE (0.3 to 1.0 recommended)
# ============================================================================

# Calculate typical eye movement distance to determine appropriate radius
# The radius should be small enough to detect when eye leaves fixation,
# but large enough to account for normal fixation variability

# Get all initial fixation positions
valid_fixations <- combined_data[
  !is.na(combined_data$InitialFixX) & !is.na(combined_data$InitialFixY) &
  is.finite(combined_data$InitialFixX) & is.finite(combined_data$InitialFixY),
]

if (nrow(valid_fixations) > 100) {
  # Calculate spread of initial fixations
  fix_spread_x <- diff(range(valid_fixations$InitialFixX, na.rm = TRUE))
  fix_spread_y <- diff(range(valid_fixations$InitialFixY, na.rm = TRUE))
  
  # Calculate typical distance eye moves during saccades
  # Sample a few trials to see typical saccade amplitude
  sample_trials <- valid_fixations[sample(min(100, nrow(valid_fixations))), ]
  saccade_distances <- c()
  
  for (i in 1:nrow(sample_trials)) {
    row <- sample_trials[i, ]
    eye_seq <- extract_eye_sequence(row)
    if (length(eye_seq$times) > 10) {
      # Distance from initial fixation to later positions
      init_x <- row$InitialFixX
      init_y <- row$InitialFixY
      for (j in 10:min(50, length(eye_seq$times))) {
        dist <- sqrt((eye_seq$x[j] - init_x)^2 + (eye_seq$y[j] - init_y)^2)
        if (is.finite(dist) && dist > 0.01) {
          saccade_distances <- c(saccade_distances, dist)
        }
      }
    }
  }
  
  if (length(saccade_distances) > 10) {
    typical_saccade <- median(saccade_distances, na.rm = TRUE)
    # Use the adjustable fixation radius factor (defined above)
    # If FIXATION_RADIUS_FACTOR is NULL, use automatic calculation
    if (exists("FIXATION_RADIUS_FACTOR") && !is.null(FIXATION_RADIUS_FACTOR)) {
      fixation_radius <- typical_saccade * FIXATION_RADIUS_FACTOR
      cat(sprintf("\nTypical saccade distance: %.4f\n", typical_saccade))
      cat(sprintf("Fixation circle radius: %.4f (%.0f%% of typical saccade - USER SET)\n", 
                  fixation_radius, FIXATION_RADIUS_FACTOR * 100))
    } else {
      # Automatic calculation: use 70% of typical saccade
      fixation_radius <- typical_saccade * 0.7
      cat(sprintf("\nTypical saccade distance: %.4f\n", typical_saccade))
      cat(sprintf("Fixation circle radius: %.4f (70%% of typical saccade - AUTO)\n", fixation_radius))
    }
  } else {
    # Fallback: use a fixed small radius
    fixation_radius <- 0.05
    cat(sprintf("\nUsing default fixation radius: %.4f\n", fixation_radius))
  }
} else {
  # Fallback: use a fixed small radius
  fixation_radius <- 0.05
  cat(sprintf("\nUsing default fixation radius: %.4f\n", fixation_radius))
}

# ============================================================================
# Step 3: Find when eye leaves initial fixation circle
# ============================================================================
# Function to check if point is within circle
point_in_circle <- function(x, y, center_x, center_y, radius) {
  dist <- sqrt((x - center_x)^2 + (y - center_y)^2)
  return(dist <= radius)
}

# Function to find when eye first leaves fixation circle
find_saccade_initiation <- function(eye_seq, initial_fix_x, initial_fix_y, 
                                   fixation_radius, cue_time, eye_offset) {
  if (length(eye_seq$times) == 0) return(list(rt = NA, exit_time = NA))
  if (is.na(initial_fix_x) || is.na(initial_fix_y)) return(list(rt = NA, exit_time = NA))
  
  # Convert eye times to absolute times
  if (!is.na(eye_offset) && is.finite(eye_offset)) {
    abs_times <- eye_offset + eye_seq$times
  } else {
    abs_times <- eye_seq$times
  }
  
  # Relative times from cue onset
  relative_times <- abs_times - cue_time
  
  # Strategy: Find when eye was last in fixation circle, then when it first leaves
  # This handles cases where eye might already be moving at cue onset
  
  # Find the last time eye was in fixation circle (before or at cue onset)
  last_in_circle_idx <- NA
  last_in_circle_time <- NA
  
  for (i in 1:length(eye_seq$times)) {
    # Look at samples from 200ms before cue to 50ms after
    if (relative_times[i] >= -0.2 && relative_times[i] <= 0.05) {
      if (point_in_circle(eye_seq$x[i], eye_seq$y[i], 
                         initial_fix_x, initial_fix_y, fixation_radius)) {
        last_in_circle_idx <- i
        last_in_circle_time <- relative_times[i]
      }
    }
  }
  
  # If we never found eye in circle, the initial fixation might be wrong
  # or the eye is already moving - skip this trial
  if (is.na(last_in_circle_idx)) {
    return(list(rt = NA, exit_time = NA))
  }
  
  # Now find first sample AFTER the last in-circle time where eye is outside
  # Start searching from after cue onset (or from last_in_circle_idx if it's after cue)
  start_idx <- max(last_in_circle_idx, 
                   min(which(relative_times >= 0), length(relative_times), na.rm = TRUE))
  
  if (is.na(start_idx) || start_idx > length(eye_seq$times)) {
    return(list(rt = NA, exit_time = NA))
  }
  
  for (i in start_idx:length(eye_seq$times)) {
    # Only consider samples after cue onset
    if (relative_times[i] >= 0) {
      # Check if eye is outside fixation circle
      if (!point_in_circle(eye_seq$x[i], eye_seq$y[i], 
                          initial_fix_x, initial_fix_y, fixation_radius)) {
        # Eye has left fixation - this is saccade initiation
        rt <- relative_times[i]
        if (rt < 0) rt <- 0  # RT can't be negative
        return(list(rt = rt, exit_time = abs_times[i]))
      }
    }
  }
  
  # Eye never left fixation circle (within our sample window)
  return(list(rt = NA, exit_time = NA))
}

# ============================================================================
# Step 4: Process all trials to redefine RT
# ============================================================================
cat("\n=== Processing trials to redefine RT (saccade initiation) ===\n")

redefined_results <- lapply(1:nrow(combined_data), function(i) {
  if (i %% 1000 == 0) {
    cat(sprintf("Processing trial %d/%d (%.1f%%)\n", 
                i, nrow(combined_data), 100*i/nrow(combined_data)))
  }
  
  row <- combined_data[i, ]
  cue_time <- row$CueTime_num
  eye_offset <- row$EyeOffsetTime_num
  init_fix_x <- row$InitialFixX
  init_fix_y <- row$InitialFixY
  
  if (is.na(cue_time) || !is.finite(cue_time) ||
      is.na(init_fix_x) || is.na(init_fix_y)) {
    return(list(new_rt = NA))
  }
  
  # Extract eye sequence
  eye_seq <- extract_eye_sequence(row)
  
  if (length(eye_seq$times) == 0) {
    return(list(new_rt = NA))
  }
  
  # Find saccade initiation
  result <- find_saccade_initiation(eye_seq, init_fix_x, init_fix_y,
                                   fixation_radius, cue_time, eye_offset)
  
  return(list(new_rt = result$rt))
})

# Add redefined RT to data
combined_data$RT_saccade_initiation <- sapply(redefined_results, function(r) r$new_rt)

cat("\n=== Summary of redefined RT (saccade initiation) ===\n")
cat(sprintf("Trials with valid redefined RT: %d / %d (%.1f%%)\n",
            sum(!is.na(combined_data$RT_saccade_initiation)), 
            nrow(combined_data),
            100 * sum(!is.na(combined_data$RT_saccade_initiation)) / nrow(combined_data)))

# Filter valid RTs - allow very small RTs (saccade initiation can be very fast)
# But exclude negative or extremely large values
valid_redefined <- combined_data[
  !is.na(combined_data$RT_saccade_initiation) & 
  is.finite(combined_data$RT_saccade_initiation) &
  combined_data$RT_saccade_initiation >= 0 & 
  combined_data$RT_saccade_initiation <= 2,  # Allow up to 2s
]

# Ensure CueCondition is available
if (!"CueCondition" %in% names(valid_redefined)) {
  cat("Warning: CueCondition not found in valid_redefined\n")
}

cat(sprintf("Valid redefined RT (0-2s): %d trials\n", nrow(valid_redefined)))
if (nrow(valid_redefined) > 0) {
  cat(sprintf("Mean original RT: %.3f s\n", mean(combined_data$RT_sec, na.rm = TRUE)))
  cat(sprintf("Mean redefined RT (saccade initiation): %.3f s\n", 
              mean(valid_redefined$RT_saccade_initiation, na.rm = TRUE)))
  cat(sprintf("RT range: %.4f to %.4f s\n",
              min(valid_redefined$RT_saccade_initiation, na.rm = TRUE),
              max(valid_redefined$RT_saccade_initiation, na.rm = TRUE)))
  if ("CueCondition" %in% names(valid_redefined)) {
    cat(sprintf("Cue conditions in valid data: %s\n", 
                paste(sort(unique(valid_redefined$CueCondition)), collapse = ", ")))
  }
} else {
  cat("WARNING: No valid redefined RT data!\n")
}

# ============================================================================
# Step 5: Plot RT distributions by cue condition
# ============================================================================
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Plot 1: Overall RT distribution comparison
# Check data before plotting
cat("\n=== Plotting diagnostics ===\n")
cat(sprintf("Original RT data: %d rows\n", sum(!is.na(combined_data$RT_sec))))
cat(sprintf("Redefined RT data: %d rows\n", nrow(valid_redefined)))
if (nrow(valid_redefined) > 0) {
  cat(sprintf("Redefined RT values: min=%.4f, max=%.4f, mean=%.4f\n",
              min(valid_redefined$RT_saccade_initiation, na.rm = TRUE),
              max(valid_redefined$RT_saccade_initiation, na.rm = TRUE),
              mean(valid_redefined$RT_saccade_initiation, na.rm = TRUE)))
}

p1 <- ggplot() +
  geom_density(data = combined_data[!is.na(combined_data$RT_sec), ],
               aes(x = RT_sec, color = "Original RT"), 
               linewidth = 1.5, alpha = 0.7) +
  labs(
    title = "Hypothesis C3: RT Distribution - Original vs Redefined (Saccade Initiation)",
    subtitle = "Redefined RT based on when eye leaves initial fixation circle",
    x = "Reaction Time (seconds)",
    y = "Density",
    color = "RT Type"
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

# Add redefined RT if we have data
if (nrow(valid_redefined) > 0) {
  valid_redefined_df_plot1 <- as.data.frame(valid_redefined)
  p1 <- p1 + geom_density(data = valid_redefined_df_plot1,
                          aes(x = RT_saccade_initiation, color = "Redefined RT (Saccade Initiation)"), 
                          linewidth = 1.5, alpha = 0.7, adjust = 0.5) +
    coord_cartesian(xlim = c(0, 1.0))  # Show both original and redefined RT ranges
}

ggsave(
  filename = file.path(output_dir, "hypothesis_C3_RT_comparison.png"),
  plot = p1,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)

# Plot 2: RT distribution by cue condition (redefined RT)
if (nrow(valid_redefined) > 0 && "CueCondition" %in% names(valid_redefined)) {
  # Convert to data.frame for ggplot (data.table can sometimes cause issues)
  valid_redefined_df <- as.data.frame(valid_redefined)
  valid_redefined_df$CueCondition_fac <- as.factor(valid_redefined_df$CueCondition)
  
  # Check data before plotting
  cat(sprintf("Plot 2: %d rows, RT range: %.4f to %.4f\n",
              nrow(valid_redefined_df),
              min(valid_redefined_df$RT_saccade_initiation, na.rm = TRUE),
              max(valid_redefined_df$RT_saccade_initiation, na.rm = TRUE)))
  
  # Use both histogram and density for better visualization
  p2 <- ggplot(valid_redefined_df, aes(x = RT_saccade_initiation, color = CueCondition_fac)) +
    geom_density(linewidth = 1.2, alpha = 0.7, adjust = 0.5) +  # Adjust bandwidth
    coord_cartesian(xlim = c(0, 0.2)) +  # Focus on relevant range
    labs(
      title = "Hypothesis C3: Redefined RT Distribution by Cue Condition",
      subtitle = "RT based on saccade initiation - Testing for bimodality",
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
    geom_vline(xintercept = c(0.05, 0.1), 
               linetype = "dashed", alpha = 0.5, color = "gray")
  
  ggsave(
    filename = file.path(output_dir, "hypothesis_C3_RT_by_condition.png"),
    plot = p2,
    width = 14,
    height = 8,
    dpi = 300,
    bg = "white"
  )
  
  # Plot 3: Individual condition plots (facetted)
  unique_conditions <- sort(unique(valid_redefined_df$CueCondition))
  if (length(unique_conditions) > 0 && length(unique_conditions) <= 10) {
    # Set appropriate x-axis limits for saccade initiation RTs (typically 0-0.2s)
    xlim_max <- min(0.2, max(valid_redefined_df$RT_saccade_initiation, na.rm = TRUE) * 1.2)
    
    cat(sprintf("Plot 3: %d conditions, xlim: 0 to %.3f\n", 
                length(unique_conditions), xlim_max))
    
    # Use histogram with density overlay for better visibility
    # Create plot with both histogram and density for visibility
    p3 <- ggplot(valid_redefined_df, aes(x = RT_saccade_initiation)) +
      # Histogram for visibility
      geom_histogram(aes(y = after_stat(density)), bins = 30, 
                    fill = "steelblue", alpha = 0.5, color = "black", linewidth = 0.3,
                    boundary = 0) +
      # Density curve overlay
      geom_density(linewidth = 1.2, color = "darkblue", alpha = 0.8, adjust = 0.5) +
      facet_wrap(~ CueCondition_fac, scales = "free_y") +
      scale_x_continuous(limits = c(0, xlim_max), expand = c(0, 0)) +  # Use scale_x instead of coord_cartesian
      labs(
        title = "Hypothesis C3: Redefined RT Distribution by Cue Condition (Individual)",
        subtitle = "Each panel shows one cue condition - RT = time to saccade initiation",
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
      geom_vline(xintercept = c(0.05, 0.1), 
                 linetype = "dashed", alpha = 0.5, color = "gray")
    
    ggsave(
      filename = file.path(output_dir, "hypothesis_C3_RT_by_condition_faceted.png"),
      plot = p3,
      width = 16,
      height = 12,
      dpi = 300,
      bg = "white"
    )
  }
}

# Summary statistics
cat("\n=== Summary Statistics by Cue Condition (Redefined RT - Saccade Initiation) ===\n")
if (nrow(valid_redefined) > 0 && "CueCondition" %in% names(valid_redefined)) {
  # Convert to data.frame for dplyr operations
  valid_redefined_df <- as.data.frame(valid_redefined)
  summary_stats <- valid_redefined_df %>%
    group_by(CueCondition) %>%
    summarise(
      n = n(),
      mean_RT = mean(RT_saccade_initiation, na.rm = TRUE),
      median_RT = median(RT_saccade_initiation, na.rm = TRUE),
      sd_RT = sd(RT_saccade_initiation, na.rm = TRUE),
      min_RT = min(RT_saccade_initiation, na.rm = TRUE),
      max_RT = max(RT_saccade_initiation, na.rm = TRUE),
      .groups = "drop"
    )
  print(summary_stats)
  
  # Save summary statistics
  write.csv(
    summary_stats,
    file = file.path(output_dir, "hypothesis_C3_summary.csv"),
    row.names = FALSE
  )
}

cat("\n=== Hypothesis C3 Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")
cat("\nKey Question: Does bimodality persist when RT is redefined as saccade initiation?\n")
cat("If bimodality disappears, it suggests the original RT measurement\n")
cat("was affected by saccade duration or landing time, not initiation time.\n")

