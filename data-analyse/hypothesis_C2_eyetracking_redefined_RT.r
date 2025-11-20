# Hypothesis C2: Redefine RT based on eye-tracking data
# Instead of using the original RT, compute RT from when the eye enters
# a larger circle around each cue location. This tests if bimodality
# persists when RT is redefined based on actual eye movement.

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
# Step 1: Determine cue/target locations from the data
# ============================================================================
# We'll infer the 4 target locations by looking at where eyes land
# when PointTargetResponse indicates each location (1-4)

cat("\n=== Determining target locations from eye-tracking data ===\n")

# Extract eye position at the end of each trial (last valid eye sample)
# and group by PointTargetResponse to find typical locations
combined_data$PointTargetResponse_num <- as.numeric(as.character(combined_data$PointTargetResponse))

# Get last eye position for each trial (optimized)
# First, find max eye sample number from NumberEyeSamples
max_eye_samples <- max(as.numeric(as.character(combined_data$NumberEyeSamples)), na.rm = TRUE)
if (is.na(max_eye_samples) || max_eye_samples > 200) {
  max_eye_samples <- 200  # Cap at 200 for efficiency
}
cat(sprintf("Maximum eye samples per trial: %d\n", max_eye_samples))

get_last_eye_position <- function(row) {
  last_x <- NA
  last_y <- NA
  last_t <- NA
  
  # Check NumberEyeSamples first to limit search
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
        last_x <- x_val
        last_y <- y_val
        last_t <- time_val
      }
    } else {
      break  # No more eye samples
    }
  }
  
  return(list(x = last_x, y = last_y, t = last_t))
}

# Extract last eye positions
cat("Extracting last eye positions...\n")
last_positions <- lapply(1:nrow(combined_data), function(i) {
  get_last_eye_position(combined_data[i, ])
})

combined_data$LastEyeX <- sapply(last_positions, function(p) p$x)
combined_data$LastEyeY <- sapply(last_positions, function(p) p$y)
combined_data$LastEyeT <- sapply(last_positions, function(p) p$t)

# Find typical locations for each target (1-4)
# Use median position for each PointTargetResponse value
target_locations <- data.frame()
for (target in 1:4) {
  target_data <- combined_data[
    !is.na(combined_data$PointTargetResponse_num) & 
    combined_data$PointTargetResponse_num == target &
    !is.na(combined_data$LastEyeX) & !is.na(combined_data$LastEyeY),
  ]
  
  if (nrow(target_data) > 10) {
    median_x <- median(target_data$LastEyeX, na.rm = TRUE)
    median_y <- median(target_data$LastEyeY, na.rm = TRUE)
    mean_x <- mean(target_data$LastEyeX, na.rm = TRUE)
    mean_y <- mean(target_data$LastEyeY, na.rm = TRUE)
    
    target_locations <- rbind(target_locations, data.frame(
      Target = target,
      X = median_x,
      Y = median_y,
      MeanX = mean_x,
      MeanY = mean_y,
      N = nrow(target_data)
    ))
    
    cat(sprintf("Target %d: (%.2f, %.2f) - %d trials\n", 
                target, median_x, median_y, nrow(target_data)))
  }
}

# If we couldn't determine locations from PointTargetResponse, try using
# the actual eye positions to cluster into 4 locations
if (nrow(target_locations) < 4) {
  cat("\nInsufficient data from PointTargetResponse. Clustering eye positions...\n")
  
  valid_eye <- combined_data[
    !is.na(combined_data$LastEyeX) & !is.na(combined_data$LastEyeY) &
    is.finite(combined_data$LastEyeX) & is.finite(combined_data$LastEyeY),
  ]
  
  if (nrow(valid_eye) > 100) {
    # Use k-means to find 4 clusters
    set.seed(42)
    clusters <- kmeans(
      cbind(valid_eye$LastEyeX, valid_eye$LastEyeY),
      centers = 4,
      nstart = 10,
      iter.max = 100
    )
    
    target_locations <- data.frame(
      Target = 1:4,
      X = clusters$centers[, 1],
      Y = clusters$centers[, 2],
      MeanX = clusters$centers[, 1],
      MeanY = clusters$centers[, 2],
      N = as.numeric(table(clusters$cluster))
    )
    
    cat("Clustered locations:\n")
    print(target_locations)
  }
}

if (nrow(target_locations) == 0) {
  stop("Could not determine target locations from eye-tracking data!")
}

# ============================================================================
# Step 2: Extract full eye-tracking sequence for each trial
# ============================================================================
cat("\n=== Extracting eye-tracking sequences ===\n")

# Define max_eye_samples for use in extract_eye_sequence
max_eye_samples_seq <- max_eye_samples

extract_eye_sequence <- function(row) {
  eye_times <- c()
  eye_x <- c()
  eye_y <- c()
  
  # Check NumberEyeSamples first to limit search
  n_samples <- as.numeric(as.character(row$NumberEyeSamples))
  if (!is.na(n_samples) && n_samples > 0) {
    max_i <- min(n_samples, max_eye_samples_seq)
  } else {
    max_i <- max_eye_samples_seq
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

# ============================================================================
# Step 3: Define larger circles around each target and detect entry
# ============================================================================
# Define circle radius (in pixels/units). We'll use a multiple of the
# typical distance between targets to make the circles larger
if (nrow(target_locations) >= 2) {
  # Calculate typical distance between targets
  distances <- c()
  for (i in 1:(nrow(target_locations)-1)) {
    for (j in (i+1):nrow(target_locations)) {
      dist <- sqrt((target_locations$X[i] - target_locations$X[j])^2 +
                   (target_locations$Y[i] - target_locations$Y[j])^2)
      distances <- c(distances, dist)
    }
  }
  typical_distance <- median(distances, na.rm = TRUE)
  
  # Calculate spread of eye positions to determine appropriate radius
  # Use a larger circle - check actual eye position spread
  all_eye_x <- combined_data$LastEyeX[!is.na(combined_data$LastEyeX)]
  all_eye_y <- combined_data$LastEyeY[!is.na(combined_data$LastEyeY)]
  eye_spread_x <- diff(range(all_eye_x, na.rm = TRUE))
  eye_spread_y <- diff(range(all_eye_y, na.rm = TRUE))
  
  # Use a radius that's large enough to capture eye positions near targets
  # Try 1.5x typical distance, or a fixed larger value
  circle_radius <- max(typical_distance * 1.5, 0.15)  # At least 0.15 units
  cat(sprintf("\nTypical distance between targets: %.4f\n", typical_distance))
  cat(sprintf("Eye coordinate spread: X=%.4f, Y=%.4f\n", eye_spread_x, eye_spread_y))
  cat(sprintf("Circle radius: %.4f (1.5x typical distance, min 0.15)\n", circle_radius))
} else {
  # Fallback: use a reasonable default based on coordinate scale
  circle_radius <- 0.1  # Based on coordinate scale seen (0.1-0.2 range)
  cat(sprintf("\nUsing default circle radius: %.4f\n", circle_radius))
}

# Function to check if a point is within a circle
point_in_circle <- function(x, y, center_x, center_y, radius) {
  dist <- sqrt((x - center_x)^2 + (y - center_y)^2)
  return(dist <= radius)
}

# Function to find when eye enters each target circle
# eye_seq$times should be absolute times (EyeOffsetTime + EyeT)
# cue_time is CueTime (absolute)
find_first_circle_entry <- function(eye_seq, cue_time, target_locs, radius) {
  if (length(eye_seq$times) == 0) return(list(target = NA, time = NA, rt = NA))
  
  # Relative times from cue onset
  relative_times <- eye_seq$times - cue_time
  
  # Check each target circle
  first_entries <- data.frame(
    target = integer(),
    time = numeric(),
    rt = numeric()
  )
  
  for (targ_idx in 1:nrow(target_locs)) {
    target_num <- target_locs$Target[targ_idx]
    center_x <- target_locs$X[targ_idx]
    center_y <- target_locs$Y[targ_idx]
    
    # Find first time eye enters this circle (after cue onset)
    for (i in 1:length(eye_seq$times)) {
      # Only consider samples after cue onset (with small tolerance for timing precision)
      if (relative_times[i] >= -0.1) {  # Allow 100ms before cue (timing precision)
        if (point_in_circle(eye_seq$x[i], eye_seq$y[i], 
                           center_x, center_y, radius)) {
          # Only count if it's after cue onset (or very close)
          if (relative_times[i] >= -0.05) {
            first_entries <- rbind(first_entries, data.frame(
              target = target_num,
              time = eye_seq$times[i],
              rt = max(0, relative_times[i])  # RT can't be negative
            ))
            break
          }
        }
      }
    }
  }
  
  # Return the first target entered (earliest RT)
  if (nrow(first_entries) > 0) {
    first_entry <- first_entries[which.min(first_entries$rt), ]
    return(list(
      target = first_entry$target,
      time = first_entry$time,
      rt = first_entry$rt
    ))
  } else {
    return(list(target = NA, time = NA, rt = NA))
  }
}

# ============================================================================
# Step 4: Process all trials to redefine RT
# ============================================================================
cat("\n=== Processing trials to redefine RT ===\n")

# Get CueTime and EyeOffsetTime for each trial
combined_data$CueTime_num <- as.numeric(as.character(combined_data$CueTime))
combined_data$EyeOffsetTime_num <- as.numeric(as.character(combined_data$EyeOffsetTime))

# Eye times appear to be relative to EyeOffsetTime, not absolute
# So we need: absolute_eye_time = EyeOffsetTime + EyeT
# And RT = absolute_eye_time - CueTime = EyeOffsetTime + EyeT - CueTime

# Diagnostic: Check a few sample trials to understand the data
cat("\n=== Diagnostic: Checking sample trials ===\n")
sample_trials <- combined_data[!is.na(combined_data$CueTime_num) & 
                               !is.na(combined_data$NumberEyeSamples) &
                               as.numeric(as.character(combined_data$NumberEyeSamples)) > 10, ][1:min(5, nrow(combined_data)), ]

if (nrow(sample_trials) > 0) {
  for (i in 1:nrow(sample_trials)) {
    row <- sample_trials[i, ]
    cat(sprintf("\nSample trial %d:\n", i))
    cat(sprintf("  CueTime: %.3f\n", row$CueTime_num))
    cat(sprintf("  NumberEyeSamples: %s\n", row$NumberEyeSamples))
    
    eye_seq <- extract_eye_sequence(row)
    eye_offset <- as.numeric(as.character(row$EyeOffsetTime))
    if (length(eye_seq$times) > 0) {
      cat(sprintf("  Eye samples: %d\n", length(eye_seq$times)))
      cat(sprintf("  EyeOffsetTime: %.3f\n", eye_offset))
      cat(sprintf("  Eye time range (relative): %.3f to %.3f\n", 
                  min(eye_seq$times), max(eye_seq$times)))
      if (!is.na(eye_offset)) {
        abs_times <- eye_offset + eye_seq$times
        cat(sprintf("  Eye time range (absolute): %.3f to %.3f\n",
                    min(abs_times), max(abs_times)))
        cat(sprintf("  Time diff from cue: %.3f to %.3f\n",
                    min(abs_times) - row$CueTime_num,
                    max(abs_times) - row$CueTime_num))
      }
      cat(sprintf("  Eye X range: %.3f to %.3f\n", 
                  min(eye_seq$x), max(eye_seq$x)))
      cat(sprintf("  Eye Y range: %.3f to %.3f\n", 
                  min(eye_seq$y), max(eye_seq$y)))
      
      # Check distances to each target
      for (targ_idx in 1:nrow(target_locations)) {
        center_x <- target_locations$X[targ_idx]
        center_y <- target_locations$Y[targ_idx]
        min_dist <- min(sqrt((eye_seq$x - center_x)^2 + (eye_seq$y - center_y)^2))
        cat(sprintf("  Min distance to target %d: %.4f (radius: %.4f)\n",
                    target_locations$Target[targ_idx], min_dist, circle_radius))
      }
    }
  }
}

# Process each trial (with progress updates)
cat("Processing trials to redefine RT...\n")
redefined_results <- lapply(1:nrow(combined_data), function(i) {
  if (i %% 1000 == 0) cat(sprintf("Processing trial %d/%d (%.1f%%)\n", 
                                   i, nrow(combined_data), 
                                   100*i/nrow(combined_data)))
  
  row <- combined_data[i, ]
  cue_time <- row$CueTime_num
  eye_offset <- row$EyeOffsetTime_num
  
  if (is.na(cue_time) || !is.finite(cue_time)) {
    return(list(
      new_rt = NA,
      detected_target = NA,
      original_target = row$PointTargetResponse_num,
      correct = NA
    ))
  }
  
  # Extract eye sequence (times are relative to EyeOffsetTime)
  eye_seq <- extract_eye_sequence(row)
  
  if (length(eye_seq$times) == 0) {
    return(list(
      new_rt = NA,
      detected_target = NA,
      original_target = row$PointTargetResponse_num,
      correct = NA
    ))
  }
  
  # Convert eye times to absolute times
  if (!is.na(eye_offset) && is.finite(eye_offset)) {
    eye_seq$times <- eye_offset + eye_seq$times
  } else {
    # If no EyeOffsetTime, assume eye times are already absolute
    # (but this is less reliable)
  }
  
  # Now find first circle entry
  entry <- find_first_circle_entry(eye_seq, cue_time, target_locations, circle_radius)
  
  # Determine correct/incorrect
  # Need to know which target had the cue (max reward)
  cue_values_str <- as.character(row$CueValues)
  if (!is.na(cue_values_str) && cue_values_str != "") {
    # Parse cue values
    clean_str <- gsub("[\\[\\]\\s]", "", cue_values_str)
    if (grepl(",", clean_str)) {
      cue_vals <- as.numeric(strsplit(clean_str, ",")[[1]])
    } else {
      cue_vals <- as.numeric(strsplit(clean_str, "")[[1]])
    }
    cue_vals <- cue_vals[!is.na(cue_vals)]
    
    if (length(cue_vals) > 0) {
      max_reward <- max(cue_vals)
      # Find which target location corresponds to max reward
      # This is tricky - we need to map cue positions to target numbers
      # For now, use PointTargetResponse as proxy for correct target
      # (assuming it's correct when it matches max reward)
      cue_response_val <- as.numeric(as.character(row$CueResponseValue))
      correct_target <- ifelse(
        !is.na(cue_response_val) && cue_response_val == max_reward,
        row$PointTargetResponse_num,
        NA
      )
    } else {
      correct_target <- NA
    }
  } else {
    correct_target <- NA
  }
  
  # Check if detected target matches correct target
  is_correct <- ifelse(
    !is.na(entry$target) && !is.na(correct_target),
    entry$target == correct_target,
    NA
  )
  
  return(list(
    new_rt = entry$rt,
    detected_target = entry$target,
    original_target = row$PointTargetResponse_num,
    correct = is_correct
  ))
})

# Add redefined RT to data
combined_data$RT_redefined <- sapply(redefined_results, function(r) r$new_rt)
combined_data$DetectedTarget <- sapply(redefined_results, function(r) r$detected_target)
combined_data$CorrectRedefined <- sapply(redefined_results, function(r) r$correct)

cat("\n=== Summary of redefined RT ===\n")
cat(sprintf("Trials with valid redefined RT: %d / %d (%.1f%%)\n",
            sum(!is.na(combined_data$RT_redefined)), 
            nrow(combined_data),
            100 * sum(!is.na(combined_data$RT_redefined)) / nrow(combined_data)))

valid_redefined <- combined_data[!is.na(combined_data$RT_redefined) & 
                                combined_data$RT_redefined > 0 & 
                                combined_data$RT_redefined <= 10, ]

cat(sprintf("Valid redefined RT (0-10s): %d trials\n", nrow(valid_redefined)))
cat(sprintf("Mean original RT: %.3f s\n", mean(combined_data$RT_sec, na.rm = TRUE)))
cat(sprintf("Mean redefined RT: %.3f s\n", mean(valid_redefined$RT_redefined, na.rm = TRUE)))

# ============================================================================
# Step 5: Plot RT distributions by cue condition
# ============================================================================
output_dir <- "figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Plot 1: Overall RT distribution comparison (original vs redefined)
p1 <- ggplot() +
  geom_density(data = combined_data[!is.na(combined_data$RT_sec), ],
               aes(x = RT_sec, color = "Original RT"), 
               linewidth = 1.5, alpha = 0.7) +
  geom_density(data = valid_redefined,
               aes(x = RT_redefined, color = "Redefined RT (Eye-tracking)"), 
               linewidth = 1.5, alpha = 0.7) +
  labs(
    title = "Hypothesis C2: RT Distribution - Original vs Redefined (Eye-tracking)",
    subtitle = "Redefined RT based on when eye enters larger circle around target",
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

ggsave(
  filename = file.path(output_dir, "hypothesis_C2_RT_comparison.png"),
  plot = p1,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)

# Plot 2: RT distribution by cue condition (redefined RT)
if (nrow(valid_redefined) > 0 && "CueCondition" %in% names(valid_redefined)) {
  valid_redefined$CueCondition_fac <- as.factor(valid_redefined$CueCondition)
  
  p2 <- ggplot(valid_redefined, aes(x = RT_redefined, color = CueCondition_fac)) +
    geom_density(linewidth = 1.2, alpha = 0.7) +
    labs(
      title = "Hypothesis C2: Redefined RT Distribution by Cue Condition",
      subtitle = "RT based on eye-tracking circle entry - Testing for bimodality",
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
    filename = file.path(output_dir, "hypothesis_C2_RT_by_condition.png"),
    plot = p2,
    width = 14,
    height = 8,
    dpi = 300,
    bg = "white"
  )
  
  # Plot 3: Individual condition plots (facetted)
  unique_conditions <- sort(unique(valid_redefined$CueCondition))
  if (length(unique_conditions) > 0 && length(unique_conditions) <= 10) {
    p3 <- ggplot(valid_redefined, aes(x = RT_redefined)) +
      geom_density(linewidth = 1.2, fill = "steelblue", alpha = 0.5) +
      facet_wrap(~ CueCondition_fac, scales = "free_y") +
      labs(
        title = "Hypothesis C2: Redefined RT Distribution by Cue Condition (Individual)",
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
      filename = file.path(output_dir, "hypothesis_C2_RT_by_condition_faceted.png"),
      plot = p3,
      width = 16,
      height = 12,
      dpi = 300,
      bg = "white"
    )
  }
}

# Summary statistics
cat("\n=== Summary Statistics by Cue Condition (Redefined RT) ===\n")
if (nrow(valid_redefined) > 0 && "CueCondition" %in% names(valid_redefined)) {
  summary_stats <- valid_redefined %>%
    group_by(CueCondition) %>%
    summarise(
      n = n(),
      mean_RT = mean(RT_redefined, na.rm = TRUE),
      median_RT = median(RT_redefined, na.rm = TRUE),
      sd_RT = sd(RT_redefined, na.rm = TRUE),
      min_RT = min(RT_redefined, na.rm = TRUE),
      max_RT = max(RT_redefined, na.rm = TRUE),
      .groups = "drop"
    )
  print(summary_stats)
  
  # Save summary statistics
  write.csv(
    summary_stats,
    file = file.path(output_dir, "hypothesis_C2_summary.csv"),
    row.names = FALSE
  )
}

cat("\n=== Hypothesis C2 Analysis Complete ===\n")
cat("Plots saved to:", output_dir, "\n")
cat("\nKey Question: Does bimodality persist with redefined RT?\n")
cat("If bimodality disappears, it suggests the original RT measurement\n")
cat("was affected by eye-tracking artifacts (re-fixations).\n")

