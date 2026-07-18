# ==============================================================================
# 09_submission.R - Generate Submission Files
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: Create and export submission files
# ==============================================================================

source("00_config.R")
print_header("Step 9: Generate Submission")

# ---- Submission File Creation ----

#' Create submission dataframe
create_submission <- function(predictions, ids, id_col = "ward", target_col = "target") {
    submission <- data.table(
        id = ids,
        prediction = predictions
    )

    names(submission) <- c(id_col, target_col)

    # Ensure correct column order
    submission <- submission[, c(id_col, target_col), with = FALSE]

    cat("Submission created with", nrow(submission), "rows\n")

    return(submission)
}

#' Validate submission format
validate_submission <- function(submission, sample_submission,
                                id_col = "ward", target_col = "target") {
    valid <- TRUE
    messages <- c()

    # Check dimensions
    if (nrow(submission) != nrow(sample_submission)) {
        valid <- FALSE
        messages <- c(messages, paste(
            "Row count mismatch:",
            nrow(submission), "vs expected",
            nrow(sample_submission)
        ))
    }

    # Check columns
    if (!all(names(submission) == names(sample_submission))) {
        valid <- FALSE
        messages <- c(messages, "Column names don't match sample submission")
    }

    # Check for missing values
    if (any(is.na(submission[[target_col]]))) {
        valid <- FALSE
        n_na <- sum(is.na(submission[[target_col]]))
        messages <- c(messages, paste(n_na, "missing values in predictions"))
    }

    # Check for negative values (if target should be positive)
    if (any(submission[[target_col]] < 0)) {
        n_neg <- sum(submission[[target_col]] < 0)
        messages <- c(messages, paste("Warning:", n_neg, "negative predictions"))
    }

    if (valid) {
        cat("✓ Submission validation passed!\n")
    } else {
        cat("✗ Submission validation failed:\n")
        for (msg in messages) {
            cat("  -", msg, "\n")
        }
    }

    return(valid)
}

#' Export submission to CSV
export_submission <- function(submission, filename, output_dir = OUTPUT_DIR) {
    filepath <- paste0(output_dir, filename)

    # Ensure directory exists
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    write.csv(submission, filepath, row.names = FALSE, quote = FALSE)

    cat("Submission exported to:", filepath, "\n")

    return(filepath)
}

# ---- Post-Processing ----

#' Apply post-processing to predictions
postprocess_predictions <- function(predictions,
                                    clip_min = NULL,
                                    clip_max = NULL,
                                    round_digits = NULL) {
    processed <- predictions

    # Clip to range
    if (!is.null(clip_min)) {
        n_clipped <- sum(processed < clip_min)
        processed[processed < clip_min] <- clip_min
        if (n_clipped > 0) cat("  Clipped", n_clipped, "values to min =", clip_min, "\n")
    }

    if (!is.null(clip_max)) {
        n_clipped <- sum(processed > clip_max)
        processed[processed > clip_max] <- clip_max
        if (n_clipped > 0) cat("  Clipped", n_clipped, "values to max =", clip_max, "\n")
    }

    # Round
    if (!is.null(round_digits)) {
        processed <- round(processed, round_digits)
        cat("  Rounded to", round_digits, "decimal places\n")
    }

    return(processed)
}

# ---- Summary Statistics ----

#' Print prediction summary statistics
print_prediction_summary <- function(predictions) {
    cat("\nPrediction Summary:\n")
    cat("  Count:", length(predictions), "\n")
    cat("  Min:", round(min(predictions), 4), "\n")
    cat("  Max:", round(max(predictions), 4), "\n")
    cat("  Mean:", round(mean(predictions), 4), "\n")
    cat("  Median:", round(median(predictions), 4), "\n")
    cat("  Std Dev:", round(sd(predictions), 4), "\n")

    # Distribution
    cat("\nQuantiles:\n")
    quantiles <- quantile(predictions, probs = c(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    print(round(quantiles, 4))
}

# ---- Complete Submission Pipeline ----

#' Run complete submission pipeline
run_submission_pipeline <- function(predictions, test_df,
                                    sample_submission = NULL,
                                    filename = "submission.csv",
                                    id_col = "ward",
                                    target_col = "target",
                                    postprocess = TRUE) {
    cat("Running submission pipeline...\n\n")

    # Get IDs from test data
    if (id_col %in% names(test_df)) {
        ids <- test_df[[id_col]]
    } else {
        ids <- seq_len(nrow(test_df))
        cat("Warning: ID column not found, using row numbers\n")
    }

    # Post-process if requested
    if (postprocess) {
        cat("Post-processing predictions...\n")
        predictions <- postprocess_predictions(predictions, clip_min = 0)
    }

    # Print summary
    print_prediction_summary(predictions)

    # Create submission
    cat("\nCreating submission...\n")
    submission <- create_submission(predictions, ids, id_col, target_col)

    # Validate if sample provided
    if (!is.null(sample_submission)) {
        cat("\nValidating submission...\n")
        validate_submission(submission, sample_submission, id_col, target_col)
    }

    # Export
    cat("\nExporting submission...\n")
    filepath <- export_submission(submission, filename)

    return(list(
        submission = submission,
        filepath = filepath
    ))
}

cat("\nSubmission functions loaded!\n")
