# ==============================================================================
# 01_data_loading.R - Data Loading and Initial Processing
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: Load train, test data and geospatial files
# ==============================================================================

source("00_config.R")
print_header("Step 1: Loading Data")

# ---- Load Main Datasets ----
cat("Loading train and test datasets...\n")

# Training data
train_raw <- as.data.table(read.csv(paste0(DATA_DIR, "Train.csv")))
cat("  Train dimensions:", dim(train_raw), "\n")

# Test data
test_raw <- as.data.table(read.csv(paste0(DATA_DIR, "Test.csv")))
cat("  Test dimensions:", dim(test_raw), "\n")

# Sample submission
sample_submission <- fread(paste0(DATA_DIR, "SampleSubmission.csv"))
cat("  Sample submission dimensions:", dim(sample_submission), "\n")

# ---- Load Geospatial Data ----
cat("\nLoading geospatial data...\n")

# Road segments (if available)
if (file.exists("road_segments.dbf") && file.exists("road_segments.shp")) {
    road_dbf <- read.dbf("road_segments.dbf")
    road_sf <- sf::read_sf("road_segments.shp")
    road_data <- as.data.table(road_dbf)
    cat("  Road segments loaded:", nrow(road_data), "records\n")
} else {
    cat("  Road segment files not found, skipping...\n")
    road_data <- NULL
}

# ---- Load VDS (Vehicle Detection System) Data ----
cat("\nLoading VDS hourly data...\n")

# Check if VDS files exist
vds_files <- c(
    "WC September 2018 Hourly.csv", "WC October 2018 Hourly.csv",
    "WC November 2018 Hourly.csv", "WC December 2018 Hourly.csv"
)

if (all(file.exists(vds_files))) {
    vds_09_18 <- fread("WC September 2018 Hourly.csv")
    vds_10_18 <- fread("WC October 2018 Hourly.csv")
    vds_11_18 <- fread("WC November 2018 Hourly.csv")
    vds_12_18 <- fread("WC December 2018 Hourly.csv")

    cat("  September 2018:", nrow(vds_09_18), "records\n")
    cat("  October 2018:", nrow(vds_10_18), "records\n")
    cat("  November 2018:", nrow(vds_11_18), "records\n")
    cat("  December 2018:", nrow(vds_12_18), "records\n")

    # Combine VDS data
    vds_combined <- rbind(vds_09_18, vds_10_18, vds_11_18, vds_12_18)
    cat("  Combined VDS records:", nrow(vds_combined), "\n")
} else {
    cat("  VDS files not found, skipping...\n")
    vds_combined <- NULL
}

# ---- Initial Data Inspection ----
cat("\n--- Train Data Summary ---\n")
cat("Columns:", ncol(train_raw), "\n")
cat("Target variable stats:\n")
cat("  Min:", min(train_raw$target, na.rm = TRUE), "\n")
cat("  Max:", max(train_raw$target, na.rm = TRUE), "\n")
cat("  Mean:", round(mean(train_raw$target, na.rm = TRUE), 2), "\n")
cat("  Median:", round(median(train_raw$target, na.rm = TRUE), 2), "\n")

# ---- Check for Missing Values ----
cat("\n--- Missing Values Check ---\n")
train_na_count <- colSums(is.na(train_raw))
cols_with_na <- train_na_count[train_na_count > 0]
if (length(cols_with_na) > 0) {
    cat("Columns with missing values:\n")
    print(cols_with_na)
} else {
    cat("No missing values in training data\n")
}

# ---- Save Loaded Data ----
cat("\nSaving loaded data...\n")
saveRDS(train_raw, paste0(OUTPUT_DIR, "train_raw.rds"))
saveRDS(test_raw, paste0(OUTPUT_DIR, "test_raw.rds"))
if (!is.null(vds_combined)) {
    saveRDS(vds_combined, paste0(OUTPUT_DIR, "vds_combined.rds"))
}
if (!is.null(road_data)) {
    saveRDS(road_data, paste0(OUTPUT_DIR, "road_data.rds"))
}

cat("\nData loading complete!\n")
