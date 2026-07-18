# ==============================================================================
# 00_config.R - Configuration and Library Loading
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction (Zindi Competition)
# Description: Load required libraries and set global configuration
# ==============================================================================

# ---- Set Global Options ----
options(scipen = 999)
options(sqldf.driver = "RSQLite")

# ---- Set Random Seed for Reproducibility ----
GLOBAL_SEED <- 777
set.seed(GLOBAL_SEED)

# ---- Required Libraries ----
cat("Loading required libraries...\n")

# Data manipulation
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(tidyverse)
    library(sqldf)
    library(Matrix)
})

# Machine Learning
suppressPackageStartupMessages({
    library(caret)
    library(xgboost)
    library(catboost)
    library(lightgbm)
    library(mlbench)
    library(caretEnsemble) # For model stacking
})

# Feature Engineering
suppressPackageStartupMessages({
    library(splitstackshape) # Sampling
    library(ROSE) # Imbalanced data
    library(DMwR) # SMOTE
    library(mice) # Missing value imputation
})

# Dimensionality Reduction
suppressPackageStartupMessages({
    library(pls) # PCA/PLSR
    library(ClustOfVar) # Variable clustering
})

# Geospatial
suppressPackageStartupMessages({
    library(sf) # Spatial features
    library(rgdal) # Shape files
    library(foreign) # DBF files
    library(geosphere) # Distance calculations
    library(pracma) # Mathematical functions
    library(googleway) # Google Maps API
    library(prettymapr)
})

# Visualization
suppressPackageStartupMessages({
    library(ggplot2)
    library(corrplot)
    library(ggcorrplot)
    library(gridExtra)
})

# Utilities
suppressPackageStartupMessages({
    library(chron) # Weekend detection
    library(Information) # Information value
    library(Metrics) # Evaluation metrics
    library(olsrr) # OLS regression utilities
    library(MASS) # stepAIC
})

# Parallel processing
suppressPackageStartupMessages({
    library(doParallel)
})

# ---- Directory Configuration ----
DATA_DIR <- "Data/"
OUTPUT_DIR <- "output/"

# Create output directory if not exists
if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
}

# ---- Column Configuration ----
# Columns to exclude during modeling (IDs, geography codes, etc.)
COLS_TO_EXCLUDE <- c(1, 3, 51:54) # Adjust based on your data structure
TARGET_COL <- "target"

# ---- Model Parameters ----
TRAIN_RATIO <- 0.80
CV_FOLDS <- 10

# XGBoost default parameters (best found from grid search)
XGBOOST_PARAMS <- list(
    booster = "gbtree",
    objective = "reg:squarederror",
    eta = 0.01,
    max_depth = 6,
    gamma = 0,
    subsample = 0.75,
    colsample_bytree = 0.8,
    min_child_weight = 3
)

# CatBoost default parameters (best found from tuning)
CATBOOST_PARAMS <- list(
    iterations = 7000,
    learning_rate = 0.03,
    depth = 6,
    loss_function = "RMSE",
    eval_metric = "RMSE",
    random_seed = GLOBAL_SEED,
    od_type = "Iter",
    metric_period = 50,
    od_wait = 30,
    l2_leaf_reg = 0.5,
    use_best_model = TRUE
)

# LightGBM default parameters
LIGHTGBM_PARAMS <- list(
    boosting_type = "gbdt",
    objective = "regression",
    metric = "rmse",
    learning_rate = 0.008,
    num_leaves = 400,
    min_gain_to_split = 0,
    feature_fraction = 0.7,
    bagging_freq = 1,
    bagging_fraction = 0.7,
    min_data_in_leaf = 200,
    lambda_l1 = 0,
    lambda_l2 = 0
)

# ---- Helper Functions ----

#' Print section header
print_header <- function(text) {
    cat("\n", paste(rep("=", 60), collapse = ""), "\n")
    cat(" ", text, "\n")
    cat(paste(rep("=", 60), collapse = ""), "\n\n")
}

#' Calculate RMSE
calc_rmse <- function(actual, predicted) {
    sqrt(mean((actual - predicted)^2))
}

#' Calculate MAE
calc_mae <- function(actual, predicted) {
    mean(abs(actual - predicted))
}

#' Create train/test split
create_split <- function(data, target_col, train_ratio = 0.8, seed = GLOBAL_SEED) {
    set.seed(seed)
    train_idx <- createDataPartition(data[[target_col]], p = train_ratio, list = FALSE)
    list(
        train = data[train_idx, ],
        test = data[-train_idx, ]
    )
}

cat("Configuration loaded successfully!\n")
cat("Global seed:", GLOBAL_SEED, "\n")
cat("Train ratio:", TRAIN_RATIO, "\n")
cat("CV folds:", CV_FOLDS, "\n\n")
cat("Note: All libraries loaded with suppressed messages.\n")
cat("      If you encounter package errors, install them using:\n")
cat("      install.packages('package_name')\n")
