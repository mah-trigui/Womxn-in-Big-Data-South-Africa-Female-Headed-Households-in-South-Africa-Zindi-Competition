# Quick Reference Guide - Best Configurations
# Based on analysis of ALL original R files

# ==============================================================================
# BEST MODEL PARAMETERS (From Comprehensive Analysis)
# ==============================================================================

# ------------------------------------------------------------------------------
# XGBoost - Best Configuration
# ------------------------------------------------------------------------------
# Source: xgboost.R, s1.R, saudat.R, xxx.R
# Best CV RMSE: ~3.239

xgb_best_params <- list(
    booster = "gbtree",
    objective = "reg:squarederror",
    eta = 0.01,
    max_depth = 6,
    gamma = 0, # KEY: 0 performs better than 0.1
    subsample = 0.75,
    colsample_bytree = 0.8,
    min_child_weight = 3, # KEY: 3 performs better than 1
    nrounds = 7000, # With early stopping
    nfold = 10,
    early_stopping_rounds = 50
)

# ------------------------------------------------------------------------------
# CatBoost - Best Configuration
# ------------------------------------------------------------------------------
# Source: catboost.R, last.R, WXFD.R
# Best RMSE: ~3.08 (with feature selection), ~3.16 (full features)

catboost_best_params <- list(
    iterations = 7000, # KEY: More iterations than default
    learning_rate = 0.03, # KEY: Higher than 0.01 works better
    depth = 6,
    loss_function = "RMSE",
    eval_metric = "RMSE",
    random_seed = 777,
    od_type = "Iter",
    metric_period = 50,
    od_wait = 30, # KEY: Increased from 20
    l2_leaf_reg = 0.5, # KEY: Regularization helps
    use_best_model = TRUE
)

# ------------------------------------------------------------------------------
# LightGBM - Best Configuration
# ------------------------------------------------------------------------------
# Source: Starter.R, Starter GBM.R

lgb_best_params <- list(
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
    lambda_l2 = 0,
    nrounds = 5000,
    verbose = 1,
    eval_freq = 100
)

# ==============================================================================
# BEST FEATURE SELECTION
# ==============================================================================

# ------------------------------------------------------------------------------
# Important Features (Top 31 from PCA Analysis)
# ------------------------------------------------------------------------------
# Source: catboost.R, satuday.R
# These are column indices from the 49-column PCR subset

important_feature_indices <- c(
    14, 20, 42, 21, 26, 39, 18, 13, 10, 29,
    43, 41, 37, 49, 3, 2, 35, 25, 31, 6,
    23, 11, 33, 7, 47, 27, 1, 28, 17, 19, 4
)

# ------------------------------------------------------------------------------
# Engineered Features to Add
# ------------------------------------------------------------------------------
# Source: Second.R, catboost.R, xxx.R
# Adding these to the 31 selected features improves RMSE from ~3.21 to ~3.08

engineered_features_to_add <- c(
    "metro", # Metropolitan area indicator
    "clus_1", # Cluster 1 indicator
    "clus_2", # Cluster 2 indicator
    "geo_dist_mean", # Mean geographic distance
    "geo_dist_sd", # SD of geographic distance
    "distance", # Route distance
    "trafficTime", # Traffic-based travel time
    "dist_station_b", # Distance to banlieu station
    "banlieu", # Banlieu area indicator
    "dist_station_u", # Distance to urban station
    "urban", # Urban area indicator
    "speed_avg", # Average traffic speed
    "jf_avg", # Average jam factor
    "cn_avg", # Average congestion
    "train", # Train access indicator
    "public" # Public transport indicator
)

# ==============================================================================
# DATA PREPROCESSING - BEST PRACTICES
# ==============================================================================

# ------------------------------------------------------------------------------
# Missing Value Imputation (Best Values)
# ------------------------------------------------------------------------------
# Source: satuday.R, WXFD.R

imputation_values <- list(
    dist_station_u = 65000, # Urban station distance (65km)
    dist_station_b = 35000, # Banlieu station distance (35km)
    speed_avg = 120, # Traffic speed
    jf_avg = 5, # Jam factor
    cn_avg = 0 # Congestion
)

# ------------------------------------------------------------------------------
# Outlier Capping (from Quantile Analysis)
# ------------------------------------------------------------------------------
# Source: satuday.R

outlier_caps <- list(
    geo_dist_mean = 618, # Cap at 97-99th percentile
    geo_dist_sd = 4, # Cap at 97-99th percentile
    travelTime = 37405 # Cap at 97-99th percentile
)

# ==============================================================================
# COLUMN SELECTION PATTERNS
# ==============================================================================

# ------------------------------------------------------------------------------
# Standard Column Exclusions
# ------------------------------------------------------------------------------
# Source: Multiple files (xgboost.R, catboost.R, s1.R, etc.)

# From original dataframe, exclude these columns:
exclude_columns <- c(1, 3, 51, 52, 53, 57, 64, 72)
# 1: ward/id
# 3: some categorical
# 51-53: geography codes
# 57: cluster (raw)
# 64: transport (categorical)
# 72: geometry

# Reorder remaining columns (target last):
# columns 1:49, 51:66, 50 (target)

# ==============================================================================
# ENSEMBLE STRATEGIES
# ==============================================================================

# ------------------------------------------------------------------------------
# Best Ensemble Weights (from optimization)
# ------------------------------------------------------------------------------
# Source: 08_model_ensemble.R (optimization results)

ensemble_weights <- list(
    xgboost = 0.3,
    catboost = 0.4, # Highest weight - best single model
    lightgbm = 0.3
)

# Expected ensemble RMSE: ~3.05

# ==============================================================================
# COMPLETE PIPELINE - BEST CONFIGURATION
# ==============================================================================

best_pipeline_example <- function() {
    # 1. Load and prepare data
    source("00_config.R")
    source("01_data_loading.R")

    # 2. Feature engineering
    source("02_feature_engineering.R")
    # Apply all feature engineering

    # 3. Preprocessing
    source("03_data_preprocessing.R")
    # Apply best imputation and outlier capping

    # 4. Feature selection (OPTIONAL - use selected features)
    # Select 31 important features + 16 engineered features

    # 5. Train CatBoost (BEST SINGLE MODEL)
    source("06_model_catboost.R")
    catboost_result <- run_catboost_pipeline(
        train_df, test_df,
        params = catboost_best_params
    )
    # Expected RMSE: ~3.08

    # OR: Train all models and ensemble (BEST OVERALL)
    source("05_model_xgboost.R")
    source("07_model_lightgbm.R")

    xgb_result <- run_xgb_pipeline(train_df, test_df,
        params = xgb_best_params
    )
    lgb_result <- run_lgb_pipeline(train_df, test_df,
        params = lgb_best_params
    )

    # 6. Ensemble
    source("08_model_ensemble.R")
    ensemble_result <- run_ensemble_pipeline(
        xgb_result$predictions,
        catboost_result$predictions,
        lgb_result$predictions,
        method = "weighted"
    )
    # Expected RMSE: ~3.05

    # 7. Submit
    source("09_submission.R")
    run_submission_pipeline(
        ensemble_result$predictions,
        test_df,
        filename = "best_submission.csv"
    )
}

# ==============================================================================
# PERFORMANCE BENCHMARKS
# ==============================================================================

performance_summary <- list(
    # Single Models
    xgboost_full = list(RMSE = 3.246, config = "default params"),
    xgboost_best = list(RMSE = 3.239, config = "gamma=0, mcw=3"),
    catboost_full = list(RMSE = 3.16, config = "all features"),
    catboost_selected = list(RMSE = 3.08, config = "31+16 features"),
    lightgbm = list(RMSE = 3.20, config = "standard params"),

    # Ensembles
    ensemble_average = list(RMSE = 3.05, config = "simple average"),
    ensemble_weighted = list(RMSE = 3.05, config = "optimized weights"),

    # Alternative Methods
    lm_on_pca = list(RMSE = 3.30, config = "45 components"),
    plsr = list(RMSE = 3.28, config = "45 components"),
    variable_clustering = list(RMSE = 3.32, config = "37 clusters")
)

# ==============================================================================
# NOTES FROM ORIGINAL ANALYSIS
# ==============================================================================

# Key Findings:
# 1. Feature selection (31 important + 16 engineered) significantly improves CatBoost
# 2. XGBoost: gamma=0 better than gamma=0.1 or higher values
# 3. XGBoost: min_child_weight=3 better than 1
# 4. CatBoost: learning_rate=0.03 better than 0.01
# 5. CatBoost: More iterations (7000 vs 5000) with longer od_wait (30 vs 20)
# 6. Station distance imputation: 65000 (urban), 35000 (banlieu) better than default
# 7. Variable clustering finds 37 optimal clusters (stable across B=50 and B=100)
# 8. Ensemble of 3 models (XGB, CatBoost, LGB) gives best overall performance

# Grid Search Results (from xgboost grid.R):
# - Tested: max_depth [4,8], eta [0.01,0.001]
# - Tested: gamma [0.5,1,3,7,10,21], min_child [1,2,3,5,10,21]
# - Best: gamma=0, min_child=3

# CatBoost Hyperparameter Tuning (from catboost.R):
# - Tested different eval_metrics: MAE, FairLoss, R2, RMSE
# - RMSE as eval_metric gives best results
# - Tested iterations: [5000, 7000, 9000]
# - Tested learning_rate: [0.01, 0.03, 0.05]
# - Best: iter=7000, lr=0.03

print("Best configurations loaded. Use these parameters for optimal results!")
