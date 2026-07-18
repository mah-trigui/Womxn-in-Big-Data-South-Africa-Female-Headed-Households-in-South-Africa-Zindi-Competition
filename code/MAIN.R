# ==============================================================================
# MAIN.R - Main Pipeline Orchestration
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction (Zindi Competition)
# Description: Run the complete ML pipeline from data loading to submission
# ==============================================================================
#
# USAGE:
#   1. Set working directory to the project folder
#   2. Ensure all data files are in the Data/ folder
#   3. Run: source("organized/MAIN.R")
#
# PIPELINE STEPS:
#   1. Load configuration and libraries
#   2. Load raw data
#   3. Feature engineering
#   4. Data preprocessing
#   5. Feature selection (optional)
#   6. Train models (XGBoost, CatBoost, LightGBM)
#   7. Ensemble predictions
#   8. Generate submission
#
# ==============================================================================

cat("\n")
cat("================================================================\n")
cat("   TRAFFIC/TRANSPORTATION TRAVEL TIME PREDICTION PIPELINE\n")
cat("================================================================\n\n")

# ---- Set Working Directory ----
# Uncomment and modify if needed:
# setwd("c:/Users/mtrigui2/Desktop/sss/")

# ---- Source Configuration ----
cat("Loading configuration...\n")
source("00_config.R")

# ---- Load Data ----
cat("\n[1/8] Loading data...\n")
source("01_data_loading.R")

# ---- Feature Engineering ----
cat("\n[2/8] Feature engineering...\n")
source("02_feature_engineering.R")

# Apply to your data as needed:
# df <- engineer_features(df, ward_here, station_near, station_near_b,
#                         traffic, transit_route, routes_public, ag)

# ---- Data Preprocessing ----
cat("\n[3/8] Data preprocessing...\n")
source("03_data_preprocessing.R")

# df <- preprocess_data(df)

# ---- Feature Selection (Optional) ----
cat("\n[4/8] Feature selection...\n")
source("04_feature_selection.R")

# Optional:
# pca_result <- perform_pca(train_data)
# importance <- calc_feature_importance(train_data)

# ---- Prepare Final Training/Test Data ----
cat("\n[5/8] Preparing model data...\n")

# Remove non-feature columns by name, not by index:
# cols_to_drop <- c("ward_id", "ADM3_PCODE", "ADM4_PCODE", "cluster", "transport", "geometry")
# xgb <- df[!is.na(df$target), !names(df) %in% cols_to_drop]
# ts  <- df[is.na(df$target),  !names(df) %in% c(cols_to_drop, "target")]

# ---- Train Models ----
cat("\n[6/8] Training models...\n")

# --- XGBoost ---
cat("\n--- Training XGBoost ---\n")
source("05_model_xgboost.R")

# xgb_result <- run_xgb_pipeline(train_df, test_df, target_col = "target",
#                                 do_cv = TRUE, do_grid_search = FALSE)
# xgb_preds <- xgb_result$predictions

# --- CatBoost ---
cat("\n--- Training CatBoost ---\n")
source("06_model_catboost.R")

# catboost_result <- run_catboost_pipeline(train_df, test_df, target_col = "target",
#                                           do_tuning = FALSE)
# catboost_preds <- catboost_result$predictions

# --- LightGBM ---
cat("\n--- Training LightGBM ---\n")
source("07_model_lightgbm.R")

# lgb_result <- run_lgb_pipeline(train_df, test_df, target_col = "target",
#                                 do_cv = TRUE)
# lgb_preds <- lgb_result$predictions

# ---- Ensemble Models ----
cat("\n[7/8] Ensembling models...\n")
source("08_model_ensemble.R")

# ensemble_result <- run_ensemble_pipeline(xgb_preds, catboost_preds, lgb_preds,
#                                           method = "average")
# final_preds <- ensemble_result$predictions

# ---- Generate Submission ----
cat("\n[8/8] Generating submission...\n")
source("09_submission.R")

# Example usage:
# submission_result <- run_submission_pipeline(
#   predictions = final_preds,
#   test_df = test_df,
#   sample_submission = sample_submission,
#   filename = "submission.csv"
# )

cat("\n")
cat("================================================================\n")
cat("   PIPELINE MODULES LOADED SUCCESSFULLY\n")
cat("================================================================\n")
cat("\nAll functions are now available. To run the full pipeline:\n\n")
cat("1. Prepare your data:\n")
cat("   df <- as.data.table(read.csv('Data/Train.csv'))\n")
cat("   test <- as.data.table(read.csv('Data/Test.csv'))\n\n")
cat("2. Run models:\n")
cat("   xgb_result <- run_xgb_pipeline(train_df, test_df)\n")
cat("   catboost_result <- run_catboost_pipeline(train_df, test_df)\n")
cat("   lgb_result <- run_lgb_pipeline(train_df, test_df)\n\n")
cat("3. Ensemble and submit:\n")
cat("   ensemble <- run_ensemble_pipeline(xgb_result$predictions,\n")
cat("                                      catboost_result$predictions,\n")
cat("                                      lgb_result$predictions)\n")
cat("   run_submission_pipeline(ensemble$predictions, test_df)\n\n")
cat("================================================================\n")
