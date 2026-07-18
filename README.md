# Travel Time Prediction — Ward-Level Regression

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Wazihub Soil Moisture Prediction 🌾 - $8 000 USD](https://zindi.africa/competitions/wazihub-soil-moisture-prediction-challenge)

Ranked in the TOP 37%
---

Predict travel time across urban wards/regions using multi-source geospatial and traffic data.
Zindi ML Competition — Regression, evaluated on RMSE.

---

## Key Engineering Decisions

### 1. Multi-source feature fusion
Road network attributes, VDS traffic sensor data (speed, jam factor, vehicle counts), transit route proximity, and weather station readings were all joined to the base ward-level grid.

### 2. Station proximity classification
Wards were classified as urban or banlieu based on distance thresholds to transit stations, creating binary accessibility indicators that captured different mobility patterns.

### 3. Geographic distance aggregation
Spatial variation within each ward was summarized as mean and standard deviation of GPS trace distances, encoding geographic spread as a model feature.

### 4. Variable clustering for dimensionality reduction
`ClustOfVar` (hierarchical variable clustering with stability analysis) was used to group correlated features before modeling, reducing noise while preserving interpretable structure.

### 5. Three-model ensemble (XGBoost + CatBoost + LightGBM)
All three gradient boosting frameworks were trained independently and blended. A `BEST_CONFIG.R` file consolidates the best hyperparameters found across all experiments as a single reference.

---

## Project Structure

```
├── 00_config.R              # Libraries, paths, model parameters
├── 01_data_loading.R        # Load train, test, geospatial, traffic, transit data
├── 02_feature_engineering.R # Geographic, traffic, transit, cluster features
├── 03_data_preprocessing.R  # Missing values, outlier caps, data preparation
├── 04_feature_selection.R   # PCA, PLSR, variable clustering, stepwise
├── 05_model_xgboost.R       # XGBoost training, CV, grid search
├── 06_model_catboost.R      # CatBoost training, tuning
├── 07_model_lightgbm.R      # LightGBM training, CV
├── 08_model_ensemble.R      # Averaging, weighted blending, OOF stacking
├── 09_submission.R          # Submission file generation
├── BEST_CONFIG.R            # Best hyperparameters from all experiments
└── MAIN.R                   # Full pipeline orchestration
```

---

## Technical Stack

- **Language**: R
- **Models**: xgboost, catboost, lightgbm
- **Geospatial**: sf, rgdal, geosphere
- **Dimensionality reduction**: pls (PLSR), ClustOfVar
- **Utilities**: data.table, dplyr, caret, mice

---

## How to Run

```r
source("MAIN.R")
```

Requires `Train.csv`, `Test.csv`, and geospatial/traffic data files in the working directory.

---

## Scope

Competition data is not included. The repository shares the pipeline structure and feature engineering approach for a multi-source geospatial regression problem.

Place these files in the root folder:
- `road_segments.dbf` - Road network data
- `road_segments.shp` - Road network shapefile
- `WC September 2018 Hourly.csv` - VDS data
- `WC October 2018 Hourly.csv` - VDS data
- `WC November 2018 Hourly.csv` - VDS data
- `WC December 2018 Hourly.csv` - VDS data

## Installation

### Required R Version
R >= 4.0.0

### Install Required Packages

```r
# Core packages
install.packages(c(
  "data.table", "dplyr", "tidyr", "tidyverse", "sqldf", "Matrix"
))

# Machine Learning
install.packages(c(
  "caret", "xgboost", "lightgbm"
))

# CatBoost (requires special installation)
# devtools::install_github("catboost/catboost", subdir = "catboost/R-package")

# Feature Engineering
install.packages(c(
  "pls", "ClustOfVar", "mice", "ROSE", "DMwR"
))

# Geospatial
install.packages(c(
  "sf", "rgdal", "foreign", "geosphere", "pracma"
))

# Visualization
install.packages(c(
  "ggplot2", "corrplot", "ggcorrplot"
))

# Utilities
install.packages(c(
  "chron", "Information", "Metrics", "olsrr", "MASS", "doParallel"
))
```

## Quick Start

### Option 1: Run Full Pipeline

```r
# Set working directory
setwd("path/to/project")

# Run main pipeline
source("organized/MAIN.R")
```

### Option 2: Step-by-Step Execution

```r
# 1. Load configuration
source("organized/00_config.R")

# 2. Load data
source("organized/01_data_loading.R")

# 3. Prepare training data
xgb <- df[!is.na(df$target), ]
xgb <- xgb[, -c(1, 3, 51:53, 57, 64, 72)]
xgb <- xgb[, c(1:49, 51:66, 50)]

# 4. Prepare test data
ts <- df[is.na(df$target), ]
ts <- ts[, -c(1, 3, 51:53, 57, 64, 72)]
ts$target <- NULL

# 5. Train XGBoost
source("organized/05_model_xgboost.R")
xgb_result <- run_xgb_pipeline(xgb, ts, do_cv = TRUE)

# 6. Train CatBoost
source("organized/06_model_catboost.R")
catboost_result <- run_catboost_pipeline(xgb, ts)

# 7. Train LightGBM
source("organized/07_model_lightgbm.R")
lgb_result <- run_lgb_pipeline(xgb, ts, do_cv = TRUE)

# 8. Ensemble predictions
source("organized/08_model_ensemble.R")
ensemble <- run_ensemble_pipeline(
  xgb_result$predictions,
  catboost_result$predictions,
  lgb_result$predictions,
  method = "average"
)

# 9. Generate submission
source("organized/09_submission.R")
run_submission_pipeline(ensemble$predictions, ts)
```

## Model Parameters

### XGBoost (Best Found)
```r
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.01,
  max_depth = 6,
  gamma = 0,
  subsample = 0.75,
  colsample_bytree = 0.8,
  min_child_weight = 3
)
# Best CV RMSE: ~3.24
```

### CatBoost (Best Found)
```r
params <- list(
  iterations = 7000,
  learning_rate = 0.03,
  depth = 6,
  loss_function = 'RMSE',
  l2_leaf_reg = 0.5
)
# Best RMSE: ~3.08
```

### LightGBM (Best Found)
```r
params <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.008,
  num_leaves = 400,
  feature_fraction = 0.7,
  bagging_fraction = 0.7,
  min_data_in_leaf = 200
)
```

## Feature Engineering

### Geographic Features
- `geo_dist_mean` - Mean geographic distance within ward
- `geo_dist_sd` - Standard deviation of geographic distance

### Traffic Features
- `speed_avg` - Average traffic speed
- `jf_avg` - Average jam factor
- `cn_avg` - Average congestion

### Transit Features
- `dist_station_u` - Distance to nearest urban station
- `dist_station_b` - Distance to nearest banlieu station
- `urban` - Binary: is urban area
- `banlieu` - Binary: is banlieu area
- `train` - Binary: has train access
- `public` - Binary: has public transport

### Cluster Features
- `clus_1` - Cluster 1 indicator
- `clus_2` - Cluster 2 indicator

## Important Feature Indices

Based on importance analysis, these are the most predictive features (column indices):
```r
c(14, 20, 42, 21, 26, 39, 18, 13, 10, 29, 43, 41, 37, 49,
  3, 2, 35, 25, 31, 6, 23, 11, 33, 7, 47, 27, 1, 28, 17, 19, 4)
```

## Output Files

| File | Description |
|------|-------------|
| `output/submission.csv` | Final submission file |
| `output/train_raw.rds` | Cached training data |
| `output/test_raw.rds` | Cached test data |

## Troubleshooting

### Memory Issues
```r
# Clear memory
rm(list = ls())
gc()

# Process in chunks if needed
```

### Package Installation Errors
```r
# For CatBoost on Windows
install.packages("devtools")
devtools::install_url('https://github.com/catboost/catboost/releases/download/v1.0.4/catboost-R-Windows-1.0.4.tgz', INSTALL_opts = c("--no-multiarch"))
```

### Geospatial Package Issues
```r
# Install sf dependencies first
install.packages("sf", dependencies = TRUE)
```

## Model Performance Summary

| Model | CV RMSE | Notes |
|-------|---------|-------|
| XGBoost | ~3.24 | Robust baseline |
| CatBoost | ~3.08 | Best single model |
| LightGBM | ~3.20 | Fast training |
| Ensemble | ~3.05 | Best overall |

## License

This project is for the Zindi competition.
