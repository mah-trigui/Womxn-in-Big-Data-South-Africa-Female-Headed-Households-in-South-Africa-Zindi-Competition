# ==============================================================================
# 03_data_preprocessing.R - Data Preprocessing and Cleaning
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: Clean data, handle missing values, outliers
# ==============================================================================

source("00_config.R")
print_header("Step 3: Data Preprocessing")

# ---- Missing Value Handling ----
cat("Setting up missing value handling functions...\n")

#' Handle missing values for station distances
handle_missing_station_dist <- function(df) {
    # Urban stations - set NA to very large value (200km)
    if ("dist_station_u" %in% names(df)) {
        df$dist_station_u[is.na(df$dist_station_u)] <- 200000
        df$dist_station_u[df$dist_station_u == 200000] <- NA
        df$dist_station_u[is.na(df$dist_station_u)] <- 65000 # Better imputation based on analysis
    }

    # Banlieu stations - set NA to large value (100km)
    if ("dist_station_b" %in% names(df)) {
        df$dist_station_b[is.na(df$dist_station_b)] <- 100000
        df$dist_station_b[df$dist_station_b == 100000] <- NA
        df$dist_station_b[is.na(df$dist_station_b)] <- 35000 # Better imputation based on analysis
    }

    return(df)
}

#' Handle missing values for traffic features
handle_missing_traffic <- function(df) {
    if ("speed_avg" %in% names(df)) {
        df$speed_avg[is.na(df$speed_avg)] <- 120
        df$speed_avg[df$speed_avg == 500] <- NA
        df$speed_avg[is.na(df$speed_avg)] <- 120
    }

    if ("jf_avg" %in% names(df)) {
        df$jf_avg[is.na(df$jf_avg)] <- 5
        df$jf_avg[df$jf_avg == 25] <- NA
        df$jf_avg[is.na(df$jf_avg)] <- 5
    }

    if ("cn_avg" %in% names(df)) {
        df$cn_avg[is.na(df$cn_avg)] <- 0
        df$cn_avg[df$cn_avg == -1] <- NA
        df$cn_avg[is.na(df$cn_avg)] <- 0
    }

    return(df)
}

# ---- Outlier Treatment ----
cat("Setting up outlier treatment functions...\n")

#' Cap outliers at specified quantiles
cap_outliers <- function(df, col, lower_q = 0.01, upper_q = 0.99) {
    if (col %in% names(df)) {
        q_vals <- quantile(df[[col]], c(lower_q, upper_q), na.rm = TRUE)
        df[[col]][df[[col]] < q_vals[1]] <- q_vals[1]
        df[[col]][df[[col]] > q_vals[2]] <- q_vals[2]
    }
    return(df)
}

#' Apply standard outlier capping based on original analysis
apply_standard_outlier_caps <- function(df) {
    # Geo distance features
    if ("geo_dist_mean" %in% names(df)) {
        df$geo_dist_mean[df$geo_dist_mean >= 618] <- 618
    }
    if ("geo_dist_sd" %in% names(df)) {
        df$geo_dist_sd[df$geo_dist_sd >= 4] <- 4
    }
    if ("travelTime" %in% names(df)) {
        df$travelTime[df$travelTime >= 37405] <- 37405
    }
    return(df)
}

# ---- Data Preparation for Modeling ----
cat("Setting up data preparation functions...\n")

#' Prepare data for XGBoost/CatBoost/LightGBM
prepare_model_data <- function(df, cols_to_exclude = NULL, target_col = "target") {
    # Create a copy
    model_df <- copy(df)

    # Remove specified columns
    if (!is.null(cols_to_exclude)) {
        exclude_names <- names(model_df)[cols_to_exclude]
        model_df <- model_df[, !names(model_df) %in% exclude_names, with = FALSE]
    }

    # Separate train and test based on target
    train_data <- model_df[!is.na(model_df[[target_col]]), ]
    test_data <- model_df[is.na(model_df[[target_col]]), ]

    # Remove target from test
    test_data[[target_col]] <- NULL

    return(list(
        train = train_data,
        test = test_data
    ))
}

#' Select important features by name (not by index)
#' Feature names should be determined after running importance analysis
select_important_features <- function(df, feature_names) {
    available <- feature_names[feature_names %in% names(df)]
    if (length(available) < length(feature_names)) {
        warning(
            length(feature_names) - length(available),
            " requested features not found in dataframe"
        )
    }
    return(df[, available, drop = FALSE])
}

#' Add engineered features from other tables
add_engineered_features <- function(base_df, xgb_df) {
    result <- copy(base_df)

    # Add geographic/traffic features if available
    feature_mappings <- c(
        "metro", "clus_1", "clus_2", "geo_dist_mean", "geo_dist_sd",
        "distance", "trafficTime", "dist_station_b", "banlieu",
        "dist_station_u", "urban", "speed_avg", "jf_avg", "cn_avg",
        "train", "public"
    )

    for (feat in feature_mappings) {
        if (feat %in% names(xgb_df) && !(feat %in% names(result))) {
            result[[feat]] <- xgb_df[[feat]]
        }
    }

    return(result)
}

# ---- Master Preprocessing Function ----

#' Apply all preprocessing steps
preprocess_data <- function(df) {
    cat("  Handling missing station distances...\n")
    df <- handle_missing_station_dist(df)

    cat("  Handling missing traffic features...\n")
    df <- handle_missing_traffic(df)

    cat("  Applying outlier caps...\n")
    df <- apply_standard_outlier_caps(df)

    return(df)
}

# ---- Standard Column Exclusion ----
# Based on original analysis, these columns should be excluded:
# Columns: 1 (ward/id), 3 (some categorical), 51:53 (geo codes),
#          57 (cluster), 64 (transport), 72 (geometry)

get_standard_exclusions <- function() {
    # Return column indices to exclude (adjust based on actual data)
    return(c(1, 3, 51, 52, 53, 57, 64, 72))
}

cat("\nData preprocessing functions loaded!\n")
