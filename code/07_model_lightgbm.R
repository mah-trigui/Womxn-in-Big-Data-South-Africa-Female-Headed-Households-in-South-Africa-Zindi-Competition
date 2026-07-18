# ==============================================================================
# 07_model_lightgbm.R - LightGBM Model Training and Prediction
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: LightGBM model with cross-validation
# ==============================================================================

source("00_config.R")
print_header("Step 7: LightGBM Modeling")

# ---- LightGBM Data Preparation ----

#' Prepare data for LightGBM
prepare_lgb_data <- function(train_df, test_df = NULL, target_col = "target") {
    # Separate features and target
    y_train <- train_df[[target_col]]
    X_train <- train_df[, !names(train_df) %in% target_col, with = FALSE]

    # Convert to matrix
    train_matrix <- as.matrix(X_train)

    # Create LightGBM dataset
    lgb_train <- lgb.Dataset(data = train_matrix, label = y_train)

    result <- list(
        lgb_train = lgb_train,
        train_matrix = train_matrix,
        y_train = y_train
    )

    # Prepare test data if provided
    if (!is.null(test_df)) {
        X_test <- test_df[, !names(test_df) %in% target_col, with = FALSE]
        test_matrix <- as.matrix(X_test)
        result$test_matrix <- test_matrix
    }

    return(result)
}

# ---- Label Encoding for Categorical Variables ----

#' Encode character columns as numeric
encode_categorical <- function(df) {
    char_cols <- names(df)[sapply(df, is.character)]

    if (length(char_cols) > 0) {
        cat("Encoding", length(char_cols), "categorical columns...\n")
        for (col in char_cols) {
            df[[col]] <- as.numeric(as.factor(df[[col]])) - 1
        }
    }

    return(df)
}

# ---- Model Training ----

#' Train LightGBM model
train_lgb_model <- function(lgb_train, params = NULL, nrounds = 5000,
                            verbose = 1, eval_freq = 100) {
    if (is.null(params)) {
        params <- LIGHTGBM_PARAMS
    }

    cat("Training LightGBM model...\n")
    cat("  Rounds:", nrounds, "\n")
    cat("  Learning rate:", params$learning_rate, "\n")
    cat("  Num leaves:", params$num_leaves, "\n")

    model <- lgb.train(
        params = params,
        data = lgb_train,
        nrounds = nrounds,
        verbose = verbose,
        eval_freq = eval_freq
    )

    return(model)
}

# ---- Cross-Validation ----

#' Run LightGBM cross-validation
run_lgb_cv <- function(lgb_train, params = NULL, nrounds = 5000,
                       nfold = 5, early_stop = 50) {
    if (is.null(params)) {
        params <- LIGHTGBM_PARAMS
    }

    cat("Running LightGBM cross-validation...\n")

    cv_result <- lgb.cv(
        params = params,
        data = lgb_train,
        nrounds = nrounds,
        nfold = nfold,
        early_stopping_rounds = early_stop,
        verbose = 1,
        eval_freq = 100
    )

    best_iter <- cv_result$best_iter
    best_score <- cv_result$best_score

    cat("\n  Best iteration:", best_iter, "\n")
    cat("  Best CV score:", round(best_score, 4), "\n")

    return(list(
        cv_result = cv_result,
        best_iter = best_iter,
        best_score = best_score
    ))
}

# ---- Prediction ----

#' Make predictions with LightGBM model
predict_lgb <- function(model, test_matrix) {
    predictions <- predict(model, test_matrix)
    return(predictions)
}

# ---- Feature Importance ----

#' Get LightGBM feature importance
get_lgb_importance <- function(model, top_n = 20) {
    importance <- lgb.importance(model)

    cat("\nTop", top_n, "important features:\n")
    print(head(importance, top_n))

    return(importance)
}

# ---- Complete LightGBM Pipeline ----

#' Run complete LightGBM pipeline
run_lgb_pipeline <- function(train_df, test_df, target_col = "target",
                             params = NULL, do_cv = TRUE) {
    cat("Starting LightGBM pipeline...\n\n")

    # Encode categorical variables
    cat("Encoding categorical variables...\n")
    train_df <- encode_categorical(train_df)
    test_df <- encode_categorical(test_df)

    # Prepare data
    cat("Preparing data...\n")
    data <- prepare_lgb_data(train_df, test_df, target_col)

    # Use provided params or defaults
    if (is.null(params)) {
        params <- LIGHTGBM_PARAMS
    }

    # Determine number of rounds
    nrounds <- 5000
    if (do_cv) {
        cat("\nRunning cross-validation...\n")
        cv <- run_lgb_cv(data$lgb_train, params)
        nrounds <- cv$best_iter
    }

    # Train model
    cat("\nTraining final model...\n")
    model <- train_lgb_model(data$lgb_train, params, nrounds)

    # Make predictions
    cat("\nMaking predictions...\n")
    predictions <- predict_lgb(model, data$test_matrix)

    # Get importance
    importance <- get_lgb_importance(model)

    return(list(
        model = model,
        predictions = predictions,
        importance = importance,
        params = params
    ))
}

cat("\nLightGBM modeling functions loaded!\n")
