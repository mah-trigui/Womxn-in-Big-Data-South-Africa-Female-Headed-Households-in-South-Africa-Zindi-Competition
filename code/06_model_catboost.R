# ==============================================================================
# 06_model_catboost.R - CatBoost Model Training and Prediction
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: CatBoost model with hyperparameter tuning
# ==============================================================================

source("00_config.R")
print_header("Step 6: CatBoost Modeling")

# ---- CatBoost Data Preparation ----

#' Prepare data for CatBoost
prepare_catboost_data <- function(train_df, test_df = NULL, target_col = "target",
                                  validation_split = 0.2) {
    # Create validation split
    set.seed(GLOBAL_SEED)
    val_idx <- createDataPartition(train_df[[target_col]], p = 1 - validation_split, list = FALSE)

    train_set <- train_df[val_idx, ]
    valid_set <- train_df[-val_idx, ]

    # Separate features and target
    y_train <- unlist(train_set[[target_col]])
    X_train <- train_set %>% select(-all_of(target_col))

    y_valid <- unlist(valid_set[[target_col]])
    X_valid <- valid_set %>% select(-all_of(target_col))

    # Create CatBoost pools
    train_pool <- catboost.load_pool(data = X_train, label = y_train)
    valid_pool <- catboost.load_pool(data = X_valid, label = y_valid)

    result <- list(
        train_pool = train_pool,
        valid_pool = valid_pool,
        y_train = y_train,
        y_valid = y_valid,
        X_train = X_train,
        X_valid = X_valid
    )

    # Prepare test pool if provided
    if (!is.null(test_df)) {
        X_test <- test_df %>% select(-any_of(target_col))
        test_pool <- catboost.load_pool(data = X_test)
        result$test_pool <- test_pool
        result$X_test <- X_test
    }

    return(result)
}

# ---- Model Training ----

#' Train CatBoost model
train_catboost_model <- function(train_pool, valid_pool = NULL, params = NULL) {
    if (is.null(params)) {
        params <- CATBOOST_PARAMS
    }

    cat("Training CatBoost model...\n")
    cat("  Iterations:", params$iterations, "\n")
    cat("  Learning rate:", params$learning_rate, "\n")
    cat("  Depth:", params$depth, "\n")

    if (!is.null(valid_pool)) {
        model <- catboost.train(
            learn_pool = train_pool,
            test_pool = valid_pool,
            params = params
        )
    } else {
        model <- catboost.train(
            learn_pool = train_pool,
            params = params
        )
    }

    return(model)
}

# ---- Hyperparameter Tuning ----

#' Test different CatBoost configurations
tune_catboost <- function(train_pool, valid_pool, y_valid,
                          iterations_list = c(3000, 5000, 7000),
                          lr_list = c(0.01, 0.03, 0.05),
                          depth_list = c(4, 6, 8)) {
    best_score <- Inf
    best_params <- NULL

    total_combos <- length(iterations_list) * length(lr_list) * length(depth_list)
    cat("Testing", total_combos, "parameter combinations...\n")

    count <- 1
    for (iter in iterations_list) {
        for (lr in lr_list) {
            for (depth in depth_list) {
                params <- list(
                    iterations = iter,
                    learning_rate = lr,
                    depth = depth,
                    loss_function = "RMSE",
                    eval_metric = "RMSE",
                    random_seed = GLOBAL_SEED,
                    od_type = "Iter",
                    metric_period = 100,
                    od_wait = 30,
                    use_best_model = TRUE
                )

                model <- catboost.train(
                    learn_pool = train_pool,
                    test_pool = valid_pool,
                    params = params
                )

                # Evaluate
                preds <- catboost.predict(model, valid_pool)
                rmse <- sqrt(mean((preds - y_valid)^2))

                if (rmse < best_score) {
                    best_score <- rmse
                    best_params <- params
                    cat(
                        "  [", count, "/", total_combos, "] New best RMSE:",
                        round(rmse, 4), " (iter=", iter, ", lr=", lr, ", depth=", depth, ")\n"
                    )
                }

                count <- count + 1
            }
        }
    }

    cat("\nBest validation RMSE:", round(best_score, 4), "\n")

    return(list(
        best_params = best_params,
        best_score = best_score
    ))
}

# ---- Prediction ----

#' Make predictions with CatBoost model
predict_catboost <- function(model, test_pool) {
    predictions <- catboost.predict(model, test_pool)
    return(predictions)
}

# ---- Model Evaluation ----

#' Evaluate CatBoost model
evaluate_catboost <- function(model, valid_pool, y_valid) {
    preds <- catboost.predict(model, valid_pool)

    metrics <- list(
        RMSE = sqrt(mean((preds - y_valid)^2)),
        MAE = mean(abs(preds - y_valid)),
        R2 = 1 - sum((y_valid - preds)^2) / sum((y_valid - mean(y_valid))^2)
    )

    cat("\nModel Evaluation:\n")
    cat("  RMSE:", round(metrics$RMSE, 4), "\n")
    cat("  MAE:", round(metrics$MAE, 4), "\n")
    cat("  R-squared:", round(metrics$R2, 4), "\n")

    # Also using caret's postResample
    caret_metrics <- postResample(preds, y_valid)
    print(caret_metrics)

    return(metrics)
}

# ---- Complete CatBoost Pipeline ----

#' Run complete CatBoost pipeline
run_catboost_pipeline <- function(train_df, test_df, target_col = "target",
                                  params = NULL, do_tuning = FALSE) {
    cat("Starting CatBoost pipeline...\n\n")

    # Prepare data
    cat("Preparing data...\n")
    data <- prepare_catboost_data(train_df, test_df, target_col)

    # Tune if requested
    if (do_tuning) {
        cat("\nTuning hyperparameters...\n")
        tune_result <- tune_catboost(
            data$train_pool,
            data$valid_pool,
            data$y_valid
        )
        params <- tune_result$best_params
    }

    # Use provided params or defaults
    if (is.null(params)) {
        params <- CATBOOST_PARAMS
    }

    # Train model
    cat("\nTraining final model...\n")
    model <- train_catboost_model(data$train_pool, data$valid_pool, params)

    # Evaluate
    cat("\nEvaluating model...\n")
    metrics <- evaluate_catboost(model, data$valid_pool, data$y_valid)

    # Make predictions
    cat("\nMaking predictions on test set...\n")
    predictions <- predict_catboost(model, data$test_pool)

    return(list(
        model = model,
        predictions = predictions,
        metrics = metrics,
        params = params
    ))
}

cat("\nCatBoost modeling functions loaded!\n")
