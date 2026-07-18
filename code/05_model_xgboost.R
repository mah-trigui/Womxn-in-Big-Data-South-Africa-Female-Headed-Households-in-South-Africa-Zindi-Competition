# ==============================================================================
# 05_model_xgboost.R - XGBoost Model Training and Prediction
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: XGBoost model with cross-validation and hyperparameter tuning
# ==============================================================================

source("00_config.R")
print_header("Step 5: XGBoost Modeling")

# ---- XGBoost Data Preparation ----

#' Prepare data for XGBoost
prepare_xgb_data <- function(train_df, test_df = NULL, target_col = "target") {
    # Separate features and target
    y_train <- train_df[[target_col]]
    X_train <- train_df[, !names(train_df) %in% target_col, with = FALSE]

    # Create sparse matrix for training
    train_matrix <- sparse.model.matrix(~ . - 1, data = X_train, sparse = FALSE)
    train_dmatrix <- xgb.DMatrix(data = train_matrix, label = y_train)

    result <- list(
        train_matrix = train_matrix,
        train_dmatrix = train_dmatrix,
        y_train = y_train
    )

    # Prepare test data if provided
    if (!is.null(test_df)) {
        X_test <- test_df[, !names(test_df) %in% target_col, with = FALSE]
        test_matrix <- sparse.model.matrix(~ . - 1, data = X_test, sparse = FALSE)
        result$test_matrix <- test_matrix
    }

    return(result)
}

# ---- Cross-Validation ----

#' Run XGBoost cross-validation
run_xgb_cv <- function(train_dmatrix, params = NULL, nrounds = 7000,
                       nfold = 10, early_stop = 50, verbose = TRUE) {
    if (is.null(params)) {
        params <- XGBOOST_PARAMS
    }

    cat("Running XGBoost cross-validation...\n")
    cat("  Params:", paste(names(params), params, sep = "=", collapse = ", "), "\n")

    cv_result <- xgb.cv(
        data = train_dmatrix,
        params = params,
        nrounds = nrounds,
        nfold = nfold,
        early_stopping_rounds = early_stop,
        maximize = FALSE,
        verbose = ifelse(verbose, 1, 0),
        print_every_n = 100
    )

    best_iteration <- cv_result$best_iteration
    best_score <- cv_result$evaluation_log$test_rmse_mean[best_iteration]

    cat("\n  Best iteration:", best_iteration, "\n")
    cat("  Best CV RMSE:", round(best_score, 4), "\n")

    return(list(
        cv_result = cv_result,
        best_iteration = best_iteration,
        best_score = best_score
    ))
}

# ---- Model Training ----

#' Train XGBoost model
train_xgb_model <- function(train_dmatrix, params = NULL, nrounds = NULL) {
    if (is.null(params)) {
        params <- XGBOOST_PARAMS
    }

    if (is.null(nrounds)) {
        # Run CV to determine optimal rounds
        cv <- run_xgb_cv(train_dmatrix, params)
        nrounds <- cv$best_iteration
    }

    cat("Training XGBoost model with", nrounds, "rounds...\n")

    model <- xgb.train(
        data = train_dmatrix,
        params = params,
        nrounds = nrounds,
        verbose = 1,
        print_every_n = 100
    )

    return(model)
}

# ---- Hyperparameter Grid Search ----

#' Grid search for XGBoost hyperparameters
xgb_grid_search <- function(train_df, target_col = "target",
                            max_depths = c(4, 6, 8),
                            etas = c(0.01, 0.05, 0.1),
                            gammas = c(0, 0.1, 1),
                            subsamples = c(0.5, 0.75, 1),
                            colsample_bytrees = c(0.5, 0.8, 1),
                            min_child_weights = c(1, 3, 5),
                            nrounds = 10000,
                            early_stop = 50) {
    # Prepare data split
    set.seed(GLOBAL_SEED)
    split_idx <- sample(1:nrow(train_df), size = floor(0.8 * nrow(train_df)))
    train_set <- train_df[split_idx, ]
    valid_set <- train_df[-split_idx, ]

    # Prepare matrices
    y_train <- train_set[[target_col]]
    X_train <- as.matrix(train_set[, !names(train_set) %in% target_col, with = FALSE])
    y_valid <- valid_set[[target_col]]
    X_valid <- as.matrix(valid_set[, !names(valid_set) %in% target_col, with = FALSE])

    dtrain <- xgb.DMatrix(data = X_train, label = y_train)
    dvalid <- xgb.DMatrix(data = X_valid, label = y_valid)
    watchlist <- list(train = dtrain, test = dvalid)

    best_params <- NULL
    best_score <- Inf
    total_combos <- length(max_depths) * length(etas) * length(gammas) *
        length(subsamples) * length(colsample_bytrees) * length(min_child_weights)

    cat("Starting grid search over", total_combos, "combinations...\n")

    count <- 1
    for (depth in max_depths) {
        for (eta in etas) {
            for (gamma in gammas) {
                for (subsample in subsamples) {
                    for (colsample in colsample_bytrees) {
                        for (mcw in min_child_weights) {
                            params <- list(
                                booster = "gbtree",
                                objective = "reg:squarederror",
                                max_depth = depth,
                                eta = eta,
                                gamma = gamma,
                                subsample = subsample,
                                colsample_bytree = colsample,
                                min_child_weight = mcw
                            )

                            model <- xgb.train(
                                data = dtrain,
                                params = params,
                                nrounds = nrounds,
                                watchlist = watchlist,
                                early_stopping_rounds = early_stop,
                                verbose = 0
                            )

                            if (model$best_score < best_score) {
                                best_score <- model$best_score
                                best_params <- params
                                best_params$best_iteration <- model$best_iteration
                                cat(
                                    "  New best score:", round(best_score, 4),
                                    "at combo", count, "/", total_combos, "\n"
                                )
                            }

                            count <- count + 1
                        }
                    }
                }
            }
        }
    }

    cat("\nGrid search complete!\n")
    cat("Best RMSE:", round(best_score, 4), "\n")
    cat("Best params:\n")
    print(best_params)

    return(list(
        best_params = best_params,
        best_score = best_score
    ))
}

# ---- Prediction ----

#' Make predictions with XGBoost model
predict_xgb <- function(model, test_matrix) {
    predictions <- predict(model, test_matrix)
    return(predictions)
}

# ---- Feature Importance ----

#' Get XGBoost feature importance
get_xgb_importance <- function(model, top_n = 20) {
    importance <- xgb.importance(model = model)

    cat("\nTop", top_n, "important features:\n")
    print(head(importance, top_n))

    return(importance)
}

#' Plot XGBoost feature importance
plot_xgb_importance <- function(model, top_n = 20) {
    importance <- xgb.importance(model = model)
    xgb.plot.importance(importance_matrix = importance, top_n = top_n)
}

# ---- Complete XGBoost Pipeline ----

#' Run complete XGBoost pipeline
run_xgb_pipeline <- function(train_df, test_df, target_col = "target",
                             params = NULL, do_cv = TRUE, do_grid_search = FALSE) {
    cat("Starting XGBoost pipeline...\n\n")

    # Prepare data
    cat("Preparing data...\n")
    data <- prepare_xgb_data(train_df, test_df, target_col)

    # Use provided params or defaults
    if (is.null(params)) {
        if (do_grid_search) {
            cat("\nRunning grid search...\n")
            grid_result <- xgb_grid_search(train_df, target_col)
            params <- grid_result$best_params
            nrounds <- params$best_iteration
        } else {
            params <- XGBOOST_PARAMS
            nrounds <- NULL
        }
    }

    # Run CV if requested
    if (do_cv && !do_grid_search) {
        cat("\nRunning cross-validation...\n")
        cv <- run_xgb_cv(data$train_dmatrix, params)
        nrounds <- cv$best_iteration
    }

    # Train model
    cat("\nTraining final model...\n")
    model <- train_xgb_model(data$train_dmatrix, params, nrounds)

    # Make predictions
    cat("\nMaking predictions...\n")
    predictions <- predict_xgb(model, data$test_matrix)

    # Get importance
    importance <- get_xgb_importance(model)

    return(list(
        model = model,
        predictions = predictions,
        importance = importance,
        params = params
    ))
}

cat("\nXGBoost modeling functions loaded!\n")
