# ==============================================================================
# 08_model_ensemble.R - Model Ensembling and Stacking
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: Combine multiple models using stacking, averaging, etc.
# ==============================================================================

source("00_config.R")
print_header("Step 8: Model Ensembling")

# ---- Simple Averaging ----

#' Average predictions from multiple models
average_predictions <- function(...) {
    preds_list <- list(...)
    n_models <- length(preds_list)

    avg_preds <- Reduce(`+`, preds_list) / n_models

    cat("Averaged predictions from", n_models, "models\n")
    return(avg_preds)
}

#' Weighted average predictions
weighted_average <- function(preds_list, weights) {
    if (length(preds_list) != length(weights)) {
        stop("Number of predictions must match number of weights")
    }

    # Normalize weights
    weights <- weights / sum(weights)

    weighted_preds <- mapply(function(p, w) p * w, preds_list, weights, SIMPLIFY = FALSE)
    final_preds <- Reduce(`+`, weighted_preds)

    cat("Weighted average with weights:", round(weights, 3), "\n")
    return(final_preds)
}

# ---- Caret Ensemble ----

#' Create model list for stacking using caretList
create_caret_model_list <- function(X_train, y_train, methods = NULL) {
    if (is.null(methods)) {
        methods <- c("lm", "ridge", "rqlasso", "relaxo")
    }

    # Setup cross-validation control
    ctrl <- trainControl(
        method = "cv",
        number = 5,
        savePredictions = "final",
        allowParallel = TRUE
    )

    cat("Training", length(methods), "base models...\n")

    # Train models
    model_list <- caretList(
        x = X_train,
        y = y_train,
        trControl = ctrl,
        methodList = methods,
        continue_on_fail = TRUE,
        preProcess = c("center", "scale")
    )

    return(model_list)
}

#' Create stacked ensemble from caret models
create_caret_stack <- function(model_list, method = "glm") {
    cat("Creating stacked ensemble using", method, "...\n")

    stack_ctrl <- trainControl(
        method = "cv",
        number = 5,
        savePredictions = "final"
    )

    stack_model <- caretStack(
        model_list,
        method = method,
        trControl = stack_ctrl
    )

    return(stack_model)
}

# ---- Manual Stacking ----

#' Generate out-of-fold predictions for stacking
generate_oof_predictions <- function(train_df, target_col = "target",
                                     model_funcs, nfolds = 5) {
    n_models <- length(model_funcs)
    n_obs <- nrow(train_df)

    # Create fold indices
    set.seed(GLOBAL_SEED)
    folds <- createFolds(train_df[[target_col]], k = nfolds, list = TRUE)

    # Initialize OOF prediction matrix
    oof_preds <- matrix(0, nrow = n_obs, ncol = n_models)

    cat("Generating OOF predictions for", n_models, "models across", nfolds, "folds...\n")

    for (fold_idx in seq_along(folds)) {
        cat("  Fold", fold_idx, "/", nfolds, "\n")

        val_idx <- folds[[fold_idx]]
        train_fold <- train_df[-val_idx, ]
        val_fold <- train_df[val_idx, ]

        for (model_idx in seq_along(model_funcs)) {
            # Train model on fold
            model <- model_funcs[[model_idx]](train_fold, target_col)

            # Predict on validation
            val_features <- val_fold[, !names(val_fold) %in% target_col, with = FALSE]
            oof_preds[val_idx, model_idx] <- model$predict(val_features)
        }
    }

    oof_df <- as.data.frame(oof_preds)
    names(oof_df) <- paste0("model_", seq_len(n_models))
    oof_df[[target_col]] <- train_df[[target_col]]

    return(oof_df)
}

#' Train meta-learner on OOF predictions
train_meta_learner <- function(oof_df, target_col = "target", method = "lm") {
    cat("Training meta-learner using", method, "...\n")

    formula_str <- as.formula(paste(target_col, "~ ."))

    if (method == "lm") {
        meta_model <- lm(formula_str, data = oof_df)
    } else {
        ctrl <- trainControl(method = "cv", number = 5)
        meta_model <- train(formula_str, data = oof_df, method = method, trControl = ctrl)
    }

    return(meta_model)
}

# ---- Blending Optimization ----

#' Find optimal blend weights using validation data
optimize_blend_weights <- function(preds_list, y_true,
                                   n_iterations = 1000) {
    n_models <- length(preds_list)
    best_weights <- rep(1 / n_models, n_models)
    best_rmse <- Inf

    cat("Optimizing blend weights for", n_models, "models...\n")

    for (i in seq_len(n_iterations)) {
        # Generate random weights
        weights <- runif(n_models)
        weights <- weights / sum(weights)

        # Calculate blended prediction
        blended <- weighted_average(preds_list, weights)

        # Calculate RMSE
        rmse <- sqrt(mean((blended - y_true)^2))

        if (rmse < best_rmse) {
            best_rmse <- rmse
            best_weights <- weights

            if (i %% 100 == 0) {
                cat("  Iteration", i, "- New best RMSE:", round(best_rmse, 4), "\n")
            }
        }
    }

    cat("\nOptimal weights found:\n")
    print(round(best_weights, 4))
    cat("Best RMSE:", round(best_rmse, 4), "\n")

    return(list(
        weights = best_weights,
        rmse = best_rmse
    ))
}

# ---- Complete Ensemble Pipeline ----

#' Run simple ensemble pipeline
run_ensemble_pipeline <- function(xgb_preds, catboost_preds, lgb_preds,
                                  y_valid = NULL, method = "average") {
    cat("Running ensemble pipeline with method:", method, "\n\n")

    preds_list <- list(xgb = xgb_preds, catboost = catboost_preds, lgb = lgb_preds)

    if (method == "average") {
        final_preds <- average_predictions(xgb_preds, catboost_preds, lgb_preds)
        weights <- rep(1 / 3, 3)
    } else if (method == "optimize" && !is.null(y_valid)) {
        opt_result <- optimize_blend_weights(preds_list, y_valid)
        weights <- opt_result$weights
        final_preds <- weighted_average(preds_list, weights)
    } else if (method == "weighted") {
        # Default weights based on typical performance
        weights <- c(0.3, 0.4, 0.3) # XGB, CatBoost, LightGBM
        final_preds <- weighted_average(preds_list, weights)
    } else {
        stop("Unknown method. Use 'average', 'weighted', or 'optimize'")
    }

    return(list(
        predictions = final_preds,
        weights = weights,
        method = method
    ))
}

cat("\nModel ensembling functions loaded!\n")
