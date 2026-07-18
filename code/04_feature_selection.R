# ==============================================================================
# 04_feature_selection.R - Feature Selection and Dimensionality Reduction
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: PCA, variable importance, feature selection
# ==============================================================================

source("00_config.R")
print_header("Step 4: Feature Selection")

# ---- PCA Analysis ----
cat("Setting up PCA analysis functions...\n")

#' Perform PCA and return transformed data
perform_pca <- function(df, target_col = "target", n_components = NULL, scale = TRUE) {
  # Remove target for PCA
  features <- df[, !names(df) %in% target_col, with = FALSE]

  # Perform PCA
  pca_result <- prcomp(features, scale. = scale, center = TRUE)

  cat("  PCA completed. Variance explained:\n")
  var_explained <- summary(pca_result)$importance[2, ]
  cumvar <- cumsum(var_explained)

  # Find components for 90% variance
  n_90 <- which(cumvar >= 0.90)[1]
  cat("    Components for 90% variance:", n_90, "\n")

  # Select number of components
  if (is.null(n_components)) {
    n_components <- n_90
  }

  # Transform data
  pca_scores <- as.data.table(pca_result$x[, 1:n_components])
  pca_scores[[target_col]] <- df[[target_col]]

  return(list(
    pca_model = pca_result,
    transformed_data = pca_scores,
    n_components = n_components,
    var_explained = var_explained
  ))
}

#' Apply PCA transformation to new data
apply_pca_transform <- function(new_data, pca_model, n_components) {
  new_features <- new_data[, !names(new_data) %in% "target", with = FALSE]
  scores <- predict(pca_model, newdata = new_features)
  return(as.data.table(scores[, 1:n_components]))
}

# ---- PLSR Analysis ----
cat("Setting up PLSR analysis functions...\n")

#' Fit Partial Least Squares Regression
fit_plsr <- function(df, target_col = "target", n_components = NULL) {
  # Create formula
  formula_str <- as.formula(paste(target_col, "~ ."))

  # Fit PLSR with cross-validation
  plsr_model <- plsr(formula_str, data = df, scale = TRUE, validation = "CV")

  # Find optimal components
  cv_error <- RMSEP(plsr_model)
  if (is.null(n_components)) {
    n_components <- which.min(cv_error$val[1, 1, ]) - 1
    n_components <- max(n_components, 1) # At least 1 component
  }

  cat("  Optimal PLSR components:", n_components, "\n")

  return(list(
    model = plsr_model,
    n_components = n_components
  ))
}

# ---- Variable Clustering ----
cat("Setting up variable clustering functions...\n")

#' Perform variable clustering using ClustOfVar
perform_variable_clustering <- function(df, target_col = "target", n_clusters = NULL) {
  # Remove target
  features <- df[, !names(df) %in% target_col, with = FALSE]

  # Hierarchical clustering of variables
  tree <- hclustvar(features)

  # Determine optimal clusters if not specified
  if (is.null(n_clusters)) {
    stab <- stability(tree, B = 50)
    n_clusters <- stab$kopt
    cat("  Optimal clusters from stability analysis:", n_clusters, "\n")

    # Also try with more iterations for confirmation
    stab2 <- stability(tree, B = 100)
    if (stab2$kopt != n_clusters) {
      cat("  Note: B=100 suggests", stab2$kopt, "clusters\n")
    }
  }

  # Perform k-means variable clustering
  km_result <- kmeansvar(features, init = n_clusters)

  # Get cluster scores
  scores <- as.data.table(km_result$scores)
  scores[[target_col]] <- df[[target_col]]

  return(list(
    tree = tree,
    kmeans = km_result,
    cluster_assignments = km_result$cluster,
    transformed_data = scores,
    n_clusters = n_clusters
  ))
}

# ---- Feature Importance (Caret) ----
cat("Setting up feature importance functions...\n")

#' Calculate feature importance using linear model
calc_feature_importance <- function(df, target_col = "target") {
  # Setup cross-validation
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

  # Train linear model
  formula_str <- as.formula(paste(target_col, "~ ."))
  model <- train(formula_str,
    data = df, method = "lm",
    preProcess = "scale", trControl = control
  )

  # Calculate importance
  importance <- varImp(model, scale = FALSE)

  return(importance)
}

#' Select top N important features
select_top_features <- function(importance, n_features = 30) {
  imp_df <- importance$importance
  imp_df$feature <- rownames(imp_df)
  imp_df <- imp_df[order(-imp_df$Overall), ]
  top_features <- imp_df$feature[1:min(n_features, nrow(imp_df))]
  return(top_features)
}

# ---- Stepwise Selection ----
cat("Setting up stepwise selection functions...\n")

#' Perform stepwise feature selection using AIC
stepwise_selection <- function(df, target_col = "target", direction = "both") {
  formula_str <- as.formula(paste(target_col, "~ ."))
  full_model <- lm(formula_str, data = df)

  # Stepwise selection
  step_model <- stepAIC(full_model, direction = direction, trace = FALSE)

  # Get selected features
  selected_features <- names(coef(step_model))[-1] # Remove intercept

  cat("  Selected", length(selected_features), "features via stepwise\n")

  return(list(
    model = step_model,
    features = selected_features
  ))
}

#' Perform forward selection using p-values (olsrr)
forward_selection_pvalue <- function(df, target_col = "target") {
  formula_str <- as.formula(paste(target_col, "~ ."))
  model <- lm(formula_str, data = df)
  result <- ols_step_forward_p(model)
  return(result)
}

# ---- Pre-defined Important Features ----
# Based on original analysis, these are the most important features

#' Get pre-analyzed important feature indices
get_important_feature_indices <- function() {
  # These indices came from the original importance analysis
  return(c(
    14, 20, 42, 21, 26, 39, 18, 13, 10, 29, 43, 41, 37, 49,
    3, 2, 35, 25, 31, 6, 23, 11, 33, 7, 47, 27, 1, 28, 17, 19, 4
  ))
}

#' Select features based on pre-analyzed importance
select_preanalyzed_features <- function(df, include_target = TRUE) {
  # Get important column indices for the PCR subset (first 49 cols + target at 50)
  indices <- get_important_feature_indices()

  # Add target column index if needed
  if (include_target) {
    indices <- c(indices, 50)
  }

  # Select columns
  selected <- df[, indices, with = FALSE]

  return(selected)
}

cat("\nFeature selection functions loaded!\n")
