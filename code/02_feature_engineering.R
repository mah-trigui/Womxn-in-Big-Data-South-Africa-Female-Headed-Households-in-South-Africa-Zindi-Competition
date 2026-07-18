# ==============================================================================
# 02_feature_engineering.R - Feature Engineering
# ==============================================================================
# Project: Traffic/Transportation Travel Time Prediction
# Description: Create geographic, traffic, and transit features
# ==============================================================================

source("00_config.R")
print_header("Step 2: Feature Engineering")

# ---- Load Pre-processed Data ----
# Assuming df contains the main dataframe after initial processing
# Load from previous step if needed:
# df <- readRDS(paste0(OUTPUT_DIR, "train_raw.rds"))

# ---- Geographic Distance Features ----
cat("Creating geographic distance features...\n")

#' Calculate geographic distance statistics by ward
create_geo_distance_features <- function(wards_data, ward_col = "ADM4_PCODE", diff_col = "diff_X") {
    ag <- ddply(wards_data[, c(ward_col, diff_col), with = FALSE],
        as.formula(paste0("~", ward_col)),
        summarise,
        geo_dist_mean = mean(get(diff_col), na.rm = TRUE),
        geo_dist_sd = sd(get(diff_col), na.rm = TRUE)
    )
    names(ag)[1] <- ward_col
    return(ag)
}

# ---- Transit and Route Features ----
cat("Creating transit features...\n")

#' Process transit route data
process_transit_routes <- function(transit_route) {
    transit_route_clean <- transit_route[, c("id", "rank", "modes", "dist_transit")]
    transit_route_clean$geometry <- NULL
    transit_route_clean <- as.data.table(transit_route_clean)
    transit_route_clean <- transit_route_clean[rank == 1]
    return(transit_route_clean)
}

#' Process station proximity data
process_station_proximity <- function(station_data, radius_type = "urban") {
    station_clean <- station_data
    station_clean$lines <- NULL
    station_clean$geometry <- NULL
    station_clean <- as.data.table(station_clean)
    station_clean <- station_clean[rank == 1]
    station_clean$rank <- NULL
    station_clean$station <- NULL

    col_name <- ifelse(radius_type == "urban", "dist_station_u", "dist_station_b")
    names(station_clean)[2] <- col_name
    return(station_clean)
}

# ---- Traffic Flow Features ----
cat("Creating traffic features...\n")

#' Aggregate traffic flow data
aggregate_traffic_flow <- function(traffic_data) {
    traffic_clean <- traffic_data[, c("id", "SP", "FF", "JF", "CN")]
    traffic_clean$geometry <- NULL
    traffic_clean <- as.data.table(traffic_clean)
    traffic_clean <- traffic_clean[CN > 0]

    ag_traffic <- as.data.table(sqldf("
    SELECT id,
           AVG(SP) as speed_avg,
           AVG(FF) as ff_avg,
           AVG(JF) as jf_avg,
           AVG(CN) as cn_avg
    FROM traffic_clean
    GROUP BY id
  "))

    return(ag_traffic)
}

# ---- Clustering Features ----
cat("Creating cluster-based features...\n")

#' Create binary cluster indicators
create_cluster_indicators <- function(df, cluster_col = "cluster") {
    df$clus_1 <- ifelse(df[[cluster_col]] == 1, 1, 0)
    df$clus_2 <- ifelse(df[[cluster_col]] == 2, 1, 0)
    return(df)
}

# ---- Urban/Banlieu Classification ----
cat("Creating urban/banlieu classification...\n")

#' Classify areas as urban or banlieu based on station proximity
classify_urban_areas <- function(df,
                                 urban_threshold = 200000,
                                 banlieu_threshold = 100000) {
    df$urban <- ifelse(df$dist_station_u < urban_threshold, 1, 0)
    df$banlieu <- ifelse(df$dist_station_b < banlieu_threshold, 1, 0)
    return(df)
}

# ---- Transport Mode Features ----
cat("Creating transport mode features...\n")

#' Create binary transport mode indicators
create_transport_indicators <- function(df) {
    df$train <- ifelse(df$transport == "Train", 1, 0)
    df$public <- ifelse(df$transport == "publicTransport", 1, 0)
    return(df)
}

# ---- Master Feature Engineering Function ----

#' Apply all feature engineering steps
engineer_features <- function(df, ward_here, station_near, station_near_b,
                              traffic, transit_route, routes_public, ag) {
    cat("  Adding geographic features...\n")
    df <- df %>% left_join(ag, by = "ADM4_PCODE")

    cat("  Adding station proximity features...\n")
    station_u <- process_station_proximity(station_near, "urban")
    station_b <- process_station_proximity(station_near_b, "banlieu")

    ward_here <- ward_here %>%
        left_join(station_u, by = "id") %>%
        left_join(station_b, by = "id")

    # Fill missing station distances with max values
    ward_here$dist_station_u[is.na(ward_here$dist_station_u)] <- 200000
    ward_here$dist_station_b[is.na(ward_here$dist_station_b)] <- 100000

    cat("  Adding traffic features...\n")
    ag_traffic <- aggregate_traffic_flow(traffic)
    ward_here <- ward_here %>% left_join(ag_traffic[, -c("ff_avg")], by = "id")

    # Fill missing traffic values
    ward_here$speed_avg[is.na(ward_here$speed_avg)] <- 500
    ward_here$jf_avg[is.na(ward_here$jf_avg)] <- 25
    ward_here$cn_avg[is.na(ward_here$cn_avg)] <- -1

    cat("  Adding transit features...\n")
    transit_clean <- process_transit_routes(transit_route)
    ward_here <- ward_here %>% left_join(transit_clean[, -c("rank", "dist_transit")], by = "id")
    ward_here$modes[is.na(ward_here$modes)] <- "none"

    # Process public routes
    routes_public_clean <- routes_public[, c("id", "mode")]
    routes_public_clean$geometry <- NULL
    routes_public_clean <- as.data.table(routes_public_clean)
    ward_here <- ward_here %>% left_join(routes_public_clean, by = "id")
    ward_here$mode <- as.character(ward_here$mode)
    ward_here$mode[is.na(ward_here$mode)] <- "none"

    # Combine transport modes (vectorized)
    ward_here$transport <- pmax(
        as.character(ward_here$modes),
        as.character(ward_here$mode)
    )
    ward_here$transport <- as.factor(ward_here$transport)

    cat("  Joining ward features to main dataframe...\n")
    df <- df %>% left_join(ward_here[, -c("ADM3_PCODE", "id", "modes", "mode")], by = "ADM4_PCODE")

    cat("  Creating derived features...\n")
    df <- create_cluster_indicators(df)
    df <- classify_urban_areas(df)
    df <- create_transport_indicators(df)

    return(df)
}

cat("\nFeature engineering functions loaded!\n")
