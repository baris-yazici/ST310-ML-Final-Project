#######################################
### GAMs ####
#######################################

# Load required libraries
library(tidymodels) 
library(mgcv)
library(dplyr)

# Load the dataset
dataset = read.csv("subsampled_superconduct.csv")

# Set up data splitting rules and the recipe
data_split = initial_split(dataset)
data_train = training(data_split)
data_test = testing(data_split)

# Names of predictor variables
predictors_col_names <- colnames(dataset)[-82]

# Number of predictors
p <- length(predictors_col_names)

# Specifying formulas with different k values k=3,4,5,6
formula_1 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 3, ")", collapse = " + ")))

formula_2 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 4, ")", collapse = " + ")))

formula_3 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 5, ")", collapse = " + ")))

formula_4 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 6, ")", collapse = " + ")))

# Create the model object
mod_gam = gen_additive_mod() |>
  set_engine("mgcv") |>
  set_mode("regression")

# Fit different models with different k values, k=3,4,5,6
fit_gam_1 <- mod_gam %>%
  fit(formula_1,
      data = data_train)

fit_gam_2 <- mod_gam %>%
  fit(formula_2,
      data = data_train)

fit_gam_3 <- mod_gam %>%
  fit(formula_3,
      data = data_train)

fit_gam_4 <- mod_gam %>%
  fit(formula_4,
      data = data_train)

# Summary of each models
fit_gam_1 %>% pluck('fit') %>% summary()
fit_gam_2 %>% pluck('fit') %>% summary()
fit_gam_3 %>% pluck('fit') %>% summary()
fit_gam_4 %>% pluck('fit') %>% summary()

# Make predictions on test data
gam_test_1 <- predict(fit_gam_1, new_data = data_test) %>%
  bind_cols(data_test)

gam_test_2 <- predict(fit_gam_2, new_data = data_test) %>%
  bind_cols(data_test)

gam_test_3 <- predict(fit_gam_3, new_data = data_test) %>%
  bind_cols(data_test)

gam_test_4 <- predict(fit_gam_4, new_data = data_test) %>%
  bind_cols(data_test)

# Calculate test error - rmse
gam_rmse_1 <- gam_test_1 %>%
  metrics(truth = critical_temp, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

gam_rmse_2 <- gam_test_2 %>%
  metrics(truth = critical_temp, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

gam_rmse_3 <- gam_test_3 %>%
  metrics(truth = critical_temp, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

gam_rmse_4 <- gam_test_4 %>%
  metrics(truth = critical_temp, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

rmse_matrix <- data.frame(
  k_value = c("3", "4", "5", "6"),
  rmse = c(gam_rmse_1, gam_rmse_2, gam_rmse_3, gam_rmse_4))

# Compare which k value is the best
best_rmse_gam <- rmse_matrix[which.min(rmse_matrix$rmse), ]

print(best_rmse_gam)
