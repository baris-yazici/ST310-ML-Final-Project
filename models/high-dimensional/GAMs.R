##########################
#### GAMs and GAMSEL #####
##########################

# Load required libraries
library(tidymodels) 
library(mgcv)
library(dplyr)
library(gamsel)

# Load the dataset
dataset = read.csv("subsampled_superconduct.csv")

# Set up data splitting rules and the recipe
data_split = initial_split(dataset)
data_train = training(data_split)
data_test = testing(data_split)
Y_train <- data_train[, 82]
X_train <- as.matrix(data_train[, -82]) # leave out y
p <- ncol(X_train)
n <- nrow(X_train)

### GAM using Tidymodels ####

# predictors names
predictors_col_names <- colnames(X_train)

# formula for gam model with different k values k=3,4,5
formula_1 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 3,  ", fx=TRUE", ")", collapse = " + ")))
formula_2 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 4,  ", fx=TRUE", ")", collapse = " + ")))
formula_3 <- as.formula(paste0("critical_temp ~ ", paste0("s(", predictors_col_names, ", k=", 5,  ", fx=TRUE", ")", collapse = " + ")))

# create the GAM model object
mod_gam = gen_additive_mod() |>
  set_engine("mgcv") |>
  set_mode("regression")

# fitting GAM models with with different k values
fit_gam_1 <- mod_gam %>%
  fit(formula_1,
      data = data_train)
fit_gam_2 <- mod_gam %>%
  fit(formula_2,
      data = data_train)
fit_gam_3 <- mod_gam %>%
  fit(formula_3,
      data = data_train)

# summary of each fitted model to observe parameters and goodness of fit
fit_gam_1 %>% pluck('fit') %>% summary()
fit_gam_2 %>% pluck('fit') %>% summary()
fit_gam_3 %>% pluck('fit') %>% summary()

# make predictions on test data
gam_test_1 <- predict(fit_gam_1, new_data = data_test) %>%
  bind_cols(data_test)
gam_test_2 <- predict(fit_gam_2, new_data = data_test) %>%
  bind_cols(data_test)
gam_test_3 <- predict(fit_gam_3, new_data = data_test) %>%
  bind_cols(data_test)

# calculate test error (rmse)
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

# Comments: With k=5, we get a high dimensional model with higher number of 
# predictors, however it's prediction accuracy is sub-optimal due to the rmse.
# With degrees of freedom, k=5 we get as close as possible to the number of 
# predictors being equal to the sample size. Moreover, fitting over k=5 was not 
# possible because then number of predictors exceeds sample size.

### GAMSEL - Regularized GAM ####

# The function gamsel() fits GAMSEL for a path of lambda values (hyperparameter) 
# and returns a gamsel object. By default, the model is fit for 50 different 
# lambda values. The returned gamsel object contains some useful information 
# on the fitted model. We use degrees of polynomial to be 5 since higher 
# degree polynomial exceeds the number of unique points in predictor matrix.

# changing degrees of freedom equal to 5 as the high dimensional GAM model
fit_gamsel <- gamsel(as.matrix(X_train), as.matrix(Y_train), degree = 5, df=5)

# printing the returned gamsel object tells us how many features, linear 
# components and non-linear components were included in the model for each 
# lambda value respectively.
print(fit_gamsel)

# Class “gamsel” objects also have a summary method which allows the user to see 
# the coefficient profiles of the linear and non-linear features. On each plot 
# (one for linear features and one for non-linear features), the x-axis is the 
# lambda value going from large to small. For linear components, the y-axis is 
# the coefficient for each variable while for the nonlinear components, it is 
# the norm of the nonlinear coefficients.
# We can include annotations to show which profile belongs to which feature by 
# specifying label = TRUE.
par(mfrow = c(1, 2))
summary(fit_gamsel, label = TRUE)

# We can perform (k)-fold cross-validation (CV) for GAMSEL with cv.gamsel(). 
# It does 10-fold cross-validation by default.
# We can change the number of folds using the nfolds option:
cv_fit <- cv.gamsel(as.matrix(X_train), as.matrix(Y_train), degree = 5, df=5, nfolds = 5)

# A cv.gamsel() call returns a cv.gamsel object. We can plot this object to 
# get the CV curve with error bars (one standard error in each direction). 
# The left vertical dotted line represents lambda.min, the lambda value which 
# attains minimum CV error, while the right vertical dotted line represents 
# lambda.1se, the largest lambda value with CV error within one standard error 
# of the minimum CV error.
plot(cv_fit)

# The numbers at the top represent the number of features in our 
# original input matrix that are included in the model.

# The two special lambda values, as well as their indices in the lambda path, 
# can be extracted directly from the cv.gamsel object:

# lambda values
cv_fit$lambda.min #optimal smoothing parameter
cv_fit$lambda.1se

# Refit the model 
# Use the optimal lambda value to refit the GAMSEL model on the entire training dataset
optimal_lambda_1 <- cv_fit$lambda.min  # or cv_fit$lambda.1se based on your preference
optimal_lambda_2 <- cv_fit$lambda.1se
fit_final <- gamsel(as.matrix(X_train), as.matrix(Y_train), degree = 5, df = 5, lambda = c(optimal_lambda_1))

# Evaluate on test data
# Once you've trained the final model, you can evaluate its performance on the 
# test dataset to estimate how well it generalizes to unseen data.
X_test <- as.matrix(data_test[, -82])
Y_test <- as.matrix(data_test[, 82])

predictions <- predict(fit_final, newdata = X_test)

# Evaluate predictions using rmse
rmse_gamsel <- sqrt(mean((predictions - Y_test)^2))

# Comments: 
# Lambda controls the amount of regularization applied to the model. 
# Higher values of lambda lead to stronger regularization, which penalizes the 
# coefficients and helps prevent overfitting by shrinking them towards zero. 
# With lambda.min, a smaller number of the predictors are penalized to 0 than 
# with the lambda.1se. Note that by default, the maximum degrees of freedom for 
# each variable is 5. This can be modified with the dfs option, with larger 
# values allowing more “wiggly” fits.

# Comparing accuracy of GAM vs GAMSEL
rmse_matrix <- data.frame(
  model = c("Gam 1", "Gam 2", "Gam 3", "Gamsel"),
  rmse = c(gam_rmse_1, gam_rmse_2, gam_rmse_3, rmse_gamsel))

## compare which estimation method is the best
best_rmse <- rmse_matrix[which.min(rmse_matrix$rmse), ]
print(best_rmse)

# Comparison between models:
# In comparing the gam and gamsel models, the gamsel model is the better model 
# for high dimensionality because lambda.min in the model penalises the 
# coefficients and helps prevent overfitting, potentially increasing the complexity 
# of the model, and thus resulting in a high dimensional model. Furthermore, whilst 
# making it more high dimensional, lambda.min regularises the model to reduce the 
# risk of overfitting. Both the gam and gamsel model have 5 degrees of freedom. 
# In terms of predictive accuracy gamsel did much better than the high dimensional 
# gam model. Thus considering all these factors, we conclude gamsel to be our best 
# high dimensional model.
