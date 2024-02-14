#######################################
## Lasso Regression using Tidymodels ##
#######################################

# load necessary libraries
library(glmnet)
library(tidymodels)
library(broom)

# load the dataset
dataset = read.csv("C:/Users/user/Desktop/UNI/3RD YEAR 2023-24/ST310 Machine Learning/Group Project/subsampled_superconduct.csv")

# set up data splitting rules
data_split = initial_split(dataset)
data_train = training(data_split)
data_test = testing(data_split)

# set lambda to a high value to push most of the coefficients to zero
high_lambda = 10

# create the lasso model object
mod_lasso = linear_reg(penalty = high_lambda, mixture = 1) %>% 
  set_engine("glmnet") # mixture = 1 specifies lasso, the tuned penalty is the lambda

# set up the recipe for lasso
regularized_recipe = training(data_split) %>%
  recipe(critical_temp ~ .) %>%
  step_normalize(all_predictors()) %>% 
  prep()

# specify the workflow
lasso_workflow = workflow() %>% 
  add_model(mod_lasso) %>% 
  add_recipe(regularized_recipe)
  
# fit the lasso model
lasso_fit = lasso_workflow %>% 
  fit(data_train) 

# coefficient plot
coefficients_df <- tidy(lasso_fit, number = 1)

# plot coefficients
plot(coefficients_df$estimate, xlab = "Coefficient Index", ylab = "Coefficient Value", main = "Lasso Coefficient Path")
abline(h = 0, col = "gray", lty = 2)
text(seq_along(coefficients_df$estimate), coefficients_df$estimate, labels = coefficients_df$term, pos = 3, cex = 0.8)

# make predictions on test data
lasso_test_predictions <- predict(lasso_fit, new_data = data_test) %>%
  bind_cols(data_test)

# calculate test error - rmse
rmse_value <- lasso_test_predictions %>%
  metrics(truth = critical_temp, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

print(rmse_value)