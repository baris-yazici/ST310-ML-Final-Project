#######################################
### Random Forest using Tidymodels ####
#######################################

# Load required libraries
library(randomForest) # Needed to fit RF when we set the engine 
library(tidymodels) # We use Tidymodels workflow

# Set up data splitting rules and the recipe
data_split = initial_split(dataset)
data_train = training(data_split)
data_test = testing(data_split)

# Set up the cross validation rule
data_cv = vfold_cv(data_train, v = 10) # 10-fold cross-validation

# Set up the recipe for RF
data_recipe = training(data_split) %>%
  recipe(critical_temp ~ .) %>%
  prep()

# Create the RF model object
mod_rf = 
  rand_forest(trees = 1000, mtry = tune()) %>% # mtry will be tuned using CV
  set_mode("regression") %>%
  set_engine("randomForest")

# Specify the workflow
workflow_rf = workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(mod_rf)

# Fit the model using 10-fold CV to find the optimal mtry
fit_rf = tune_grid(
  workflow_rf, 
  data_cv,
  metrics = yardstick::metric_set(yardstick::rmse)
)

# Choose the best model
rf_best = fit_rf %>% 
  select_best()

# Finalize the model
rf_final = finalize_model(
  mod_rf, 
  rf_best)

# Calculate the test error for the final model
rf_test = 
  workflow_rf %>%
  update_model(rf_final) %>%
  last_fit(split = data_split) %>%
  collect_metrics() # test error
