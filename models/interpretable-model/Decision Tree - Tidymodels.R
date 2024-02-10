#######################################
### Decision Tree using Tidymodels ####
#######################################

# Load required libraries
library(tidymodels) # We use Tidymodels workflow

# Set up data splitting rules and the recipe
data_split = initial_split(dataset)
data_train = training(data_split)
data_test = testing(data_split)

# Set up the cross validation rule
data_cv = vfold_cv(data_train, v = 10) # 10-fold cross-validation

# Set up the recipe for tree
data_recipe = training(data_split) %>%
  recipe(critical_temp ~ .) %>%
  prep()

# Create the model object
mod_tree = decision_tree(tree_depth = 8,
                         cost_complexity = tune("C")) %>% # we tune the cost_complexity parameter using CV
  set_engine("rpart") %>% 
  set_mode("regression")

# Specify the workflow
workflow_tree = workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(mod_tree)

# Fit the model using 10-fold CV to find the optimal cost complexity
fit_tree = tune_grid(
  workflow_tree,
  grid = data.frame(C = 2^(-20:0)),
  data_cv,
  metrics = yardstick::metric_set(yardstick::rmse)
)

# Choose the best model
tree_best = fit_tree %>% select_best()

# Finalize the model
tree_final = finalize_model(mod_tree, tree_best)

# Calculate the test error for the final model
tree_test =
  workflow_tree %>%
  update_model(tree_final) %>%
  last_fit(split = data_split) %>%
  collect_metrics() # test error
