#######################################
### GAM using Tidymodels ####
#######################################

# Load required libraries
library(tidymodels) 
library(mgcv)
library(dplyr)
library(Metrics)

# Set up data splitting rules and the recipe
data_split = initial_split(dataset)
data_train = training(data_split)
data_test = testing(data_split)

# Create the GAM model object
mod_gam = gen_additive_mod() |>
  set_engine("mgcv") |>
  set_mode("regression")

# Fit a GAM model
fit_gam <- mod_gam %>%
  fit(critical_temp ~ 
        s(number_of_elements, k=7) + 
        s(mean_atomic_mass, k=7) + 
        s(entropy_atomic_mass, k=7) +
        s(mean_fie, k=7) +
        s(entropy_fie, k=7) + 
        s(mean_atomic_radius, k=7) + 
        s(entropy_atomic_radius, k=7) +
        s(mean_Density, k=7) +
        s(entropy_Density, k=7) +
        s(mean_ElectronAffinity, k=7) + 
        s(entropy_ElectronAffinity, k=7) + 
        s(mean_FusionHeat, k=7) +
        s(entropy_FusionHeat, k=7) +
        s(mean_ThermalConductivity, k=7) + 
        s(entropy_ThermalConductivity, k=7) + 
        s(mean_Valence, k=7) +
        s(entropy_Valence, k=7) +
        s(entropy_Density, k=7),
      method = "REML",
      data = data_train)
# s() stands for splines, indicating a non-linear relationship
# using the restricted maximum likelihood method REML which is more robust for 
# small sample sizes to avoid overfitting.

# make predictions on the test set
pred_gam <- unlist(predict(fit_gam, new_data = data_test))

# calculate the test error
gam_test <- rmse(data_test[, 82], pred_gam)
