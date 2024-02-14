################################################
### Ridge Regression using Gradient Descent ####
################################################

library(dplyr)

# Helper functions

## Loss function of Ridge
ridge_loss = function(X, y, beta, lambda) {
  # X is a design matrix without the intercept
  # We assume all the data is centered, as required by Ridge Regression
  residuals = y - X %*% beta # calculates residuals
  rss = sum(residuals^2)
  penalty = lambda*sum(beta^2) # penalty term with tuning parameter lambda
  r_loss = rss + penalty
  return(r_loss)
}

## Gradient of the loss function
ridge_gradient = function(X, y, beta, lambda) {
  residuals = y - X %*% beta
  gradient = -2*t(X)%*%residuals + 2*lambda*beta 
  return(gradient)
}

## Hyperparameters
stop_threshold = 0.001 # when the loss doesn't get better than 
  # stop_threshold value in consecutive updates the algorithm will stop
learning_rate = 0.01 # learning rate is how fast the algorithm updates the beta

# Gradient Descent Algorithm

gradient_descent_ridge = function(X, y, lambda, stop_threshold, learning_rate, max_iterations) {

  # hyperparameters
  ## lambda: user supplied lambda (regularization parameter for ridge regression)
  ## stop_threshold: when the loss doesn't get better than 
    ### stop_threshold value in consecutive updates the algorithm will stop
  ## learning rate: how fast the algorithm updates the beta
  ## max_iterations: maximum number of iterations we allow for
  
  # initializing the algorithm
  beta_init = rnorm(ncol(X), mean = 0, sd = 0.01) # we initialize beta from a normal distribution
  loss_beta_init = ridge_loss(X, y, beta_init, lambda) # calculate the loss for initial beta
  loss_beta_est = 0
  no_iteration = 0 # we keep track of number of iterations
  # the training loop, runs until updates are small
  while ((abs(loss_beta_init - loss_beta_est) > stop_threshold) && (no_iteration < max_iterations)){
    current_gradient = ridge_gradient(X, y, beta_init, lambda)
    beta_est = beta_init - learning_rate * current_gradient # update the previous beta with gradient
    loss_beta_est = ridge_loss(X, y, beta_est, lambda) # loss with updated beta
    beta_init = beta_est
    loss_beta_init = loss_beta_est
    no_iteration = no_iteration + 1
  }
  
  if (no_iteration == max_iterations) {
    warning("No convergence. Maximum iterations reached.")
  }
  return(beta_est)
}

# Fitting the Model
file_path = "~/Desktop/R/ST 310/Summative/Final Project/"
dataset = read.csv(paste0(file_path, "subsampled_superconduct.csv"))
X = dataset %>% dplyr::select(-critical_temp) %>% as.matrix() %>% scale() # normalize X
y = dataset$critical_temp %>% as.matrix()

# Set hyperparameters
lambda = 0.0001
stop_threshold = 0.001
learning_rate = 0.01
max_iterations = 10000

# Fit the model
beta_ridge = gradient_descent_ridge(X_normalized, y, lambda, stop_threshold, learning_rate, max_iterations)

# Print the coefficients
print(beta_ridge)
