# Install and load the dplyr package
install.packages("dplyr")
library(dplyr)

# Subsample n observations from the dataset
n = 1000
subsample = superconduct_train %>% sample_n(n)

# Export as a csv
write.csv(subsample, "subsampled_superconduct.csv", row.names = FALSE)
