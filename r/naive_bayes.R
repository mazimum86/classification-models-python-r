# ============================================
# 🌼 Naive Bayes Classification
# ============================================

# 1. Load Dataset
dataset <- read.csv('../data/social_network_ads.csv')
dataset$Purchased <- factor(dataset$Purchased)  # Ensure target is a factor

# 2. Split into Training and Test Sets
library(caTools)
set.seed(123)
split          <- sample.split(dataset$Purchased, SplitRatio = 0.8)
train_set      <- subset(dataset, split == TRUE)
test_set       <- subset(dataset, split == FALSE)

# 3. Feature Scaling (standardizing Age and EstimatedSalary)
train_set[, 1:2] <- scale(train_set[, 1:2])
test_set[,  1:2] <- scale(test_set[,  1:2])

# 4. Train Naive Bayes Model
library(e1071)
classifier <- naiveBayes(
  x = train_set[, 1:2],
  y = train_set$Purchased
)

# 5. Predict on Test Set
y_pred <- predict(classifier, newdata = test_set[, 1:2])

# 6. Confusion Matrix
cm <- table(
  Actual    = test_set$Purchased,
  Predicted = y_pred
)
print(cm)

# 7. Decision Boundary Plot Function
plot_naive_bayes_boundary <- function(data, title) {
  # Create grid
  X1 <- seq(min(data$Age) - 1, max(data$Age) + 1, by = 0.01)
  X2 <- seq(min(data$EstimatedSalary) - 1, max(data$EstimatedSalary) + 1, by = 0.01)
  grid <- expand.grid(Age = X1, EstimatedSalary = X2)
  
  # Predict grid labels
  grid$Purchased <- predict(classifier, newdata = grid)
  
  # Base plot
  plot(
    data$Age, data$EstimatedSalary,
    type = 'n',
    main = title,
    xlab = 'Age (scaled)',
    ylab = 'Estimated Salary (scaled)',
    xlim = range(X1),
    ylim = range(X2)
  )
  
  # Decision regions
  contour(
    X1, X2,
    matrix(as.numeric(grid$Purchased), length(X1), length(X2)),
    add = TRUE
  )
  
  # Grid points colored by predicted class
  points(
    grid$Age, grid$EstimatedSalary,
    pch = '.', 
    col = ifelse(grid$Purchased == 1, 'springgreen3', 'tomato'),
    cex = 0.5
  )
  
  # Actual data points
  points(
    data$Age, data$EstimatedSalary,
    pch = 21,
    bg  = ifelse(data$Purchased == 1, 'green4', 'red3'),
    col = 'black'
  )
}

# 8. Visualize Training Set
plot_naive_bayes_boundary(train_set, title = 'Naive Bayes (Training Set)')

# 9. Visualize Test Set
plot_naive_bayes_boundary(test_set,  title = 'Naive Bayes (Test Set)')
