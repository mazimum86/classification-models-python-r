# ==============================================
# 🌳 Decision Tree Classification (R)
# ==============================================

# 1. Load Dataset
dataset <- read.csv('../data/social_network_ads.csv')

# 2. Split into Training and Test Sets
library(caTools)
set.seed(123)
split           <- sample.split(dataset$Purchased, SplitRatio = 0.8)
train_set       <- subset(dataset, split == TRUE)
test_set        <- subset(dataset, split == FALSE)

# 3. Retain Unscaled Copies for Tree Visualization
train_unscaled  <- train_set
test_unscaled   <- test_set

# 4. Feature Scaling (for decision boundary plots)
train_set[, 1:2] <- scale(train_set[, 1:2])
test_set[,  1:2] <- scale(test_set[,  1:2])

# 5. Train Decision Tree Models
library(rpart)

# 5a. Unscaled model (for plotting tree structure)
tree_unscaled <- rpart(
  Purchased ~ .,
  data = train_unscaled,
  
)

# 5b. Scaled model (for decision region)
tree_scaled   <- rpart(
  Purchased ~ .,
  data = train_set,
  
)

# 6. Predict on Test Set (scaled)
y_pred <- as.numeric(predict(tree_scaled, newdata = test_set) > 0.5)

# 7. Confusion Matrix
cm <- table(
  Actual    = test_set$Purchased,
  Predicted = y_pred
)
print(cm)

# 8. Plot the Tree Structure (unscaled)
plot(tree_unscaled, main = "Decision Tree Structure (Unscaled Data)")
text(tree_unscaled, use.n = TRUE, all = TRUE, cex = 0.8)

# 9. Decision Boundary Plot Function
plot_decision_boundary <- function(data, title) {
  X1 <- seq(min(data$Age) - 1, max(data$Age) + 1, by = 0.01)
  X2 <- seq(min(data$EstimatedSalary) - 1, max(data$EstimatedSalary) + 1, by = 0.01)
  grid <- expand.grid(Age = X1, EstimatedSalary = X2)
  
  # Predict and threshold
  preds <- as.numeric(predict(tree_scaled, newdata = grid) > 0.5)
  
  # Base plot
  plot(
    data$Age, data$EstimatedSalary,
    type = 'n', main = title,
    xlab = 'Age (scaled)', ylab = 'Estimated Salary (scaled)',
    xlim = range(X1), ylim = range(X2)
  )
  contour(X1, X2, matrix(preds, length(X1), length(X2)), add = TRUE)
  
  # Plot decision regions
  points(
    grid, pch = '.', 
    col = ifelse(preds == 1, 'springgreen3', 'tomato'),
    cex = 0.5
  )
  # Plot true points
  points(
    data$Age, data$EstimatedSalary,
    pch = 21, bg = ifelse(data$Purchased == 1, 'green4', 'red3')
  )
}

# 10. Visualize Decision Regions
plot_decision_boundary(train_set, "Decision Tree (Training Set)")
plot_decision_boundary(test_set,  "Decision Tree (Test Set)")
