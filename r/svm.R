# ============================================
# 🛡️ Support Vector Classification (SVC)
# ============================================

# 📂 1. Load Dataset
dataset <- read.csv('../data/Social_network_ads.csv')
# Columns: Age, EstimatedSalary, Purchased

# 🔀 2. Split into Training and Test Sets
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
train_set <- subset(dataset, split == TRUE)
test_set  <- subset(dataset, split == FALSE)

# ⚖️ 3. Feature Scaling
train_set[, 1:2] <- scale(train_set[, 1:2])
test_set[,  1:2] <- scale(test_set[,  1:2])

# 🧠 4. Train SVM Classifier
library(e1071)
classifier <- svm(
  x    = train_set[, 1:2],
  y    = train_set$Purchased,
  type = 'C-classification',
  kernel = 'radial'  # default; change to 'linear' or others if desired
)

# 🔍 5. Predict on Test Set
y_pred <- predict(classifier, newdata = test_set[, 1:2])

# 📊 6. Confusion Matrix
cm <- table(Actual = test_set$Purchased, Predicted = y_pred)
print(cm)

# 📈 7. Decision Boundary Plot Function
plot_svc_boundary <- function(data, title) {
  # Grid for plotting
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
    xlim = c(min(X1), max(X1)),
    ylim = c(min(X2), max(X2))
  )
  
  # Add decision boundary
  contour(
    X1, X2,
    matrix(as.numeric(grid$Purchased), length(X1), length(X2)),
    add = TRUE
  )
  
  # Plot grid points
  points(
    grid$Age, grid$EstimatedSalary,
    pch = '.', 
    col = ifelse(grid$Purchased == 1, 'springgreen3', 'tomato'),
    cex = 0.5
  )
  
  # Plot actual observations
  points(
    data$Age, data$EstimatedSalary,
    pch = 21,
    bg  = ifelse(data$Purchased == 1, 'green4', 'red3'),
    col = 'black',
    cex = 1
  )
}

# 🖼️ 8. Visualize Training Set
plot_svc_boundary(train_set, title = 'SVC (Training Set)')

# 🖼️ 9. Visualize Test Set
plot_svc_boundary(test_set, title = 'SVC (Test Set)')
