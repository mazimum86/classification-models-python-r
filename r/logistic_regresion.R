# Logistic Regression on Social Network Ads

# ==================== Import Dataset ====================
dataset <- read.csv('../data/Social_network_Ads.csv')
# Columns: Age, EstimatedSalary, Purchased

# ==================== Train/Test Split ====================
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 2/3)
dataset_train <- subset(dataset, split == TRUE)
dataset_test <- subset(dataset, split == FALSE)

# ==================== Feature Scaling ====================
dataset_train[,1:2] <- scale(dataset_train[,1:2])
dataset_test[,1:2] <- scale(dataset_test[,1:2])

# ==================== Fit Logistic Regression ====================
classifier <- glm(
  formula = Purchased ~ .,
  family = binomial,
  data = dataset_train
)

# ==================== Prediction ====================
prob_pred <- predict(classifier, type='response', newdata=dataset_test)
y_pred <- as.numeric(prob_pred > 0.5)

# ==================== Confusion Matrix ====================
cm <- table(Actual = dataset_test$Purchased, Predicted = y_pred)
print(cm)

# ==================== Plot Decision Boundaries ====================

# Helper function for plotting
plot_decision_boundary <- function(set, title) {
  X1 <- seq(min(set[,1]) - 1, max(set[,1]) + 1, by=0.1)
  X2 <- seq(min(set[,2]) - 1, max(set[,2]) + 1, by=0.1)
  grid <- expand.grid(Age=X1, EstimatedSalary=X2)
  
  grid_pred <- as.numeric(predict(classifier, type='response', newdata=grid) > 0.5)
  
  plot(
    set[,1:2],
    main=title,
    xlab='Age (scaled)',
    ylab='Estimated Salary (scaled)',
    xlim=range(X1),
    ylim=range(X2)
  )
  
  contour(
    X1, X2,
    matrix(grid_pred, length(X1), length(X2)),
    add=TRUE
  )
  
  points(
    grid, pch='.', col=ifelse(grid_pred==1, 'springgreen3', 'tomato')
  )
  
  points(
    set[,1:2],
    pch=21,
    bg=ifelse(set$Purchased==1, 'green4', 'red3')
  )
}

# ==================== Visualize Training Set ====================
plot_decision_boundary(dataset_train, title='Logistic Regression (Training Set)')

# ==================== Visualize Test Set ====================
plot_decision_boundary(dataset_test, title='Logistic Regression (Test Set)')

