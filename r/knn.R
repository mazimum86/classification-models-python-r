# K-Nearest Neighbors (K-NN) for Social Network Ads

# ==================== Import Dataset ====================
dataset <- read.csv('../data/social_network_ads.csv')

# ==================== Split into Training and Test Sets ====================
library(caTools)
set.seed(123)  # for reproducibility
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
dataset_train <- subset(dataset, split == TRUE)
dataset_test <- subset(dataset, split == FALSE)

# ==================== Feature Scaling ====================
dataset_train[, -3] <- scale(dataset_train[, -3])
dataset_test[, -3] <- scale(dataset_test[, -3])

# ==================== Fit K-NN Model and Predict ====================
library(class)
y_pred <- knn(
  train = dataset_train[, -3],
  test = dataset_test[, -3],
  cl = dataset_train[, 3],
  k = 5
)

# ==================== Confusion Matrix ====================
cm <- table(dataset_test$Purchased, y_pred)
print(cm)

# ==================== Decision Boundary - Training Set ====================
set <- dataset_train
X1 <- seq(min(set$Age)-1, max(set$Age)+1, by = 0.01)
X2 <- seq(min(set$EstimatedSalary)-1, max(set$EstimatedSalary)+1, by = 0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- knn(
  train = set[, -3],
  cl = set[, 3],
  k = 5,
  test = grid_set
)
par(mar = c(4.5, 4.5, 2.5, 1))  # slightly smaller margins
# Plot training set decision boundary
plot(
  set[, -3],
  main = 'K-NN (Training Set)',
  xlab = 'Age',
  ylab = 'Estimated Salary',
  xlim = range(X1),
  ylim = range(X2)
)
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch='.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set[, -3], pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# ==================== Decision Boundary - Test Set ====================
set <- dataset_test
X1 <- seq(min(set$Age)-1, max(set$Age)+1, by = 0.01)
X2 <- seq(min(set$EstimatedSalary)-1, max(set$EstimatedSalary)+1, by = 0.01)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) <- c('Age', 'EstimatedSalary')
y_grid <- knn(
  train = set[, -3],
  cl = set[, 3],
  k = 5,
  test = grid_set
)

# Plot test set decision boundary
plot(
  set[, -3],
  main = 'K-NN (Test Set)',
  xlab = 'Age',
  ylab = 'Estimated Salary',
  xlim = range(X1),
  ylim = range(X2)
)
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch='.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(
  set[, -3], pch = 21,
  bg = ifelse(set[, 3] == 1, 'green4', 'red3')
)
