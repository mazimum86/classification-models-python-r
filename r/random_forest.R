# Random Forest Classification

# Load dataset
dataset <- read.csv('../data/Social_network_ads.csv')

# Split dataset into training and test sets
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
dataset_train <- subset(dataset, split == TRUE)
dataset_test  <- subset(dataset, split == FALSE)

# Feature Scaling
dataset_train[,-3] <- scale(dataset_train[,-3])
dataset_test[,-3]  <- scale(dataset_test[,-3])

# Fit Random Forest model
library(randomForest)
classifier <- randomForest(
  x = dataset_train[,-3],
  y = factor(dataset_train$Purchased),
  ntree = 10
)

# Predict on test set
y_pred <- predict(classifier, newdata = dataset_test[,-3])

# Confusion Matrix
cm <- table(Actual = dataset_test$Purchased, Predicted = y_pred)
print(cm)

# Visualize decision boundary - Training Set
set <- dataset_train
X1 <- seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.01)
X2 <- seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.01)
grid_set <- expand.grid(Age = X1, EstimatedSalary = X2)
y_grid <- predict(classifier, newdata = grid_set)

# Ensure plot area is large enough
par(mfrow = c(1, 1))
par(mar = c(4, 4, 2, 1))

plot(set[,-3],
     main = 'Random Forest (Training Set)',
     xlab = 'Age',
     ylab = 'Estimated Salary',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3] == 1, 'green4', 'red3'))

# Visualize decision boundary - Test Set
set <- dataset_test
X1 <- seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.01)
X2 <- seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.01)
grid_set <- expand.grid(Age = X1, EstimatedSalary = X2)
y_grid <- predict(classifier, newdata = grid_set)

plot(set[,-3],
     main = 'Random Forest (Test Set)',
     xlab = 'Age',
     ylab = 'Estimated Salary',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3] == 1, 'green4', 'red3'))
