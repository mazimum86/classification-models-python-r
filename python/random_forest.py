# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 11:26:46 2025
@author: USER
"""

# 🌲 Random Forest Classification

# =============================
# 1. Import Required Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# =============================
# 2. Load Dataset
# =============================
dataset = pd.read_csv('../data/Social_network_ads.csv')
X = dataset.iloc[:, :-1].values  # Age and EstimatedSalary
y = dataset.iloc[:, -1].values   # Purchased

# =============================
# 3. Split Dataset
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================
# 4. Feature Scaling
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# 5. Train Random Forest Classifier
# =============================
classifier = RandomForestClassifier(n_estimators=10, random_state=0)
classifier.fit(X_train, y_train)

# =============================
# 6. Predictions and Evaluation
# =============================
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =============================
# 7. Visualization Function
# =============================
def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contourf(X1, X2, Z, alpha=0.25, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            color=('red', 'green')[i], label=f"Class {j}"
        )
    
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================
# 8. Visualize Training Set Results
# =============================
plot_decision_boundary(X_train, y_train, "Random Forest Classifier - Training Set")

# =============================
# 9. Visualize Test Set Results
# =============================
plot_decision_boundary(X_test, y_test, "Random Forest Classifier - Test Set")
