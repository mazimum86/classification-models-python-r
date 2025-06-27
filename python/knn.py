# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:08:34 2025
Author: Chukwuka Chijioke Jerry

K-Nearest Neighbors (K-NN) Classifier on Social Network Ads
"""

# ==================== Import Libraries ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ==================== Import Dataset ====================
dataset = pd.read_csv('../data/social_network_ads.csv')
X = dataset.iloc[:, :2].values  # Age, EstimatedSalary
y = dataset.iloc[:, -1].values  # Purchased

# ==================== Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==================== Feature Scaling ====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================== Fit the K-NN Model ====================
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# ==================== Prediction and Evaluation ====================
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ==================== Decision Boundary Visualization ====================

def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 0.01),
        np.arange(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 0.01)
    )
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

    plt.contourf(X1, X2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            color=['red', 'green'][i], label=j
        )
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==================== Visualizing Training Set ====================
plot_decision_boundary(X_train, y_train, title='K-NN (Training Set)')

# ==================== Visualizing Test Set ====================
plot_decision_boundary(X_test, y_test, title='K-NN (Test Set)')
