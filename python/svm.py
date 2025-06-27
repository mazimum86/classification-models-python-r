# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:22:00 2025
Author: Chukwuka Chijioke Jerry

Support Vector Classification on Social Network Ads
"""

# ===========================
# 📦 Import Libraries
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ===========================
# 📂 Load Dataset
# ===========================
dataset = pd.read_csv('../data/social_network_ads.csv')
X = dataset.iloc[:, :2].values  # Age, EstimatedSalary
y = dataset.iloc[:, -1].values  # Purchased

# ===========================
# 🔀 Split into Train/Test
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ===========================
# ⚖️ Feature Scaling
# ===========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================
# 🧠 Train SVC Model
# ===========================
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# ===========================
# 🔍 Predict & Evaluate
# ===========================
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ===========================
# 📊 Decision Boundary Plotter
# ===========================
def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contourf(
        X1, X2, Z,
        alpha=0.3,
        cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for idx, label in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            c=('red', 'green')[idx],
            label=f'Class {label}'
        )
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
   
    plt.tight_layout()
    plt.show()

# ===========================
# 🖼️ Plot for Training & Test Sets
# ===========================
plot_decision_boundary(X_train, y_train, 'SVC (Training Set)')
plot_decision_boundary(X_test, y_test, 'SVC (Test Set)')
