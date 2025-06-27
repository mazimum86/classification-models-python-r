# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:08:41 2025
@ Author: Chukwuka Chijioke Jerry
Description: Logistic Regression on Social Network Ads dataset
"""

# ==================== Import Libraries ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== Import Dataset ====================

dataset = pd.read_csv('../data/Social_Network_Ads.csv')

# Select Age & EstimatedSalary as features, Purchased as target
X = dataset.iloc[:,[0,1] ].values
y = dataset.iloc[:, -1].values

# ==================== Split into Train/Test ====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==================== Feature Scaling ====================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================== Fit Logistic Regression ====================

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# ==================== Predictions ====================

y_pred = classifier.predict(X_test)

# ==================== Evaluation ====================

print("\n=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='g',
    cmap='viridis', cbar=False,
    xticklabels=['Not Purchased', 'Purchased'],
    yticklabels=['Not Purchased', 'Purchased']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ==================== Visualization Function ====================

def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.05),
        np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.05)
    )
    plt.contourf(
        X1, X2,
        classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape),
        alpha=0.25, cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    colors = ['red', 'green']
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            color=colors[i], label=f'Class {j}'
        )
    plt.title(title)
    plt.xlabel('Age (standardized)')
    plt.ylabel('Estimated Salary (standardized)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==================== Plot Decision Boundaries ====================

plot_decision_boundary(X_train, y_train, 'Logistic Regression (Training Set)')
plot_decision_boundary(X_test, y_test, 'Logistic Regression (Test Set)')
