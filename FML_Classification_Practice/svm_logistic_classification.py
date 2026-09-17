"""FML classification practice: Logistic Regression and SVM.

Includes multiclass classification, feature transformation, class imbalance,
three SVM kernels and GridSearchCV hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Wine is a convenient multiclass dataset for demonstrating the classifiers.
data = load_wine(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression with scaling
logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, multi_class="auto")),
])
logistic.fit(X_train, y_train)
log_pred = logistic.predict(X_test)
print("Logistic Regression")
print(classification_report(y_test, log_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, log_pred))

# Compare the three requested SVM kernels.
for kernel in ["linear", "poly", "rbf"]:
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel=kernel)),
    ])
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    print(f"\nSVM kernel: {kernel}")
    print(classification_report(y_test, pred))

# Data transformation example.
transform_pipe = Pipeline([
    ("transform", PowerTransformer()),
    ("model", SVC(kernel="rbf")),
])
transform_pipe.fit(X_train, y_train)
print("\nPowerTransformer + RBF SVM accuracy:", transform_pipe.score(X_test, y_test))

# Hyperparameter tuning for SVM.
param_grid = {
    "model__kernel": ["linear", "poly", "rbf"],
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto"],
}

grid = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print("\nBest SVM parameters:")
print(grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)
print("Test accuracy:", grid.score(X_test, y_test))

# Class imbalance note:
# For an imbalanced dataset, use stratified splitting and evaluate with
# precision, recall and F1-score instead of accuracy alone. Class weights
# can also be used, e.g. SVC(class_weight="balanced") or
# LogisticRegression(class_weight="balanced").
