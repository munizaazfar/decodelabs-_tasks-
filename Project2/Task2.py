import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ==========================================
# STEP 1: INPUT - Load & Understand Dataset
# ==========================================
print("--- Step 1: Loading Iris Dataset ---")
iris = load_iris()
X = iris.data  # Features (Sepal/Petal dimensions)
y = iris.target  # Classes (Setosa, Versicolor, Virginica)

print(f"Features: {iris.feature_names}")
print(f"Classes (Targets): {iris.target_names}")
print(f"Total Dataset Shape: {X.shape} (150 Samples, 4 Dimensions)\n")

# ==========================================
# STEP 2: PROCESS - Train-Test Split
# ==========================================
print("--- Step 2: Splitting Data ---")
# 80% Training aur 20% Testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")

# ==========================================
# STEP 3: PROCESS - Feature Scaling
# ==========================================
print("--- Step 3: Applying Feature Scaling (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler applied successfully (Mean = 0, Variance = 1)!\n")

# ==========================================
# STEP 4: PROCESS - Train KNN Classifier
# ==========================================
print("--- Step 4: Training KNN Model ---")
# K = 5 lekar model define aur train kar rahe hain
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("KNN Classifier model trained successfully!\n")

# ==========================================
# STEP 5: OUTPUT - Predictions & Evaluation
# ==========================================
print("--- Step 5: Model Predictions & Evaluation ---")
y_pred = knn.predict(X_test_scaled)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# 2. F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (Weighted): {f1:.4f}\n")

# Detailed Report
print("Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))