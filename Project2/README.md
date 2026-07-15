# AI Project 2 - Data Classification (Iris Dataset)

## 📌 Project Description
This project implements a supervised machine learning classification model on the classic **Iris Flower Dataset**. Using the **K-Nearest Neighbors (KNN)** algorithm, the model classifies iris flowers into three distinct species (`Setosa`, `Versicolor`, and `Virginica`) based on their physical dimensions (sepal and petal lengths/widths).

## 🚀 Pipeline (IPO Framework)
Following the standard Input-Process-Output (IPO) framework, the project is structured as follows:

* **Input:** * Loaded the Iris Dataset containing 150 samples, 4 features, and 3 target classes.
* **Process:**
  * **Train-Test Split:** Split the dataset into 80% Training and 20% Testing sets with randomized shuffling to eliminate any order bias.
  * **Feature Scaling:** Applied `StandardScaler` to normalize features, centering the mean at 0 with a variance of 1.
  * **Model Training:** Trained a **K-Nearest Neighbors (KNN)** classifier with $K = 5$.
* **Output:** * Generated predictions and evaluated performance using a **Confusion Matrix** and **F1-Score**.

## 📊 Results & Performance Evaluation
The model achieved perfect classification accuracy on the test dataset:
* **F1-Score (Weighted):** 1.0000 (100% Accuracy)
* **Confusion Matrix:**
  ```text
  [[10  0  0]
   [ 0  9  0]
   [ 0  0 11]]