# 🌸 Iris Flower Classification Dashboard (KNN)

## 📌 Project Overview
This project presents an interactive **Supervised Machine Learning Web Application** built using **Streamlit**, **Scikit-Learn**, and **Python**. The application implements the **K-Nearest Neighbors (KNN)** classification algorithm on the classic **Iris Flower Dataset** to predict plant species (`Setosa`, `Versicolor`, and `Virginica`) based on custom physical measurements (sepal & petal dimensions).

---

## ✨ Key Features & Interactive Capabilities

* **🎛️ Real-Time Hyperparameter Tuning:** Adjust the $K$-neighbors parameter ($1$ to $21$) and test-train split ratio ($10\%$ to $50\%$) on the fly via sidebar controls.
* **🔮 Live Prediction & Probability Allocation:** Slide custom feature values to instantly compute predicted species and view confidence percentages with class probability bar charts.
* **📈 Model Performance Diagnostics:**
  * Real-time calculation of **Accuracy**, **Weighted F1-Score**, and **Test Sample Counts**.
  * **Confusion Matrix** heatmap visualization.
  * **Elbow Curve Diagnostics** (Accuracy vs. $K$-value plot) to evaluate hyperparameter stability.
* **📊 2D Feature Projection Scatter Plot:** Interactive multi-dimensional scatter mapping with selectable X/Y axes and visual indicator markers for simulated user inputs.
* **🎨 Custom Dark-Slate UI/UX:** High-contrast, custom CSS styling optimized for enhanced readability, clean metric cards, and zero header clutter.

---

## 🚀 Pipeline Architecture (IPO Framework)

Following the standard **Input-Process-Output (IPO)** engineering model:

### 1. Input Data
* **Dataset:** 150 samples with 4 continuous numerical features (`Sepal Length`, `Sepal Width`, `Petal Length`, `Petal Width`) and 3 target classes.
* **User Input:** Dynamic values supplied via UI interactive sliders.

### 2. Processing Pipeline
* **Train-Test Split:** Stratified splitting based on the user-selected test set ratio (default: 80% Train / 20% Test).
* **Feature Normalization:** Applied `StandardScaler` to transform features to standard normal distribution ($\mu = 0$, $\sigma = 1$).
* **Classification Algorithm:** Trained `KNeighborsClassifier` dynamically reacting to selected $K$ values.

### 3. Output & Metrics
* Class prediction outputs, confidence percentages, Seaborn/Matplotlib visualization graphs, and performance metrics.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn

---