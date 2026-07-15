# AI Project 3 - AI Recommendation Logic

## 📌 Project Description
This project implements a content-based recommendation system that aligns user-defined preferences with pre-defined projects and course learning paths. It addresses the limitation of simple binary mapping (1s and 0s) by utilizing advanced statistical vectorization to analyze and suggest relevant items.

## 🚀 Recommendation Methodology (Mathematical Framework)
To overcome the issue where generic high-frequency words skew recommendations, the system uses:
* **TF-IDF Vectorization (Term Frequency - Inverse Document Frequency):** This mathematical technique downweights highly repetitive, generic stopwords (e.g., "want", "build", "to") and scales up rare, highly descriptive technical terms (e.g., "Flutter", "KNN", "SQL").
* **Cosine Similarity:** Calculates the cosine angle between the multi-dimensional user preference vector and item vectors in a shared semantic vocabulary space to extract matching percentages.

## 🏃‍♂️ Pipeline (IPO Framework)
* **Input:** Interactive user query input specifying skills, topics, or fields of interest.
* **Process:**
  * Fits the user query alongside pre-defined item catalogs.
  * Transforms corpus texts into vectorized TF-IDF spaces.
  * Measures the cosine distance/similarity score between vectors.
* **Output:** Displays ranked recommendation candidates sorted by their exact matching percentage, exporting outputs instantly to `results.txt`.

## 🛠️ Technologies & Libraries Used
* **Python 3**
* **scikit-learn** (`TfidfVectorizer` and `cosine_similarity` metrics)
* **numpy** (for efficient array ranking operations)
