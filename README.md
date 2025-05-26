# NutriYield-VML-DL
# CropNutriNet

A machine-learning pipeline for **Exploring the impact of soil nutrients on crop yield**, implemented in Python and evaluated across eight classification algorithms (Random Forest, AdaBoost, SVM, KNN, Logistic Regression, LightGBM, CatBoost, and XGBoost).

---

## 🔖 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Model Training & Evaluation](#model-training--evaluation)
7. [Results Summary](#results-summary)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

--- 
![download - 2025-05-26T231439 434](https://github.com/user-attachments/assets/c37f56ba-bf9a-4c31-9434-602ff6d27db8)
![download - 2025-05-26T231429 689](https://github.com/user-attachments/assets/d5bf3f80-981e-4948-b6e2-b9f4bc1e1f32)


## 📝 Project Overview

* **Objective:** Predict crop type from soil-nutrient measurements (N, P, K, pH, EC, Mn, Zn, Fe, S, …)
* **Approach:**

  1. Data cleaning & outlier removal
  2. Box–Cox transformation of skewed features
  3. Min–Max scaling & label encoding
  4. Dimensionality reduction for visualization (t-SNE, PCA)
  5. Training eight classifiers
  6. Evaluating with confusion matrices and accuracy

---

## ✨ Key Features

* Comprehensive **data preprocessing**: outlier detection, power transforms, scaling
* **Eight classification models** for robust comparison
* Publication-quality **confusion‐matrix visualizations**
* End‐to‐end Jupyter notebook, ready for adaptation and extension

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/CropNutriNet.git
   cd CropNutriNet
   ```
2. Create and activate a Python environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Or manually install:*

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn lightgbm catboost xgboost
   ```

---

## 📂 Dataset

* **File:** `dataset.csv`
* **Features:**

  * Soil macronutrients: N, P, K
  * Soil micronutrients: Mn, Zn, Fe, S
  * Physicochemical: pH, EC
  * Target: `label` (crop type: grapes, mango, mulberry, pomegranate, potato, ragi)
* **Source & Format:** CSV with \~X00 samples, 12 columns

---

## 🔄 Preprocessing Pipeline

1. **Train–Test Split**

   * 80 % training, 20 % testing
   * Stratified by crop label
2. **Outlier Removal**

   * IQR method on “S”
   * Remove samples flagged as outliers in >1 feature
3. **Box–Cox Transformation**

   * Applied to Mn, Zn, Fe, K, EC (λ = 0.15)
   * Reduces right skew
4. **Feature Scaling**

   * Min–Max scaling to \[0,1]
5. **Label Encoding**

   * Converts crop names → integer codes

---

## 🤖 Model Training & Evaluation

| Model               | Hyperparameters                      |
| ------------------- | ------------------------------------ |
| Random Forest       | 300 trees, max\_depth = 10           |
| AdaBoost            | 100 estimators, learning\_rate = 0.5 |
| SVM                 | C = 1, RBF kernel, probability=True  |
| KNN                 | k = 5                                |
| Logistic Regression | max\_iter = 500                      |
| LightGBM            | 200 estimators, learning\_rate = 0.1 |
| CatBoost            | 200 iterations, learning\_rate = 0.1 |
| XGBoost             | 200 estimators, learning\_rate = 0.1 |

* **Metric:** Accuracy & confusion matrix
* **Visualization:** 3×3 grid of confusion matrices for all models; standalone XGBoost matrix

---

## 📊 Results Summary

* **Overall Accuracies:**

  * Random Forest: **0.96**
  * AdaBoost: **0.94**
  * SVM: **0.96**
  * KNN: **0.96**
  * Logistic Regression: **0.95**
  * LightGBM: **0.96**
  * CatBoost: **0.96**
  * XGBoost: **0.96**

* **Common Misclassifications:**

  * *grapes* vs. *mulberry*: 1–2 samples
  * *mulberry* vs. *ragi* (XGBoost): 2 samples
  * *pomegranate* vs. *grapes* (across several models): 1 sample

---

## 🚀 Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter lab
   ```
2. Execute cells in order:

   1. Installation & imports
   2. Data loading & EDA
   3. Preprocessing
   4. Model training
   5. Evaluation plots
3. Adapt hyperparameters or add new models as needed.

---

## 🤝 Contributing

* Feel free to submit pull requests for:

  * Additional preprocessing techniques
  * Hyperparameter tuning scripts
  * New model architectures (e.g., neural networks)
* Please follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).

---

## 📄 License

This project is released under the [MIT License](LICENSE).

—
*Happy modeling!*
