# Titanic Survival Prediction

## Overview

This project uses machine learning classification algorithms to predict whether a passenger survived the Titanic disaster based on passenger information.

The project focuses on the end-to-end machine learning workflow, from data cleaning and feature engineering to model training, evaluation, and model saving.

## Objectives

- Clean and prepare the Titanic dataset for machine learning.
- Engineer meaningful features to improve predictive performance.
- Train and compare multiple classification algorithms.
- Evaluate model performance using several classification metrics and cross-validation.
- Select and save the best-performing model.

## Dataset

The project uses the Titanic passenger dataset, containing information such as passenger class, sex, age, fare, family relationships, and survival status.

**Target variable:** `Survived`

- `0` — Did not survive
- `1` — Survived

## Data Cleaning and Feature Engineering

The following preprocessing steps were performed:

- Extract passenger titles from the `Name` column.
- Group rare titles and standardize similar titles.
- Handle missing `Age` values using title-based median imputation.
- Fill missing `Embarked` values using the mode.
- Fill missing `Fare` values using the median.
- Create `FamilySize` by combining `SibSp` and `Parch`.
- Create `IsAlone` to identify passengers travelling alone.
- Categorize passengers into age groups using `AgeGroup`.
- Drop irrelevant or redundant columns, including `PassengerId`, `Ticket`, `Cabin`, and `Name`.
- Split the dataset into training and testing sets using stratified sampling.
- Apply one-hot encoding to categorical features.
- Standardize numerical features for algorithms that benefit from scaling.

## Machine Learning Algorithms

Four classification algorithms were trained and compared:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Random Forest**

Feature scaling was applied for Logistic Regression and KNN, while Decision Tree and Random Forest were trained on unscaled numerical features.

## Model Evaluation

Model performance was evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions. |
| Precision | Measures how reliable positive survival predictions are. |
| Recall | Measures how many actual survivors are correctly identified. |
| F1-score | Balances precision and recall. |
| ROC-AUC | Measures how well the model distinguishes between the two classes. |
| Cross-validation accuracy | Assesses performance across multiple training folds. |

The models were evaluated on a held-out test set, with stratified five-fold cross-validation used to compare their performance.

### Results

| Algorithm | Accuracy | F1-score | ROC-AUC | CV Mean Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.849| 0.797 | 0.878 | 0.816 |
| KNN | 0.816 | 0.748 | 0.862 | 0.803 |
| Decision Tree | 0.810 | 0.746 | 0.788 |0.768 |
| Random Forest | 0.810 | 0.750 | 0.824 | 0.803 |

**Best-performing model:** Logistic Regression

The selected model was further evaluated using a confusion matrix and classification report, then saved for future use.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
titanic-survival-prediction/
├── data.csv
├── Titanic_Survival_Prediction.ipynb
├── best_titanic_model.pkl
├── titanic_encoder.pkl
├── titanic_scaler.pkl
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
   ```

3. Open the project notebook:

   ```bash
   jupyter notebook
   ```

4. Run the notebook cells in order to reproduce the preprocessing, training, and evaluation steps.

## Key Learnings

This project provided practical experience in:

- Handling missing data and preparing features.
- Feature engineering for classification problems.
- Encoding categorical variables and scaling numerical features.
- Training and comparing multiple machine learning algorithms.
- Evaluating classification models using multiple metrics.
- Using cross-validation to assess model performance.
- Saving trained models for reuse.

## Future Improvements

- Improve the preprocessing workflow to prevent data leakage during cross-validation.
- Tune model hyperparameters using cross-validation.
- Explore additional feature engineering techniques.
- Build a simple application for making predictions on new passenger records.

---
## Data Source

Kaggle
