# Salary Classification for Data Science Roles

This project applies machine learning techniques to classify salaries for data science-related job roles based on various job-related attributes. The aim is to predict whether an employee's salary falls into the **Low**, **Medium**, or **High** category using classification algorithms.

---

## Project Overview

- **Problem**: Salary ranges can vary significantly depending on job title, experience level, and work setting. Being able to classify salaries helps job seekers and employers make informed decisions.
- **Objective**: Build a machine learning model that accurately predicts salary categories from job-related data.
- **Approach**: Implement and evaluate three models – **Random Forest**, **Support Vector Machine (SVM)**, and **XGBoost** – using standard classification metrics.

---

## Dataset

- **Source**: Kaggle  
  [Jobs and Salaries of Employees in Data Science](https://www.kaggle.com/datasets/pabitrakumarsahoo/jobs-and-salaries-of-employess-in-data-science)
- **Key Features**:
  - Job Title
  - Experience Level
  - Employment Type
  - Work Setting
  - Salary in USD
- **Target Variable**: Salary Class (Low, Medium, High)

---

## Tools and Libraries

- Python
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib / seaborn

---

## Data Pre-processing

- Removed unnecessary columns
- Created a target variable by categorising salaries into Low, Medium, and High
- Applied label encoding and one-hot encoding for categorical data
- Standardised features using `StandardScaler`
- Used stratified train-test split to preserve class distribution

---

## Models Implemented

| Model            | Accuracy | Summary |
|------------------|----------|---------|
| XGBoost          | 25%      | Underperformed due to class imbalance |
| Random Forest    | 54%      | Improved with balanced class weights |
| SVM (Linear)     | 56%      | Best overall performance |

- Models evaluated using accuracy, precision, recall, and F1-score
- Class imbalance handled using `class_weight="balanced"` where applicable

---

## Key Findings

- Pre-processing and class balancing significantly affect model performance
- SVM performed best with scaled and balanced data
- XGBoost underperformed without hyperparameter tuning or rebalancing
- Tree-based models like Random Forest still performed reasonably well with minimal tuning
