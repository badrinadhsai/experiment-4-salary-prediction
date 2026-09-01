# Experiment 4 - Employee Salary Prediction

A complete Machine Learning pipeline for predicting employee salaries based on demographic and professional features, with MLflow experiment tracking.

---

## Objective

The goal of this experiment is to predict employee salary using machine learning regression algorithms. The project demonstrates a full MLOps workflow including data cleaning, preprocessing, model training, evaluation, comparison, and experiment tracking with MLflow.

## Dataset

The experiment uses the **Salary Data** dataset containing employee information with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Employee age |
| Gender | Categorical | Male / Female |
| Education Level | Categorical | Bachelor's / Master's / PhD |
| Job Title | Categorical | Various job titles (174 unique) |
| Years of Experience | Numerical | Total work experience |
| Salary | Numerical | Annual salary (target variable) |

**Dataset Statistics:**
- Rows after cleaning: 325 (removed duplicates and missing values)
- Salary range: $350 - $250,000
- Features: 5 input columns, 1 target column

## Technologies Used

- **Python 3.13**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **Matplotlib** - Data visualization
- **MLflow** - Experiment tracking and model logging
- **Joblib** - Model serialization
- **JupyterLab** - Interactive development environment

## Data Preprocessing

The preprocessing pipeline includes:

1. **Duplicate Removal** - Removed duplicate rows from the dataset
2. **Missing Value Handling**:
   - Numerical columns (Age, Years of Experience): Filled with median
   - Categorical columns (Gender, Education Level, Job Title): Filled with mode
3. **Outlier Removal** - Removed rows with Salary = 350 (data entry error)
4. **Feature/Target Separation**:
   - Features: Age, Gender, Education Level, Job Title, Years of Experience
   - Target: Salary
5. **Numerical Preprocessing**: StandardScaler applied to Age and Years of Experience
6. **Categorical Preprocessing**: OneHotEncoder applied to Gender, Education Level, and Job Title (handle_unknown="ignore")

## Machine Learning Algorithms

Four regression algorithms were trained and evaluated:

1. **Linear Regression** - Baseline linear model
2. **Decision Tree Regressor** - Tree-based model (max_depth=10)
3. **Random Forest Regressor** - Ensemble of decision trees (100 estimators)
4. **Gradient Boosting Regressor** - Sequential ensemble boosting (100 estimators, learning_rate=0.05)

## Evaluation Metrics

- **MAE (Mean Absolute Error)** - Average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error)** - Square root of average squared differences
- **R2 Score** - Proportion of variance explained by the model (1.0 = perfect)

## Model Comparison

| Model | R2 Score | MAE | RMSE |
|-------|----------|-----|------|
| **Gradient Boosting** | **0.9354** | 8,979 | 12,145 |
| Random Forest | 0.9310 | 8,505 | 12,555 |
| Linear Regression | 0.9166 | 10,056 | 13,799 |
| Decision Tree | 0.8632 | 11,355 | 17,671 |

## Best Model

**Gradient Boosting** achieved the highest R2 score of **0.9354**, explaining approximately 93.5% of the variance in employee salaries.

- R2 Score: 0.9354
- MAE: $8,979
- RMSE: $12,145

## MLflow Tracking

MLflow was used for experiment tracking and model logging:

- **Experiment name**: salary_prediction
- **Tracking URI**: Local file store (`salary_mlops/data/mlruns/`)
- **Logged metrics**: MAE, RMSE, R2
- **Logged model**: Best pipeline (preprocessor + regressor)
- **Run name**: Based on best algorithm

## Project Structure

```
experiment-4-salary-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── experiment-4.ipynb
├── model_results.csv
├── best_salary_model.pkl
├── salary_prediction_model.pkl
└── salary_mlops/
    └── data/
        ├── raw/
        │   └── salary_data.xlsx
        ├── processed/
        │   └── salary_cleaned.csv
        ├── reports/
        │   ├── best_metrics.json
        │   ├── eda_summary.txt
        │   ├── final_best.json
        │   ├── final_report.txt
        │   ├── leaderboard.csv
        │   └── metrics.json
        ├── src/
        │   └── salary_pipeline.py
        ├── models/
        │   └── salary_final_best.pkl
        └── features/
            └── X_test_features.csv
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/badrinadhsai/experiment-4-salary-prediction.git
   cd experiment-4-salary-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open JupyterLab**
   ```bash
   jupyter lab
   ```

4. **Run the experiment**
   - Open `experiment-4.ipynb`
   - Run all cells sequentially

5. **View MLflow tracking** (optional)
   ```bash
   mlflow ui --backend-store-uri salary_mlops/data/mlruns
   ```

## Results

The experiment successfully demonstrated that ensemble methods (Gradient Boosting, Random Forest) outperform linear models on this dataset. Gradient Boosting achieved the best performance with an R2 score of 0.9354, making it the selected model for salary prediction.

## Author

**Bandaru Badri Nadh Sai**
