"""
Salary Prediction MLOps Pipeline
Author: Human-style clean implementation
Dataset: Salary Data (Age, Gender, Education Level, Job Title, Years of Experience -> Salary)
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_clean_data(path):
    """Load raw data and handle missing values & outliers."""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")

    # Clean: Salary is target, drop rows where salary missing
    df = df.dropna(subset=["Salary"])

    # Remove unrealistic outlier (350 -> data entry error)
    df = df[df["Salary"] > 5000]
    
    # Fill missing values (if any)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Years of Experience"] = df["Years of Experience"].fillna(df["Years of Experience"].median())
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Education Level"] = df["Education Level"].fillna(df["Education Level"].mode()[0])
    df["Job Title"] = df["Job Title"].fillna(df["Job Title"].mode()[0])
    
    print(f"Cleaned data: {df.shape}, Salary range: {df['Salary'].min():.0f} - {df['Salary'].max():.0f}")
    return df


def create_preprocessor():
    """Create preprocessing for numeric + categorical features."""
    numeric_features = ["Age", "Years of Experience"]
    categorical_features = ["Gender", "Education Level", "Job Title"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """Train multiple models and return leaderboard."""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1000, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_pipes = {}
    
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        
        # Fit and test
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        metrics = {
            "Model": name,
            "CV_R2": round(cv_scores.mean(), 4),
            "Test_R2": round(r2_score(y_test, preds), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2)
        }
        results.append(metrics)
        trained_pipes[name] = pipe
        print(f"{name:18} | CV R2: {metrics['CV_R2']:.4f} | Test R2: {metrics['Test_R2']:.4f} | MAE: {metrics['MAE']:.0f}")

    leaderboard = pd.DataFrame(results).sort_values("Test_R2", ascending=False)
    return leaderboard, trained_pipes


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # go to project root
    # Fallback for direct run
    if not os.path.exists(os.path.join(base_dir, "salary_mlops")):
        base_dir = r"C:\Users\ASUS\Documents\Default Project\salary_mlops"
    else:
        base_dir = os.path.join(base_dir, "salary_mlops")
    
    # Ensure paths
    data_path = os.path.join(base_dir, "data/processed/salary_cleaned.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "data/processed/salary_cleaned.csv")
    
    print("="*50)
    print("SALARY PREDICTION PIPELINE")
    print("="*50)
    
    df = load_and_clean_data(data_path)
    X = df.drop("Salary", axis=1)
    y = df["Salary"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    preprocessor, num_cols, cat_cols = create_preprocessor()
    leaderboard, pipes = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    print("\n--- Leaderboard (sorted by Test R2) ---")
    print(leaderboard.to_string(index=False))
    
    # Pick best model (highest Test R2)
    best_name = leaderboard.iloc[0]["Model"]
    best_pipe = pipes[best_name]
    best_metrics = leaderboard.iloc[0].to_dict()
    
    print(f"\nBest Model: {best_name}")
    print(f"Metrics: R2={best_metrics['Test_R2']}, MAE={best_metrics['MAE']}, RMSE={best_metrics['RMSE']}")
    
    # Save artifacts
    os.makedirs(os.path.join(base_dir, "data/models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/reports"), exist_ok=True)
    
    with open(os.path.join(base_dir, "data/models/salary_final_best.pkl"), "wb") as f:
        pickle.dump(best_pipe, f)
    
    leaderboard.to_csv(os.path.join(base_dir, "data/reports/leaderboard.csv"), index=False)
    
    with open(os.path.join(base_dir, "data/reports/final_best.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    
    # Example prediction
    sample = X_test.iloc[:1]
    print("\nExample Input:")
    print(sample.to_string(index=False))
    print(f"Predicted Salary: ${best_pipe.predict(sample)[0]:,.2f}")
    print(f"Actual Salary:    ${y_test.iloc[0]:,.2f}")
    
    print("\nPipeline completed! Artifacts saved to data/models & data/reports")


if __name__ == "__main__":
    main()
