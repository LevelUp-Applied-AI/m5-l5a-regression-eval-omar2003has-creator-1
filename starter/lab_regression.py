import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, r2_score, accuracy_score, 
                             precision_score, recall_score, f1_score)

# --- Task 1: Load Data and Basic EDA ---
def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)
    print(f"Data Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("-" * 30)
    return df

# --- Task 2: Split the Data ---
def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Use stratification only for classification (churned)
    # Task 4 requires no stratification for continuous targets
    strat = y if target_col == "churned" else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    
    print(f"Split completed for target: {target_col}")
    return X_train, X_test, y_train, y_test

# --- Task 3: Logistic Regression Pipeline ---
def build_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])

def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# --- Task 4: Ridge Regression Pipeline ---
def build_ridge_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])

def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

# --- Task 5: Lasso Regularization ---
def build_lasso_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=0.1))
    ])

# --- Task 6: Cross-Validation ---
def run_cross_validation(pipeline, X_train, y_train):
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy")
    print(f"CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores

if __name__ == "__main__":
    df = load_data()
    
    # 1. Classification Task
    features = ["tenure", "monthly_charges", "total_charges", "num_support_calls", 
                "senior_citizen", "has_partner", "has_dependents"]
    
    X_train, X_test, y_train, y_test = split_data(df[features + ["churned"]], "churned")
    log_pipe = build_logistic_pipeline()
    cls_metrics = evaluate_classifier(log_pipe, X_train, X_test, y_train, y_test)
    run_cross_validation(log_pipe, X_train, y_train)

    # 2. Regression Task (Predicting monthly_charges)
    reg_features = [f for f in features if f != "monthly_charges"]
    X_tr, X_te, y_tr, y_te = split_data(df[reg_features + ["monthly_charges"]], "monthly_charges")
    
    ridge_pipe = build_ridge_pipeline()
    lasso_pipe = build_lasso_pipeline()
    
    print(f"\nRidge Metrics: {evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)}")
    
    # Task 5: Compare Coefficients
    lasso_pipe.fit(X_tr, y_tr)
    ridge_pipe.fit(X_tr, y_tr)
    
    print("\nFeature Coefficients (Ridge vs Lasso):")
    for feat, r_coef, l_coef in zip(reg_features, ridge_pipe['regressor'].coef_, lasso_pipe['regressor'].coef_):
        print(f"{feat}: Ridge={r_coef:.3f}, Lasso={l_coef:.3f}")
 
# --- Task 7: Summary of Findings ---
"""
1. Important Features: 'tenure' and 'num_support_calls' appear most significant for churn.
2. Performance: Logistic Regression with balanced weights provides better recall. 
   Recall is more concerning here because missing a potential churner is costlier than a false alarm.
3. Next Steps: Consider non-linear models like Random Forest and perform hyperparameter tuning.
"""