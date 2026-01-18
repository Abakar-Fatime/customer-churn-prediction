"""
src/model_training.py
Train an XGBoost classifier with SMOTE handling and GridSearchCV.
Saves model and artifacts to output directory.

Example:
python src/model_training.py --input data/processed/features.csv --output_dir models/
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def load_data(path):
    return pd.read_csv(path)

def prepare_xy(df, target='Churn'):
    X = df.drop(columns=[target, 'customerID'], errors='ignore')
    y = df[target]
    # Drop non-numeric columns if any remain
    X = X.select_dtypes(include=[np.number])
    return X, y

def build_pipeline():
    # SMOTE will be applied before classifier in a custom pipeline (handled in fit)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0))
    ])
    return pipe

def train_and_tune(X_train, y_train):
    pipe = build_pipeline()
    param_grid = {
        'clf__max_depth': [3,5],
        'clf__learning_rate': [0.05, 0.1],
        'clf__n_estimators': [100, 200],
        'clf__min_child_weight': [1,3]
    }
    # Use GridSearchCV with 3-fold
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    # Apply SMOTE before GridSearch by wrapping training data
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    grid.fit(X_res, y_res)
    return grid

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, preds, digits=4))
    print("F1:", f1_score(y_test, preds))

def save_artifacts(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "xgboost_final.pkl"))
    # save feature names and pipeline steps
    joblib.dump(model.best_estimator_.named_steps['scaler'], os.path.join(output_dir, "scaler.pkl"))
    print(f"Saved model and artifacts to {output_dir}")

def main(args):
    df = load_data(args.input)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    grid = train_and_tune(X_train, y_train)
    print("Best params:", grid.best_params_)
    evaluate(grid, X_test, y_test)
    save_artifacts(grid, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args)
