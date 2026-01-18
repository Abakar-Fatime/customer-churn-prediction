"""
src/scoring.py
Load saved model and score incoming customers. Produces a CSV of high-risk customers.

Example:
python src/scoring.py --model models/xgboost_final.pkl --input data/processed/features.csv --output reports/high_risk_customers.csv --threshold 0.75
"""
import argparse
import os
import pandas as pd
import joblib

def load_model(path):
    return joblib.load(path)

def score(model, df, threshold=0.75):
    # Keep customerID if present
    customer_col = 'customerID' if 'customerID' in df.columns else None
    X = df.select_dtypes(include=['number'])
    # Apply scaler if present as a separate artifact (not in this simple model load)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:,1]
    else:
        # If GridSearchCV object:
        best = model.best_estimator_
        probs = best.predict_proba(X)[:,1]
    df_out = df.copy()
    df_out['churn_prob'] = probs
    high_risk = df_out[df_out['churn_prob'] >= threshold].sort_values('churn_prob', ascending=False)
    return high_risk

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_csv(args.input)
    model = load_model(args.model)
    high_risk = score(model, df, threshold=args.threshold)
    high_risk.to_csv(args.output, index=False)
    print(f"Saved {len(high_risk)} high-risk customers to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    main(args)
