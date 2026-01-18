"""
src/feature_engineering.py
Create engineered features described in the project overview:
- tenure groups
- total services subscribed
- interaction features
- simple one-hot encoding for categorical variables

Run:
python src/feature_engineering.py --input data/processed/processed.csv --output data/processed/features.csv
"""
import argparse
import os
import pandas as pd
import numpy as np

TENURE_BINS = [0, 12, 24, 48, 1000]
TENURE_LABELS = ['0-12','12-24','24-48','48+']

def tenure_group(df):
    df['tenure_group'] = pd.cut(df['tenure'], bins=TENURE_BINS, labels=TENURE_LABELS, right=False)
    return df

def count_services(df):
    services = ['PhoneService','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    # Treat 'No' and 'No internet service' as 0, Yes as 1
    def svc_val(x):
        if pd.isna(x): return 0
        if str(x).strip().lower() in ('yes','dsl','fiber optic'):
            # keep as 1 for most service flags; InternetService will be encoded differently later
            return 1
        return 0
    # For total services, count 'Yes' across the service flags (skip InternetService column to avoid double-count)
    svc_flags = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines','PhoneService']
    df['total_services'] = df[svc_flags].applymap(lambda x: 1 if str(x).strip().lower()=='yes' else 0).sum(axis=1)
    return df

def interaction_features(df):
    df['tenure_months_x_monthly'] = df['tenure'] * df['MonthlyCharges']
    return df

def encode_categoricals(df):
    # Selected categorical columns to one-hot encode (drop_first to reduce cardinality)
    cats = ['gender','Contract','PaymentMethod','InternetService','PaperlessBilling']
    df = pd.get_dummies(df, columns=[c for c in cats if c in df.columns], drop_first=True)
    return df

def pipeline(input_path, output_path):
    df = pd.read_csv(input_path)
    df = tenure_group(df)
    df = count_services(df)
    df = interaction_features(df)
    df = encode_categoricals(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pipeline(args.input, args.output)
