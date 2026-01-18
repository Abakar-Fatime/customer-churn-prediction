"""
src/data_processing.py
Simple ETL: load raw CSV, basic cleaning, save processed CSV.
Run: python src/data_processing.py --input data/raw/sample_telco.csv --output data/processed/processed.csv
"""
import argparse
import os
import pandas as pd

def load_raw(path):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    # Trim whitespace, handle TotalCharges as numeric
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Convert TotalCharges to numeric, coerce errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing TotalCharges with MonthlyCharges * tenure
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    # Standardize churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def save_processed(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")

def main(args):
    df = load_raw(args.input)
    df = basic_clean(df)
    save_processed(df, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
