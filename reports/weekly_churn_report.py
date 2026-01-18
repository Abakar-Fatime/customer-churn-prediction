"""
reports/weekly_churn_report.py
Generate a small weekly report (CSV) listing high-risk customers and recommended actions.
This is a simple template showing how to rank by churn probability and LTV.
"""
import pandas as pd
import argparse

def generate(input_csv, output_csv, top_n=100):
    df = pd.read_csv(input_csv)
    # Example: compute a simple LTV proxy if TotalCharges exists
    if 'TotalCharges' in df.columns:
        df['LTV'] = df['TotalCharges']
    else:
        df['LTV'] = 0
    df = df.sort_values(['churn_prob','LTV'], ascending=[False, False])
    df = df.head(top_n)
    df['recommended_action'] = df.apply(lambda r: "Priority call + 20% discount" if r['churn_prob']>0.8 else "Email retention offer", axis=1)
    df.to_csv(output_csv, index=False)
    print(f"Weekly report saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()
    generate(args.input, args.output, args.top_n)
