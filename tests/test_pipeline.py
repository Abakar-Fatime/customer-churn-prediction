"""
tests/test_pipeline.py
Minimal tests that run the pipeline on the sample dataset to ensure scripts execute.
"""
import subprocess
import os
import sys

def test_full_pipeline(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw = os.path.join(repo_root, "data", "raw", "sample_telco.csv")
    processed = tmp_path / "processed.csv"
    features = tmp_path / "features.csv"
    models_dir = tmp_path / "models"
    reports_out = tmp_path / "high_risk.csv"

    # Run data processing
    subprocess.check_call([sys.executable, os.path.join(repo_root, "src", "data_processing.py"), "--input", raw, "--output", str(processed)])
    # Feature engineering
    subprocess.check_call([sys.executable, os.path.join(repo_root, "src", "feature_engineering.py"), "--input", str(processed), "--output", str(features)])
    # Train (will be quick on tiny sample)
    subprocess.check_call([sys.executable, os.path.join(repo_root, "src", "model_training.py"), "--input", str(features), "--output_dir", str(models_dir)])
    # Score
    # Find model artifact
    model_path = os.path.join(str(models_dir), "xgboost_final.pkl")
    assert os.path.exists(model_path)
    subprocess.check_call([sys.executable, os.path.join(repo_root, "src", "scoring.py"), "--model", model_path, "--input", str(features), "--output", str(reports_out), "--threshold", "0.0"])
    assert os.path.exists(str(reports_out))
