from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer-churn-dataset-testing-master.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = REPORTS_DIR / "model_comparison.csv"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
CONFIG_PATH = ROOT / "config.json"

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place customer-churn-dataset-testing-master.csv in the project's data folder."
        )
    return pd.read_csv(path)

def validate_dataset(df):
    if df.empty:
        raise ValueError("Dataset is empty.")
    if "Churn" not in df.columns:
        raise ValueError("Required target column 'Churn' was not found.")
    if df["Churn"].isna().any():
        raise ValueError("Target column 'Churn' contains missing values.")
    values = set(df["Churn"].unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Expected Churn encoded as 0/1; found values: {sorted(values, key=str)}")
    if len(values) != 2:
        raise ValueError(f"Expected both target classes 0 and 1; found: {values}")
