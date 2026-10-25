import pandas as pd
from src.common import validate_dataset

def test_validate_accepts_binary_target():
    df = pd.DataFrame({"Age": [25, 40], "Churn": [0, 1]})
    validate_dataset(df)

def test_validate_rejects_missing_target():
    df = pd.DataFrame({"Age": [25], "Churn": [None]})
    try:
        validate_dataset(df)
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Expected missing target to be rejected")

def test_validate_rejects_non_binary_target():
    df = pd.DataFrame({"Age": [25, 40], "Churn": [0, 2]})
    try:
        validate_dataset(df)
    except ValueError as exc:
        assert "0/1" in str(exc)
    else:
        raise AssertionError("Expected non-binary target to be rejected")
