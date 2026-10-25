import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.common import load_dataset, validate_dataset, load_config, REPORTS_DIR
from train_model import build_preprocessor

def main():
    cfg = load_config()
    df = load_dataset()
    validate_dataset(df)
    X = df.drop(columns=[cfg["target_column"]] + [c for c in cfg["id_columns"] if c in df.columns])
    y = df[cfg["target_column"]].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y
    )
    pipe = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", RandomForestClassifier(random_state=cfg["random_state"], n_jobs=-1))
    ])
    params = {
        "model__n_estimators": [100, 180, 250],
        "model__max_depth": [None, 10, 18, 26],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.8]
    }
    cv = StratifiedKFold(n_splits=int(cfg.get("tuning_cv", 3)), shuffle=True, random_state=cfg["random_state"])
    search = RandomizedSearchCV(pipe, params, n_iter=8, scoring="f1", cv=cv, n_jobs=-1, random_state=cfg["random_state"], verbose=1, refit=True)
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_test)
    proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    result = {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba))
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_json(REPORTS_DIR / "tuning_results.json", orient="records", indent=2)
    print(pd.Series(result).to_string())
    print(f"\nTuning report saved to {REPORTS_DIR / 'tuning_results.json'}")
    print("Tuning is a separate experiment; compare it with the baseline before replacing the saved app model.")

if __name__ == "__main__":
    main()
