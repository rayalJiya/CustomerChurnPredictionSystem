import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import clone
import joblib
from src.common import load_dataset, validate_dataset, load_config, ARTIFACTS_DIR, REPORTS_DIR, MODEL_PATH, METRICS_PATH, METADATA_PATH

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ])

def metrics_for(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "confusion_matrix": confusion_matrix(y, pred).tolist()
    }

def main():
    cfg = load_config()
    df = load_dataset()
    validate_dataset(df)
    target = cfg["target_column"]
    id_cols = [c for c in cfg["id_columns"] if c in df.columns]
    X = df.drop(columns=[target] + id_cols)
    y = df[target].astype(int)

    # Hold out the final test set first. Model choice is made using validation F1,
    # not test scores, to reduce test-set selection leakage.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
    	X_dev, y_dev, test_size=0.2, random_state=cfg["random_state"], stratify=y_dev
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=cfg["random_state"]),
        "Random Forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=2, n_jobs=-1, random_state=cfg["random_state"]),
        "SVM (LinearSVC)": LinearSVC(random_state=cfg["random_state"], max_iter=5000),
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)
    }
    svm_limit = int(cfg.get("svm_max_training_rows", 12000))
    validation_rows = []
    test_rows = []
    fitted_final = {}
    for name, estimator in models.items():
        print(f"\nTraining and validating {name} ...")
        X_fit, y_fit = X_train, y_train
        if name.startswith("SVM") and len(X_train) > svm_limit:
            X_fit, _, y_fit, _ = train_test_split(
                X_train, y_train, train_size=svm_limit,
                random_state=cfg["random_state"], stratify=y_train
            )
            print(f"Runtime safeguard: SVM validation fit uses a stratified {len(X_fit):,}-row subset.")
        validation_pipe = Pipeline([
            ("preprocess", build_preprocessor(X_fit)),
            ("model", clone(estimator))
        ])
        validation_pipe.fit(X_fit, y_fit)
        val_metrics = metrics_for(validation_pipe, X_val, y_val)
        validation_rows.append({
            "model": name, "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"], "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"]
        })

        # Refit each candidate on the complete development set; evaluate once on held-out test.
        X_final, y_final = X_dev, y_dev
        if name.startswith("SVM") and len(X_dev) > svm_limit:
            X_final, _, y_final, _ = train_test_split(
                X_dev, y_dev, train_size=svm_limit,
                random_state=cfg["random_state"], stratify=y_dev
            )
        final_pipe = Pipeline([
            ("preprocess", build_preprocessor(X_final)),
            ("model", clone(estimator))
        ])
        final_pipe.fit(X_final, y_final)
        test_metrics = metrics_for(final_pipe, X_test, y_test)
        test_rows.append({
            "model": name, "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"], "test_roc_auc": test_metrics["roc_auc"],
            "test_rows": len(X_test), "final_training_rows": len(X_final),
            "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"])
        })
        fitted_final[name] = (final_pipe, test_metrics)
        print("Validation metrics:", {k: round(val_metrics[k], 4) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})
        print("Held-out test metrics:", {k: round(test_metrics[k], 4) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})

    validation_df = pd.DataFrame(validation_rows).sort_values(["validation_f1", "validation_roc_auc"], ascending=False)
    comparison = pd.DataFrame(test_rows).merge(validation_df, on="model")
    # Choose using validation metrics, not held-out test metrics.
    best_name = validation_df.iloc[0]["model"]
    best_model, best_metrics = fitted_final[best_name]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "best_model_name": best_name,
        "target_column": target,
        "id_columns_excluded": id_cols,
        "feature_columns": X.columns.tolist(),
        "categorical_options": {
            col: sorted([str(v) for v in df[col].dropna().unique()])
            for col in X.select_dtypes(exclude=["number"]).columns
        },
        "metrics": {k: v for k, v in best_metrics.items()},
        "test_size": cfg["test_size"],
        "random_state": cfg["random_state"],
        "selection_rule": "Highest validation F1, with validation ROC-AUC as tie-breaker.",
        "probability_supported": hasattr(best_model.named_steps["model"], "predict_proba")
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n===== MODEL COMPARISON (HELD-OUT TEST RESULTS) =====")
    print(comparison.round(4).to_string(index=False))
    print(f"\nSelected using validation F1: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved comparison: {METRICS_PATH}")
    print("SVM uses a stratified subset for runtime; check final_training_rows when comparing models.")

if __name__ == "__main__":
    main()
