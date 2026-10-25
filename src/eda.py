import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.common import load_dataset, validate_dataset, FIGURES_DIR, REPORTS_DIR

def run_eda():
    df = load_dataset()
    validate_dataset(df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save an inspection report, so findings are based on the current CSV.
    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "target_counts": {str(k): int(v) for k, v in df["Churn"].value_counts().sort_index().items()},
        "target_percent": {str(k): round(float(v * 100), 2) for k, v in df["Churn"].value_counts(normalize=True).sort_index().items()},
    }
    import json
    (REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.describe(include="all").transpose().to_csv(REPORTS_DIR / "descriptive_statistics.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Class Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
    plt.close()

    numeric = [c for c in df.select_dtypes(include="number").columns if c not in ["CustomerID"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric].corr(), cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    numeric_features = [c for c in numeric if c != "Churn"]
    for col in numeric_features:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x="Churn", y=col)
        plt.title(f"{col} by Churn")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{col.lower().replace(' ', '_')}_by_churn.png", dpi=130)
        plt.close()

    categorical = [c for c in ["Gender", "Subscription Type", "Contract Length"] if c in df.columns]
    for col in categorical:
        rates = df.groupby(col, dropna=False)["Churn"].mean().sort_values(ascending=False).reset_index()
        rates["Churn Rate (%)"] = rates["Churn"] * 100
        rates.to_csv(REPORTS_DIR / f"{col.lower().replace(' ', '_')}_churn_rates.csv", index=False)
        plt.figure(figsize=(7, 4))
        sns.barplot(data=rates, x=col, y="Churn Rate (%)")
        plt.title(f"Churn Rate by {col}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{col.lower().replace(' ', '_')}_churn_rate.png", dpi=150)
        plt.close()

    print(f"EDA completed. Figures: {FIGURES_DIR}")
    print(f"Reports: {REPORTS_DIR}")

if __name__ == "__main__":
    run_eda()
