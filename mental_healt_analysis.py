import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
import joblib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
DATA_PATH = Path("mental_health_dataset.csv")
TARGET = "mental_health_risk"
LEAKAGE_COLS = ["depression_score", "anxiety_score"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – check the path or filename.")
    return pd.read_csv(path)


def remove_leakage(df: pd.DataFrame, leakage_cols=LEAKAGE_COLS) -> pd.DataFrame:
    present = [c for c in leakage_cols if c in df.columns]
    if present:
        print(f"Dropping leakage columns: {present}")
    return df.drop(columns=present, errors="ignore")


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", cat_transformer, cat_features),
    ])


def get_models():
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    return {"log_reg": log_reg, "rf": rf}


def get_param_grids():
    log_reg_grid = {
        "model__C": np.logspace(-3, 2, 6),
    }
    rf_grid = {
        "model__n_estimators": [200, 400, 600, 800, 1000],
        "model__max_depth": [None, 10, 20, 30, 50],
        "model__min_samples_leaf": [1, 2, 4],
    }
    return {"log_reg": log_reg_grid, "rf": rf_grid}


def fit_and_tune(name: str, model, param_grid, preprocessor, X_train, y_train):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"{name}: best CV macro‑F1 = {gs.best_score_:.3f}")
    print(" best params:", gs.best_params_)
    return gs.best_estimator_, gs.best_score_


def evaluate(estimator, X, y, label="test"):
    y_pred = estimator.predict(X)
    print(f"\n=== {label.upper()} METRICS ===")
    print(classification_report(y, y_pred))
    print("Cohen’s κ:", cohen_kappa_score(y, y_pred).round(3))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    return f1_score(y, y_pred, average="macro")


def explain(estimator, X_sample):
    import shap
    import matplotlib.pyplot as plt

    model = estimator.named_steps["model"]
    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model.predict_proba, estimator.named_steps["pre"].transform)

    shap_values = explainer(estimator.named_steps["pre"].transform(X_sample))
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    plt.close()
    print("Saved global importance plot to shap_summary.png")


def main():
    df = load_data()
    df = remove_leakage(df)

    print("Dataset shape after leakage removal:", df.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    models = get_models()
    grids = get_param_grids()

    best_estimators = {}
    best_scores = {}

    for name, model in models.items():
        est, cv_score = fit_and_tune(name, model, grids[name], preprocessor, X_train, y_train)
        val_macro_f1 = evaluate(est, X_val, y_val, label="validation")
        best_estimators[name] = est
        best_scores[name] = val_macro_f1

    chosen_name = max(best_scores, key=best_scores.get)
    final_estimator = best_estimators[chosen_name]
    print(f"\nSelected model: {chosen_name} (val macro‑F1 = {best_scores[chosen_name]:.3f})")

    evaluate(final_estimator, X_test, y_test, label="test")

    joblib.dump(final_estimator, "mental_health_model.pkl")
    print("Model saved to mental_health_model.pkl")

    sample = X_train.sample(n=min(500, len(X_train)), random_state=RANDOM_STATE)
    explain(final_estimator, sample)


if __name__ == "__main__":
    main()
