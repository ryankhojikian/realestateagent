import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
METRICS_PATH = BASE_DIR / "training_metrics.json"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "KitchenQual",
    "HouseStyle",
    "LotArea",
    "Neighborhood",
    "ExterQual",
    "BedroomAbvGr",
]
NUMERIC_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "LotArea",
    "BedroomAbvGr",
]
CATEGORICAL_COLUMNS = ["KitchenQual", "HouseStyle", "Neighborhood", "ExterQual"]
TARGET_COLUMN = "SalePrice"


def build_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_COLUMNS),
            ("cat", cat_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def build_candidates():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ridge": Ridge(alpha=1.0),
    }


def calculate_metrics(y_true: pd.Series, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(mse ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_summary_stats(y_train: pd.Series):
    return {
        "median_sale_price": float(y_train.median()),
        "mean_sale_price": float(y_train.mean()),
        "typical_sale_price_range": {
            "low": float(y_train.quantile(0.25)),
            "high": float(y_train.quantile(0.75)),
        },
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    candidate_results = {}
    best_name = ""
    best_pipeline = None
    best_validation_rmse = float("inf")

    for name, estimator in build_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        train_metrics = calculate_metrics(y_train, pipeline.predict(X_train))
        validation_metrics = calculate_metrics(
            y_validation, pipeline.predict(X_validation)
        )
        candidate_results[name] = {
            "train": train_metrics,
            "validation": validation_metrics,
        }

        if validation_metrics["rmse"] < best_validation_rmse:
            best_name = name
            best_pipeline = pipeline
            best_validation_rmse = validation_metrics["rmse"]

    if best_pipeline is None:
        raise RuntimeError("No model pipeline was trained.")

    winner_test_metrics = calculate_metrics(y_test, best_pipeline.predict(X_test))
    metrics_artifact = {
        "selection_metric": "rmse",
        "feature_columns": FEATURE_COLUMNS,
        "split_sizes": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
        },
        "candidates": candidate_results,
        "winner": {
            "name": best_name,
            "train": candidate_results[best_name]["train"],
            "validation": candidate_results[best_name]["validation"],
            "test": winner_test_metrics,
        },
        "summary_stats": build_summary_stats(y_train),
    }

    joblib.dump(best_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics_artifact, indent=2), encoding="utf-8")

    print(f"Saved winning pipeline to {MODEL_PATH.name}")
    print(f"Saved training metrics to {METRICS_PATH.name}")
    print(
        f"Winner: {best_name} "
        f"(validation RMSE: {metrics_artifact['winner']['validation']['rmse']:.2f}, "
        f"test RMSE: {winner_test_metrics['rmse']:.2f})"
    )


if __name__ == "__main__":
    main()