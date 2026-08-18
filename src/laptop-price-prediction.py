"""
Laptop Price Prediction
------------------------
Cleans the raw laptop spec dataset, trains and compares four regression
models (Linear Regression, Ridge, Random Forest, Gradient Boosting), and
reports 5-fold cross-validated R^2 plus held-out MAE/RMSE.

Run from the `src/` directory:
    python laptop-price-prediction.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42


def load_and_clean(path="../data/Cleaned_Laptop_data.csv"):
    df = pd.read_csv(path)

    # Fix mislabeled source columns
    df = df.rename(columns={"Apps": "ram_type", "weight": "weight_category"})

    # Normalize brand casing ('lenovo' vs 'Lenovo' were separate categories)
    df["brand"] = df["brand"].str.strip().str.title()

    # display_size should be numeric; a few rows have corrupted values
    df["display_size"] = pd.to_numeric(df["display_size"], errors="coerce")

    # Drop rows missing the target or with corrupted display_size,
    # and exact duplicates
    df = df.dropna(subset=["Price", "display_size"]).drop_duplicates()

    # 'model' has 116 near-unique values on ~830 rows -> drop (no generalizable signal)
    df = df.drop(columns=["model"])

    return df


def build_pipeline(categorical_cols, model):
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough",
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def main():
    df = load_and_clean()
    print("Clean shape:", df.shape)

    y_raw = df["Price"].values
    X = df.drop(columns=["Price"])
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    y = np.log1p(y_raw)  # Price is right-skewed

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in models.items():
        pipe = build_pipeline(categorical_cols, model)
        pipe.fit(X_train, y_train)

        pred_price = np.expm1(pipe.predict(X_test))
        r2 = r2_score(y_test_raw, pred_price)
        mae = mean_absolute_error(y_test_raw, pred_price)
        rmse = np.sqrt(mean_squared_error(y_test_raw, pred_price))
        cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

        results.append(
            {
                "model": name,
                "test_r2": round(r2, 4),
                "test_mae": round(mae, 0),
                "test_rmse": round(rmse, 0),
                "cv_r2_mean": round(cv_scores.mean(), 4),
                "cv_r2_std": round(cv_scores.std(), 4),
            }
        )
        print(
            f"{name:20s} test_R2={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  "
            f"CV_R2={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

    results_df = pd.DataFrame(results).sort_values("cv_r2_mean", ascending=False)
    print("\n=== Summary (sorted by CV R2) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
