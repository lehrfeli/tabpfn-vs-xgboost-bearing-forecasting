"""
Model training for the TabPFN vs. XGBoost bearing-RMS forecasting comparison.

Trains three models on an identical, strict time-based 80/20 split:
  1. XGBoost with default hyperparameters (baseline)
  2. XGBoost tuned via RandomizedSearchCV over a TimeSeriesSplit
  3. TabPFN (zero-shot, no tuning)

Run directly to execute the full preprocess -> features -> train -> evaluate
pipeline end to end:

    python train.py
"""

import os
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from evaluate import compute_metrics, plot_actual_vs_predicted, print_comparison_table
from features import engineer_features, split_features_target
from preprocess import load_and_preprocess

DATA_DIR = Path("data/NASA_bearing_datasest/2nd_test/2nd_test")
TRAIN_FRACTION = 0.80  # first 80% chronologically - no shuffling, avoids leakage

XGB_PARAM_GRID = {
    "n_estimators": [100, 300, 500, 1000],   # number of trees - more capacity, more overfit risk
    "max_depth": [3, 6, 9],                  # tree depth - deeper captures more complex patterns
    "learning_rate": [0.01, 0.05, 0.1, 0.3],  # per-tree contribution - lower needs more trees
    "subsample": [0.6, 0.8, 1.0],             # row sampling per tree - regularization
    "colsample_bytree": [0.6, 0.8, 1.0],      # column sampling per tree - regularization
    "min_child_weight": [1, 3, 5],            # min samples per leaf - higher is more conservative
}


def time_based_split(X, y, train_fraction: float = TRAIN_FRACTION):
    """Strict chronological split - shuffling would leak future rows into training."""
    split_idx = int(len(X) * train_fraction)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_xgboost_baseline(train_X, train_y) -> XGBRegressor:
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(train_X, train_y)
    return model


def tune_xgboost(train_X, train_y) -> XGBRegressor:
    """Walk-forward CV (expanding window, no shuffling) over the training set only."""
    tscv = TimeSeriesSplit(n_splits=5)
    base_model = XGBRegressor(random_state=42, device="cuda", tree_method="hist")

    search = RandomizedSearchCV(
        base_model,
        param_distributions=XGB_PARAM_GRID,
        n_iter=35,  # 35 random combinations
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(train_X, train_y)
    print("Best parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)
    return search.best_estimator_


def train_tabpfn(train_X, train_y):
    """
    Requires a TabPFN access token (see README.md for how to get one). The
    token is read from an environment variable rather than hardcoded, since
    it grants API access under your account and must never be committed.
    """
    from tabpfn_client import TabPFNRegressor, set_access_token

    token = os.environ.get("TABPFN_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Set the TABPFN_ACCESS_TOKEN environment variable before running "
            "TabPFN training. See README.md for how to obtain a token."
        )
    set_access_token(token)

    model = TabPFNRegressor()
    model.fit(train_X, train_y)
    return model


def main():
    df = load_and_preprocess(DATA_DIR)
    df = engineer_features(df)
    X, y = split_features_target(df)
    train_X, test_X, train_y, test_y = time_based_split(X, y)

    results = {}
    models = {}

    models["XGBoost"] = train_xgboost_baseline(train_X, train_y)
    models["XGBoost Tuned"] = tune_xgboost(train_X, train_y)
    models["TabPFN"] = train_tabpfn(train_X, train_y)

    for name, model in models.items():
        predictions = model.predict(test_X)
        results[name] = compute_metrics(test_y, predictions)
        plot_actual_vs_predicted(test_y, predictions, title=f"{name}: Actual vs Predicted RMS (Bearing 1)")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
