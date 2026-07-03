"""
Evaluation utilities: metrics and actual-vs-predicted plots shared across
all three models (baseline XGBoost, tuned XGBoost, TabPFN) so comparisons
are computed identically for each.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Correlation": np.corrcoef(y_true, y_pred)[0, 1],
    }


def print_comparison_table(results: dict) -> None:
    header = f"{'Model':<15}{'RMSE':>10}{'MAE':>10}{'Correlation':>14}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<15}{metrics['RMSE']:>10.4f}{metrics['MAE']:>10.4f}"
            f"{metrics['Correlation']:>14.4f}"
        )


def plot_actual_vs_predicted(y_true, y_pred, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(np.asarray(y_true), label="Actual", color="blue")
    plt.plot(np.asarray(y_pred), label="Predicted", color="red", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("RMS")
    plt.legend()
    plt.show()
