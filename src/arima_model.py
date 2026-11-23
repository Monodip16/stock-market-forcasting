"""
ARIMA (Autoregressive Integrated Moving Average) Modeling Module.

This module provides:
1. Autocorrelation (ACF) and Partial Autocorrelation (PACF) plotting.
2. Grid search over (p, d, q) orders based on the Akaike Information Criterion (AIC)
   fitted strictly on the training partition.
3. Model fitting on training data and multi-step forecasting for the test horizon.
4. Scale inversion (e.g. log / differencing reconstruction) back to original price levels.
5. Evaluation metrics (MAE, RMSE) on the hold-out test set.

THEORY:
-------
- p (AR order): Number of lag observations included in the model.
- d (Differencing degree): Number of times raw observations are differenced to induce stationarity.
- q (MA order): Size of the moving average window representing lagged forecast errors.
- AIC (Akaike Information Criterion): Balances goodness-of-fit against model complexity (lower is better).
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from src.utils import calculate_metrics


def plot_acf_pacf_diagnostics(
    series: pd.Series,
    lags: int = 30,
    save_path: str = "outputs/acf_pacf.png",
    title_suffix: str = "Differenced Series",
) -> None:
    """
    Generate and save ACF (Autocorrelation) and PACF (Partial Autocorrelation) plots.

    These diagnostics guide hyperparameter selection:
    - ACF cut-off after lag q suggests an MA(q) process.
    - PACF cut-off after lag p suggests an AR(p) process.

    Parameters:
    -----------
    series : pd.Series
        Stationary series (typically differenced) to analyze.
    lags : int
        Number of lags to calculate (default: 30).
    save_path : str
        Destination path for saving the diagnostic figure.
    title_suffix : str
        Descriptor added to plot titles.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    clean = series.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(clean, lags=lags, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Autocorrelation Function (ACF) - {title_suffix}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lags", fontsize=10)
    axes[0].set_ylabel("Autocorrelation", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    plot_pacf(clean, lags=lags, ax=axes[1], color="#2ca02c", method="ywm")
    axes[1].set_title(f"Partial Autocorrelation (PACF) - {title_suffix}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lags", fontsize=10)
    axes[1].set_ylabel("Partial Autocorrelation", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] ACF and PACF diagnostic plots saved to {save_path}")


def arima_grid_search(
    train_series: pd.Series,
    p_values: tuple[int, ...] = (0, 1, 2),
    d_values: tuple[int, ...] = (0, 1),
    q_values: tuple[int, ...] = (0, 1, 2),
) -> tuple[tuple[int, int, int], float, pd.DataFrame]:
    """
    Perform a grid search over (p, d, q) orders using AIC on the training dataset.

    Parameters:
    -----------
    train_series : pd.Series
        Training price series.
    p_values : tuple of ints
        Candidate AR orders.
    d_values : tuple of ints
        Candidate differencing orders.
    q_values : tuple of ints
        Candidate MA orders.

    Returns:
    --------
    tuple[tuple[int, int, int], float, pd.DataFrame]:
        - Best (p, d, q) order minimizing AIC
        - Best AIC value
        - DataFrame of all evaluated configurations and their AIC
    """
    best_aic = float("inf")
    best_order = (1, 1, 1)
    results_list = []
    y_vals = np.asarray(train_series, dtype=float).flatten()

    print("[INFO] Starting ARIMA grid search over (p, d, q) configurations on training set...")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Skip (0, 0, 0) as it represents a trivial white-noise model
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(y_vals, order=(p, d, q))
                        fitted = model.fit()
                        aic = float(fitted.aic)

                        results_list.append({"Order": f"({p},{d},{q})", "p": p, "d": d, "q": q, "AIC": aic})

                        if aic < best_aic and not np.isnan(aic):
                            best_aic = aic
                            best_order = (p, d, q)
                except Exception:
                    # Ignore non-convergent or non-invertible parameter configurations
                    continue

    results_df = pd.DataFrame(results_list).sort_values("AIC").reset_index(drop=True)
    print(f"[INFO] Optimal ARIMA order found: {best_order} with AIC = {best_aic:.2f}")

    return best_order, best_aic, results_df


def train_and_forecast_arima(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int] | None = None,
    transform_type: str = "none",
) -> tuple[pd.Series, ARIMAResults, dict[str, float]]:
    """
    Fit the optimal ARIMA model on the training series and forecast the test horizon.

    Handles:
    - Grid search if order is not provided.
    - Automatic integrated differencing (d) on original prices or log prices.
    - Inversion back to the original price scale.
    - Alignment to exact test dates.
    - Metric calculation (MAE, RMSE).

    Parameters:
    -----------
    train_series : pd.Series
        Training closing prices.
    test_series : pd.Series
        Hold-out test closing prices.
    order : tuple or None
        Specific (p, d, q) order or None to trigger grid search.
    transform_type : str
        'none', 'first_difference', or 'log_difference'.

    Returns:
    --------
    tuple[pd.Series, ARIMAResults, dict[str, float]]:
        - Forecasted prices as pd.Series indexed with test_series.index
        - Fitted ARIMA model results object
        - Dictionary of evaluation metrics (MAE, RMSE, MAPE)
    """
    # If log transform was indicated, train on log prices
    if transform_type == "log_difference":
        y_train = np.log(train_series)
    else:
        y_train = train_series

    if order is None:
        d_search = (1,) if transform_type != "none" else (0, 1)
        best_order, _, _ = arima_grid_search(y_train, p_values=(0, 1, 2), d_values=d_search, q_values=(0, 1, 2))
    else:
        best_order = order

    print(f"[INFO] Fitting ARIMA{best_order} on training data ({len(train_series)} points)...")
    y_vals = np.asarray(y_train, dtype=float).flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y_vals, order=best_order)
        fitted_model = model.fit()

    # Forecast horizon exactly matching test set length
    horizon = len(test_series)
    forecast_raw = fitted_model.forecast(steps=horizon)

    # Invert scale back to original closing prices if needed
    if transform_type == "log_difference":
        forecast_values = np.exp(forecast_raw)
    else:
        forecast_values = forecast_raw

    # Construct aligned Series with identical test dates
    forecast_series = pd.Series(forecast_values, index=test_series.index, name="ARIMA Forecast")

    # Compute evaluation metrics
    metrics = calculate_metrics(test_series.values, forecast_series.values)
    print(f"[INFO] ARIMA{best_order} Test Results -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

    return forecast_series, fitted_model, metrics
