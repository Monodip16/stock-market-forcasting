"""
Stationarity Analysis and Transformation Module.

This module provides tools to analyze and ensure time series stationarity:
1. Augmented Dickey-Fuller (ADF) hypothesis test with statsmodels.
2. Rolling statistics (rolling mean and rolling standard deviation).
3. First differencing and log differencing transformations if non-stationary.
4. Publication-ready visualization of rolling statistics and transformed stationary series.

THEORETICAL EXPLANATION OF THE ADF TEST:
----------------------------------------
The Augmented Dickey-Fuller (ADF) test evaluates whether a unit root is present
in an autoregressive time series model:
  Null Hypothesis (H0): The series has a unit root (rho = 1), implying it is non-stationary
                        and its mean/variance change over time.
  Alternative Hypothesis (H1): The series has no unit root (|rho| < 1), implying it is stationary
                               with time-invariant statistical properties.

Interpretation:
- If p-value <= 0.05 (or ADF Statistic < Critical Value at 5%):
  We reject H0. The series is considered stationary.
- If p-value > 0.05 (or ADF Statistic > Critical Value at 5%):
  We fail to reject H0. The series is non-stationary and must be differenced
  before statistical modeling (e.g., ARIMA).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def perform_adf_test(series: pd.Series, series_name: str = "Series") -> dict:
    """
    Execute the Augmented Dickey-Fuller (ADF) test on a given 1D time series.

    Parameters:
    -----------
    series : pd.Series
        Time series to evaluate.
    series_name : str
        Descriptive label for output printing.

    Returns:
    --------
    dict:
        Contains adf_statistic, p_value, used_lag, nobs, critical_values, is_stationary.
    """
    clean_series = series.dropna()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(clean_series, autolag="AIC")

    adf_stat = float(result[0])
    p_val = float(result[1])
    used_lag = int(result[2])
    nobs = int(result[3])
    crit_values = {k: float(v) for k, v in result[4].items()}

    # A series is stationary if p-value <= 0.05 and ADF statistic is less than the 5% critical value
    is_stationary = (p_val <= 0.05) and (adf_stat < crit_values["5%"])

    print("=" * 65)
    print(f" Augmented Dickey-Fuller Test: {series_name}")
    print("=" * 65)
    print(f" ADF Statistic        : {adf_stat:.6f}")
    print(f" p-value              : {p_val:.6e}")
    print(f" Lags Used            : {used_lag}")
    print(f" Number of Observations: {nobs}")
    print(" Critical Values:")
    for key, val in crit_values.items():
        print(f"   {key:>4} : {val:.4f}")
    print("-" * 65)
    if is_stationary:
        print(f" Result: Reject H0 at 5% alpha. The series is STATIONARY.")
    else:
        print(f" Result: Fail to reject H0 at 5% alpha. The series is NON-STATIONARY.")
    print("=" * 65)

    return {
        "adf_statistic": adf_stat,
        "p_value": p_val,
        "used_lag": used_lag,
        "nobs": nobs,
        "critical_values": crit_values,
        "is_stationary": is_stationary,
    }


def plot_rolling_statistics(
    series: pd.Series,
    window: int = 30,
    save_path: str = "outputs/stationarity_rolling.png",
    title: str = "Stock Closing Price - Rolling Mean & Standard Deviation",
) -> None:
    """
    Plot closing price along with its rolling mean and rolling standard deviation.

    Parameters:
    -----------
    series : pd.Series
        Stock price series with DatetimeIndex.
    window : int
        Window size in days for rolling calculations (default: 30 days).
    save_path : str
        File path to save the generated figure.
    title : str
        Title for the plot.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label="Original Closing Price", color="#1f77b4", alpha=0.85, linewidth=1.2)
    plt.plot(rolling_mean.index, rolling_mean, label=f"{window}-Day Rolling Mean", color="#d62728", linewidth=1.8)
    plt.plot(rolling_std.index, rolling_std, label=f"{window}-Day Rolling Std Dev", color="#2ca02c", linestyle="--", linewidth=1.5)

    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Rolling statistics plot saved to {save_path}")


def make_stationary(
    series: pd.Series,
    save_path: str = "outputs/stationarity_transformed.png",
) -> tuple[pd.Series, str, int, dict]:
    """
    Assess stationarity of the input series. If non-stationary, apply first differencing
    or log-and-differencing until stationary, and plot the resulting stationary series.

    Parameters:
    -----------
    series : pd.Series
        Original stock closing price series.
    save_path : str
        File path to save the transformed series plot.

    Returns:
    --------
    tuple[pd.Series, str, int, dict]:
        - Transformed series (stationary)
        - Transformation name ('none', 'first_difference', 'log_difference')
        - Integration order d (0 or 1)
        - ADF test results dictionary
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Initial ADF test on raw closing price
    raw_adf = perform_adf_test(series, series_name="Raw Closing Price")

    if raw_adf["is_stationary"]:
        return series, "none", 0, raw_adf

    # First differencing: y'(t) = y(t) - y(t-1)
    diff_series = series.diff().dropna()
    diff_adf = perform_adf_test(diff_series, series_name="First Differenced Series (d=1)")

    transformed_series = diff_series
    transform_name = "first_difference"
    d_order = 1
    final_adf = diff_adf

    # If first differencing is not stationary, try log-difference
    if not diff_adf["is_stationary"]:
        log_series = np.log(series)
        log_diff_series = log_series.diff().dropna()
        log_diff_adf = perform_adf_test(log_diff_series, series_name="Log Differenced Series")

        transformed_series = log_diff_series
        transform_name = "log_difference"
        d_order = 1
        final_adf = log_diff_adf

    # Plot the transformed stationary series
    plt.figure(figsize=(12, 5))
    plt.plot(transformed_series.index, transformed_series, color="#9467bd", linewidth=1.0, label=f"Transformed ({transform_name})")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.title(f"Stationary Transformed Series ({transform_name.replace('_', ' ').title()})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Differenced Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Transformed stationary plot saved to {save_path}")

    return transformed_series, transform_name, d_order, final_adf
