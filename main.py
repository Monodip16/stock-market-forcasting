"""
Main CLI Pipeline for Stock Market Prediction (Financial Forecasting).

This script coordinates the entire end-to-end workflow:
1. Parse CLI arguments (--ticker, --start, --end, etc.).
2. Set random seeds across libraries for deterministic execution.
3. Download or load historical stock data via yfinance with offline fallback.
4. Stationarity analysis: rolling statistics, ADF test, and differencing.
5. Chronological 80/20 train/test data split (no temporal leakage).
6. ARIMA modeling: ACF/PACF diagnostics, AIC grid search, fit, and forecast.
7. LSTM deep learning: MinMaxScaler normalization, sequence generation, Keras training, forecast.
8. Date-aligned comparison graph saved to outputs/arima_vs_lstm.png.
9. Comparative metrics saved to outputs/metrics.csv and printed to terminal.
"""

import argparse
import sys
import os

# Ensure local src/ modules are discoverable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.utils import set_random_seeds, calculate_metrics
from src.data_loader import download_stock_data
from src.stationarity import plot_rolling_statistics, make_stationary
from src.arima_model import plot_acf_pacf_diagnostics, train_and_forecast_arima
from src.lstm_model import train_and_forecast_lstm
from src.visualize import plot_model_comparison, save_and_display_metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command line options for ticker, date ranges, and hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Stock Market Prediction: ARIMA vs. LSTM Financial Forecasting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL, NVDA)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (defaults to current date)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=60,
        help="Window sequence length (number of past days) for LSTM input",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum training epochs for LSTM neural network",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for LSTM training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across NumPy, Python, and TensorFlow",
    )
    parser.add_argument(
        "--no-plot-show",
        action="store_true",
        help="Do not invoke interactive plt.show() (recommended in headless/CLI environments)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end forecasting pipeline."""
    args = parse_arguments()

    print("=" * 70)
    print("      STOCK MARKET PREDICTION (FINANCIAL FORECASTING) PIPELINE       ")
    print("=" * 70)
    print(f" Ticker      : {args.ticker.upper()}")
    print(f" Date Range  : {args.start} to {args.end or 'Today'}")
    print(f" LSTM Window : {args.seq_length} past days")
    print(f" LSTM Epochs : {args.epochs}")
    print(f" Random Seed : {args.seed}")
    print("=" * 70)

    # 1. Reproducibility
    set_random_seeds(args.seed)

    # 2. Data Fetching
    try:
        df = download_stock_data(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            data_dir="data",
        )
    except Exception as exc:
        print(f"[FATAL] Data loading terminated with error: {exc}")
        sys.exit(1)

    closing_series = df["Close"]

    # 3. Stationarity Analysis
    print("\n--- PHASE 1: STATIONARITY ANALYSIS ---")
    plot_rolling_statistics(
        closing_series,
        window=30,
        save_path="outputs/stationarity_rolling.png",
        title=f"{args.ticker.upper()} Closing Price - 30-Day Rolling Statistics",
    )

    stationary_series, transform_type, d_order, adf_results = make_stationary(
        closing_series,
        save_path="outputs/stationarity_transformed.png",
    )

    # 4. Chronological Train / Test Split (80% Train, 20% Test)
    total_len = len(closing_series)
    split_idx = int(total_len * 0.8)
    train_series = closing_series.iloc[:split_idx]
    test_series = closing_series.iloc[split_idx:]

    print(f"\n[INFO] Chronological Split (80/20):")
    print(f"       Train set: {len(train_series)} points ({train_series.index[0].date()} to {train_series.index[-1].date()})")
    print(f"       Test set : {len(test_series)} points ({test_series.index[0].date()} to {test_series.index[-1].date()})")

    metrics_records = []
    arima_forecast = None
    lstm_forecast = None

    # 5. ARIMA Model
    print("\n--- PHASE 2: STATISTICAL ARIMA MODEL ---")
    try:
        # Diagnostic ACF/PACF on differenced series
        plot_acf_pacf_diagnostics(
            stationary_series,
            lags=30,
            save_path="outputs/acf_pacf.png",
            title_suffix=f"{args.ticker.upper()} Transformed ({transform_type})",
        )

        arima_forecast, arima_model, arima_metrics = train_and_forecast_arima(
            train_series=train_series,
            test_series=test_series,
            order=None,  # triggers grid search
            transform_type=transform_type,
        )
        metrics_records.append({
            "Model": "ARIMA",
            "MAE": arima_metrics["MAE"],
            "RMSE": arima_metrics["RMSE"],
            "MAPE": arima_metrics["MAPE"],
        })
    except Exception as arima_err:
        print(f"[WARNING] ARIMA modeling encountered an error: {arima_err}")
        print("[WARNING] Proceeding with LSTM model without ARIMA results.")

    # 6. LSTM Deep Learning Model
    print("\n--- PHASE 3: DEEP LEARNING LSTM MODEL ---")
    try:
        lstm_forecast, lstm_model, lstm_metrics, _, _ = train_and_forecast_lstm(
            series=closing_series,
            seq_length=args.seq_length,
            train_ratio=0.8,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=5,
            seed=args.seed,
        )
        metrics_records.append({
            "Model": "LSTM",
            "MAE": lstm_metrics["MAE"],
            "RMSE": lstm_metrics["RMSE"],
            "MAPE": lstm_metrics["MAPE"],
        })
    except Exception as lstm_err:
        print(f"[WARNING] LSTM modeling encountered an error: {lstm_err}")
        print("[WARNING] Proceeding without LSTM results.")

    # 7. Final Deliverable: Graph and Metrics
    print("\n--- PHASE 4: FINAL DELIVERABLE & EVALUATION ---")
    plot_model_comparison(
        actual_series=test_series,
        arima_forecast=arima_forecast,
        lstm_forecast=lstm_forecast,
        ticker=args.ticker.upper(),
        save_path="outputs/arima_vs_lstm.png",
        display=not args.no_plot_show,
    )

    if metrics_records:
        save_and_display_metrics(metrics_records, save_path="outputs/metrics.csv")
    else:
        print("[ERROR] Neither ARIMA nor LSTM produced forecasts successfully.")

    print("\n[SUCCESS] Pipeline execution complete.")


if __name__ == "__main__":
    main()
