"""
High-Performance REST API & Web Dashboard Server for Stock Market Forecasting.

Serves:
- Single-page dashboard application (HTML/CSS/JS)
- Real-time / precomputed forecasting benchmark endpoints:
  /api/overview
  /api/forecasts
  /api/stationarity
  /api/acf_pacf
  /api/history
  /api/qa
  /api/outputs/<filename>
"""

import os
import sys
import json
import mimetypes
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DASHBOARD_DIR)
STATIC_DIR = os.path.join(DASHBOARD_DIR, "static")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

QA_DATA = [
    {
        "id": 1,
        "category": "Data Engineering & Splitting",
        "question": "Why did you split the dataset chronologically (80/20) instead of using random k-fold cross-validation?",
        "answer": "Financial market data is a time-dependent stochastic process where observation y_t depends on y_{t-1}, y_{t-2}, etc. If we randomly shuffle data into k-folds or train/test partitions, future market prices leak into past training samples (lookahead bias / temporal leakage). A model trained on future data would report artificially inflated, deceptive accuracy that completely collapses in real-world forward testing. Chronological splitting strictly preserves the arrow of time.",
        "key_takeaway": "Preserving the temporal sequence prevents lookahead bias and simulates true real-world forward forecasting."
    },
    {
        "id": 2,
        "category": "Econometric Modeling",
        "question": "Why did the optimal ARIMA(0, 1, 0) forecast collapse into a straight line over the test horizon?",
        "answer": "ARIMA(0, 1, 0) with d=1 represents a geometric random walk with drift: y_t = y_{t-1} + c + epsilon_t. When forecasting multiple steps ahead without receiving real-time daily actuals (recursive multi-step forecasting), the expected innovation E[epsilon_{t+h}] = 0 for all future horizons h >= 1. Hence, the forecast simply extrapolates y_{T+h} = y_T + h*c, forming a linear drift line. Classical ARIMA lacks memory gating and non-linear feature mapping, so it inevitably reverts to its long-term deterministic trajectory.",
        "key_takeaway": "Without exogenous variables or non-linear recurrent updates, multi-step ARIMA naturally converges to a constant drift trajectory."
    },
    {
        "id": 3,
        "category": "Deep Learning & Data Hygiene",
        "question": "How did you ensure zero data leakage when applying the MinMaxScaler for the LSTM?",
        "answer": "The MinMaxScaler was fitted exclusively on the 80% training partition (train_series). The minimum and maximum scaling parameters (X_min, X_max) were computed solely from training prices: X_scaled = (X - X_min) / (X_max - X_min). The test set was then transformed using those exact same training parameters without refitting. Furthermore, to forecast day 1 of the test set, the sliding window utilized the final 60 days of the training split, ensuring zero lookahead bias and 100% calendar alignment.",
        "key_takeaway": "Fit the scaler ONLY on training data; transform the test set using training parameters."
    },
    {
        "id": 4,
        "category": "Statistical Hypothesis Testing",
        "question": "What is the statistical hypothesis of the Augmented Dickey-Fuller (ADF) test, and why was differencing necessary?",
        "answer": "The ADF test assesses whether a unit root is present in the autoregressive formulation: Delta y_t = alpha + beta*t + gamma*y_{t-1} + sum(delta_i * Delta y_{t-i}) + epsilon_t. The null hypothesis H0 states gamma = 0 (unit root exists, series is non-stationary). The alternative hypothesis H1 states gamma < 0 (stationary). For raw Apple stock, the ADF statistic was +0.76 (p-value = 0.991), failing to reject H0. Non-stationary time series lead to spurious regressions with invalid t-statistics. Applying first differencing (d=1) resulted in ADF = -16.89 (p-value = 1.57e-29), decisively rejecting H0 and confirming weak stationarity.",
        "key_takeaway": "Stationarity ensures time-invariant mean and variance, preventing spurious statistical inferences."
    },
    {
        "id": 5,
        "category": "Model Evaluation",
        "question": "Why did you report MAE, RMSE, and MAPE simultaneously instead of just one metric?",
        "answer": "Each metric reveals a distinct error characteristic: 1) Mean Absolute Error (MAE) measures average error magnitude in raw dollars ($15.73 for LSTM vs $38.96 for ARIMA) with linear penalties; 2) Root Mean Squared Error (RMSE) squares errors before averaging ($20.62 for LSTM vs $53.18 for ARIMA), disproportionately penalizing large outlier errors and flash crashes; 3) Mean Absolute Percentage Error (MAPE) provides scale-independent relative accuracy (5.64% for LSTM vs 13.67% for ARIMA). Reporting all three gives a balanced, rigorous evaluation of both typical and extreme deviations.",
        "key_takeaway": "MAE reflects typical dollar deviations, RMSE penalizes dangerous outliers, and MAPE normalizes across price scales."
    },
    {
        "id": 6,
        "category": "Practical Quantitative Finance",
        "question": "Could an investor directly execute profitable algorithmic trades using this LSTM forecast in live markets?",
        "answer": "While the LSTM achieved a 5.64% MAPE and 50.5% directional accuracy, raw price forecasting alone is insufficient for guaranteed trading profitability. Real-world quantitative trading must account for transaction costs (brokerage, exchange fees), bid-ask spread slippage, market impact of orders, and regime-shifting volatility. Furthermore, stock markets adhere closely to the Semi-Strong Efficient Market Hypothesis (EMH), where price movements reflect random walk innovations. To become an actionable strategy, this model would need to be coupled with risk parity, volatility targeting, stop-loss execution, and portfolio optimization.",
        "key_takeaway": "Academic accuracy does not equate to trading alpha without factoring in slippage, execution fees, and risk management."
    },
    {
        "id": 7,
        "category": "Deep Learning Architecture",
        "question": "Why did you choose a sequence length of 60 trading days for the LSTM sliding window?",
        "answer": "Stock markets operate approximately 252 trading days per year, meaning 60 consecutive trading days corresponds precisely to one financial quarter (approx. 3 months). A 60-day window provides sufficient temporal context to capture medium-term momentum, earnings cycle announcements, and quarterly institutional rebalancing cycles, while remaining compact enough to prevent excessive memory degradation and vanishing gradients during Backpropagation Through Time (BPTT).",
        "key_takeaway": "60 trading days aligns with one financial quarter of institutional market dynamics."
    },
    {
        "id": 8,
        "category": "Comparative Analysis",
        "question": "How do LSTM memory cells solve the vanishing gradient limitation of traditional recurrent networks?",
        "answer": "Traditional RNNs suffer from vanishing/exploding gradients as the backpropagated error signal is repeatedly multiplied by the weight matrix W across T time steps. LSTMs introduce an additive Constant Error Carousel (CEC) via cell state C_t = f_t * C_{t-1} + i_t * ~C_t. Because the derivative dC_t/dC_{t-1} includes an additive component regulated by the forget gate f_t rather than a purely multiplicative matrix product, error gradients can flow backwards across hundreds of time steps without exponentially decaying.",
        "key_takeaway": "Additive cell state updates and forget gates enable uninterrupted gradient flow across extended temporal horizons."
    }
]


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handles static web requests and REST API responses."""

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dashboard development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Route API requests
        if path.startswith("/api/"):
            self.handle_api(path, parsed.query)
            return

        # Route static files
        if path == "/" or path == "/index.html":
            self.serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return

        clean_path = path.lstrip("/")
        # Check static dir
        target = os.path.join(STATIC_DIR, clean_path)
        if os.path.exists(target) and os.path.isfile(target):
            mime_type, _ = mimetypes.guess_type(target)
            self.serve_file(target, mime_type or "application/octet-stream")
            return

        # 404
        self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")

    def handle_api(self, path: str, query: str):
        """Handle REST API endpoints."""
        if path == "/api/overview":
            self.serve_overview()
        elif path == "/api/forecasts":
            self.serve_forecasts()
        elif path == "/api/stationarity":
            self.serve_stationarity()
        elif path == "/api/acf_pacf":
            self.serve_acf_pacf()
        elif path == "/api/history":
            self.serve_history()
        elif path == "/api/qa":
            self.send_json(QA_DATA)
        elif path.startswith("/api/outputs/"):
            filename = os.path.basename(path)
            target = os.path.join(OUTPUTS_DIR, filename)
            if os.path.exists(target) and os.path.isfile(target):
                mime, _ = mimetypes.guess_type(target)
                self.serve_file(target, mime or "application/octet-stream")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"Output artifact {filename} not found")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown API endpoint: {path}")

    def serve_overview(self):
        """Return executive benchmark overview metrics."""
        metrics_file = os.path.join(OUTPUTS_DIR, "model_metrics.json")
        stationarity_file = os.path.join(OUTPUTS_DIR, "stationarity_summary.json")

        metrics_data = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

        stationarity_data = {}
        if os.path.exists(stationarity_file):
            with open(stationarity_file, "r") as f:
                stationarity_data = json.load(f)

        overview = {
            "project_name": "Stock Market Prediction (Financial Forecasting)",
            "subtitle": "A Comparative Academic Benchmark: Classical Econometric ARIMA vs. Deep Learning LSTM",
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "total_samples": 2991,
            "date_range": {
                "start": "2015-01-02",
                "end": "2026-11-23"
            },
            "split": {
                "train_ratio": 0.8,
                "test_ratio": 0.2,
                "train_samples": 2392,
                "test_samples": 599,
                "train_dates": "2015-01-02 to 2024-07-05",
                "test_dates": "2024-07-08 to 2026-11-23"
            },
            "hyperparameters": {
                "lstm_seq_length": 60,
                "lstm_epochs": 25,
                "lstm_batch_size": 32,
                "lstm_architecture": "Sequential(LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(25) -> Dense(1))",
                "arima_order": "(0, 1, 0)",
                "arima_aic": 9735.41,
                "differencing_degree": 1
            },
            "metrics": metrics_data,
            "stationarity": stationarity_data,
            "data_hygiene": {
                "lookahead_bias": "Zero (Strictly chronological 80/20 train/test split)",
                "test_leakage": "Zero (MinMaxScaler fitted strictly on 80% train partition)",
                "calendar_alignment": "100% Date Alignment on exact holdout test dates"
            }
        }
        self.send_json(overview)

    def serve_forecasts(self):
        """Return daily test forecast comparisons."""
        csv_file = os.path.join(OUTPUTS_DIR, "forecast_comparison.csv")
        if not os.path.exists(csv_file):
            self.send_error(HTTPStatus.NOT_FOUND, "forecast_comparison.csv not found")
            return

        import csv
        records = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "date": row["Date"],
                    "actual": float(row["Actual"]),
                    "arima": float(row["ARIMA"]),
                    "lstm": float(row["LSTM"]),
                    "arima_residual": float(row["ARIMA_Residual"]),
                    "lstm_residual": float(row["LSTM_Residual"]),
                    "arima_abs_err": float(row["ARIMA_AbsError"]),
                    "lstm_abs_err": float(row["LSTM_AbsError"]),
                })
        self.send_json(records)

    def serve_stationarity(self):
        """Return ADF test summary."""
        stat_file = os.path.join(OUTPUTS_DIR, "stationarity_summary.json")
        if os.path.exists(stat_file):
            with open(stat_file, "r") as f:
                self.send_json(json.load(f))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "stationarity_summary.json not found")

    def serve_acf_pacf(self):
        """Return ACF/PACF lag data."""
        acf_file = os.path.join(OUTPUTS_DIR, "acf_pacf.json")
        if os.path.exists(acf_file):
            with open(acf_file, "r") as f:
                self.send_json(json.load(f))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "acf_pacf.json not found")

    def serve_history(self):
        """Return historical closing price data."""
        hist_file = os.path.join(OUTPUTS_DIR, "history_series.json")
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                self.send_json(json.load(f))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "history_series.json not found")

    def send_json(self, data):
        """Send JSON response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_file(self, file_path: str, content_type: str):
        """Serve a binary or text file."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error reading file: {e}")

    def log_message(self, format, *args):
        """Keep console logging clean and concise."""
        # Suppress routine static asset 200 GET logs to prevent terminal spam
        if args and "GET /static/" in str(args[0]):
            return
        sys.stderr.write(f"[DASHBOARD] {self.address_string()} - {format % args}\n")


def run_server(port: int = 8050, bind: str = "0.0.0.0"):
    """Launch the dashboard HTTP server."""
    server_address = (bind, port)
    try:
        httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    except OSError:
        # Fall back to alternative port if in use
        port = port + 1
        server_address = (bind, port)
        httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)

    print("=" * 75)
    print("   STOCK MARKET FORECASTING - ACADEMIC SHOWCASE DASHBOARD SERVER   ")
    print("=" * 75)
    print(f" Web Dashboard URL : http://localhost:{port}")
    print(f" Network Access    : http://127.0.0.1:{port}")
    print(f" REST API Base     : http://localhost:{port}/api/overview")
    print(" Press Ctrl+C at any time to safely shut down the dashboard server.")
    print("=" * 75)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard server shutting down gracefully...")
        httpd.server_close()
        print("[INFO] Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock Market Prediction Dashboard Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind server (default: 8050)")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    run_server(port=args.port, bind=args.bind)
