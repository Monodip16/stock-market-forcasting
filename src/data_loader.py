"""
Data Loader Module for Stock Market Historical Data.

Handles:
- Downloading daily OHLCV data using yfinance for a given ticker and date range.
- Correctly resolving yfinance MultiIndex column structures into flat columns.
- Handling auto-adjust parameters and utilizing Adjusted Close consistently.
- Offline fallback to a cached CSV in data/ directory if internet or download fails.
- Saving raw downloaded data to data/<TICKER>_raw.csv.
"""

import os
from datetime import datetime
import pandas as pd
import yfinance as yf


def clean_yfinance_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten and standardize yfinance output dataframe:
    - Resolves MultiIndex columns (Price, Ticker) into single-level column names.
    - Normalizes column names (e.g., 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume').
    - Ensures 'Close' reflects Adjusted Close if available.
    - Guarantees 'Close' is a 1D numeric Series.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe returned from yfinance.
    ticker : str
        Stock ticker symbol.

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with DatetimeIndex and numeric columns.
    """
    if df.empty:
        return df

    cleaned = df.copy()

    # Handle MultiIndex columns (common in newer yfinance versions)
    if isinstance(cleaned.columns, pd.MultiIndex):
        ticker_upper = ticker.upper()
        # Case 1: Ticker is in level 1 (e.g. ('Close', 'AAPL'))
        if cleaned.columns.nlevels >= 2:
            if ticker_upper in cleaned.columns.get_level_values(1):
                cleaned = cleaned.xs(ticker_upper, axis=1, level=1)
            elif ticker_upper in cleaned.columns.get_level_values(0):
                cleaned = cleaned.xs(ticker_upper, axis=1, level=0)
            else:
                # Fall back to taking level 0 names
                cleaned.columns = [col[0] for col in cleaned.columns]

    # Normalize column names: strip whitespace
    cleaned.columns = [str(c).strip() for c in cleaned.columns]

    # If Adjusted Close is present, use it consistently for 'Close' as required
    if "Adj Close" in cleaned.columns:
        cleaned["Close"] = cleaned["Adj Close"]
    elif "Adj_Close" in cleaned.columns:
        cleaned["Close"] = cleaned["Adj_Close"]

    # Ensure DatetimeIndex
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index)

    cleaned.index.name = "Date"
    cleaned = cleaned.sort_index()

    # Ensure 'Close' is strictly numeric and 1-dimensional
    cleaned["Close"] = pd.to_numeric(cleaned["Close"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Close"])

    return cleaned


def download_stock_data(
    ticker: str = "AAPL",
    start: str = "2015-01-01",
    end: str | None = None,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Download daily historical OHLCV data using yfinance.

    If internet connection fails or the download encounters an exception,
    falls back to a local CSV in `data/<TICKER>_raw.csv` or `data/<TICKER>.csv`.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOGL').
    start : str
        Start date in 'YYYY-MM-DD' format (default: '2015-01-01').
    end : str or None
        End date in 'YYYY-MM-DD' format. If None, defaults to current date.
    data_dir : str
        Relative or absolute path to local data storage directory.

    Returns:
    --------
    pd.DataFrame
        Processed historical stock price dataframe.

    Raises:
    -------
    FileNotFoundError / RuntimeError:
        If download fails and no local cached CSV exists.
    """
    ticker = ticker.strip().upper()
    os.makedirs(data_dir, exist_ok=True)
    raw_csv_path = os.path.join(data_dir, f"{ticker}_raw.csv")
    alt_csv_path = os.path.join(data_dir, f"{ticker}.csv")

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    df = None
    download_error = None

    print(f"[INFO] Fetching historical data for {ticker} from {start} to {end}...")

    # Attempt online download via yfinance
    try:
        # Explicitly pass auto_adjust=False to control adjusted close handling consistently
        df_raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if df_raw is not None and not df_raw.empty:
            # Save raw downloaded data directly before processing
            df_raw.to_csv(raw_csv_path)
            print(f"[INFO] Raw downloaded data saved successfully to {raw_csv_path}")
            df = clean_yfinance_dataframe(df_raw, ticker)
        else:
            download_error = f"yfinance returned empty dataset for ticker {ticker}."
    except Exception as exc:
        download_error = str(exc)

    # If online download succeeded and produced data, return
    if df is not None and not df.empty:
        print(f"[INFO] Successfully loaded {len(df)} records for {ticker} ({df.index[0].date()} to {df.index[-1].date()}).")
        return df

    # Offline Fallback Handling
    print(f"[WARNING] Online download failed ({download_error}). Attempting local CSV fallback...")

    fallback_file = None
    if os.path.exists(raw_csv_path):
        fallback_file = raw_csv_path
    elif os.path.exists(alt_csv_path):
        fallback_file = alt_csv_path

    if fallback_file is not None:
        print(f"[INFO] Fallback activated: Loading cached data from {fallback_file}")
        try:
            # Inspect first lines to detect whether it is a MultiIndex CSV from yfinance
            with open(fallback_file, "r") as f:
                first_line = f.readline()
                second_line = f.readline()

            if "Ticker" in second_line or "Price" in first_line:
                cached_df = pd.read_csv(fallback_file, header=[0, 1], index_col=0)
            else:
                cached_df = pd.read_csv(fallback_file, header=0, index_col=0)

            cleaned_df = clean_yfinance_dataframe(cached_df, ticker)

            # Filter by date range if available
            start_dt = pd.to_datetime(start)
            cleaned_df = cleaned_df[cleaned_df.index >= start_dt]
            if end:
                end_dt = pd.to_datetime(end)
                cleaned_df = cleaned_df[cleaned_df.index <= end_dt]

            if not cleaned_df.empty:
                print(f"[INFO] Loaded {len(cleaned_df)} records from offline cache.")
                return cleaned_df
        except Exception as read_exc:
            print(f"[ERROR] Could not parse fallback file {fallback_file}: {read_exc}")

    raise RuntimeError(
        f"Failed to download data for ticker '{ticker}' and no valid local CSV fallback "
        f"was found in '{data_dir}/'. Please ensure internet connectivity or place '{ticker}_raw.csv' in '{data_dir}/'."
    )
