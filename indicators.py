# utils/indicators.py
import pandas as pd

def add_indicators(df):
    df = df.copy()

    if 'Close' not in df.columns:
        raise KeyError("Missing 'Close' column in data")

    # Simple Moving Average
    df['SMA_10'] = df['Close'].rolling(window=10).mean()

    # Exponential Moving Average
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # MACD and Signal Line
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Daily Returns
    df['Return'] = df['Close'].pct_change()

    # Volatility
    df['Volatility'] = df['Return'].rolling(window=10).std()

    # Drop rows with NaNs (from rolling/ewm ops)
    df = df.dropna()

    return df