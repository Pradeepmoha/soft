import yfinance as yf
import pandas as pd
import numpy as np

# Download stock data (e.g., Apple - AAPL)
data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")

# Calculate technical indicators
data['Price Change %'] = data['Close'].pct_change() * 100
data['Volume Change %'] = data['Volume'].pct_change() * 100
data['5D MA'] = data['Close'].rolling(window=5).mean()
data['10D MA'] = data['Close'].rolling(window=10).mean()
data['MA Trend'] = data['5D MA'] - data['10D MA']

# RSI calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data['RSI'] = calculate_rsi(data['Close'])

# Drop NaN and keep relevant columns
data = data.dropna()
data = data[['Price Change %', 'Volume Change %', 'MA Trend', 'RSI']]

# Preview
print(data.head())

# Save for fuzzy input
data.to_csv("stock_inputs.csv", index=False)
