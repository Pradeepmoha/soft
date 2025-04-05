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

# Drop NaN values
data = data.dropna()

# Add Label column for classification
def label_output(row):
    if row['Price Change %'] > 2:
        return 1  # Buy
    elif row['Price Change %'] < -2:
        return -1  # Sell
    else:
        return 0  # Hold

data['Label'] = data.apply(label_output, axis=1)

# Keep only relevant columns
data = data[['Price Change %', 'Volume Change %', 'MA Trend', 'RSI', 'Label']]

# Preview first few rows
print(data.head())

# Save to CSV
data.to_csv("stock_inputs.csv", index=False)
