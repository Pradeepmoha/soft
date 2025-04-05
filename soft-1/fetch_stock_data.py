import yfinance as yf
import pandas as pd
import datetime

# Settings
symbol = 'AAPL'  # You can change this to any stock symbol like 'TSLA', 'INFY.NS', etc.
start_date = datetime.datetime.now() - datetime.timedelta(days=90)  # last 3 months
end_date = datetime.datetime.now()

# Fetch historical stock data
data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

# Drop rows with missing values
data.dropna(inplace=True)

# Calculate indicators
data['Price Change %'] = data['Close'].pct_change() * 100
data['Volume Change %'] = data['Volume'].pct_change() * 100
data['MA Trend'] = data['Close'] - data['Close'].rolling(window=5).mean()
data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean() /
                                data['Close'].pct_change().rolling(14).std()))

# Drop NA values created by rolling calculations
data.dropna(inplace=True)

# Select the latest row to use for prediction
latest = data.iloc[-1][['Price Change %', 'Volume Change %', 'MA Trend', 'RSI']]
latest_df = pd.DataFrame([latest])

# Save to CSV
latest_df.to_csv('stock_inputs.csv', index=False)
print("Data saved to stock_inputs.csv")
