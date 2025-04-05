import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# Load the input CSV
df = pd.read_csv("stock_inputs.csv")

# Corrected column names
input_columns = ['Price Change %', 'Volume Change %', 'MA Trend', 'RSI']
X = df[input_columns]

# Scale the input features
scaler = MinMaxScaler()
scaler.fit(X)

# Save the scaler to a file
dump(scaler, "scaler.pkl")

print("Scaler has been created and saved as scaler.pkl")
