import tensorflow as tf
import numpy as np
from joblib import load  # For loading the original scaler

# Load trained model and scaler once globally
model = tf.keras.models.load_model("anfis_model.h5")
scaler = load("scaler.pkl")  # Make sure this file exists and was saved during training

label_map = {0: "Sell", 1: "Hold", 2: "Buy"}

def predict(close, volume, high, low):
    # Compute technical indicators
    price_change = ((close - low) / low) * 100  # % price recovery from low
    volume_change = ((volume - 1e6) / 1e6)       # Normalize volume
    ma_trend = (high + low + close) / 3          # Mid-point MA approximation
    rsi = 50  # Default RSI value; replace with real RSI if available

    # Prepare feature vector
    features = np.array([[price_change, volume_change, ma_trend, rsi]])

    # Scale features using pre-trained scaler
    scaled = scaler.transform(features)

    # Make prediction
    preds = model.predict(scaled)
    label_index = np.argmax(preds)

    return label_map[label_index]
