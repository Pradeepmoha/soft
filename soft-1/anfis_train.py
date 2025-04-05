import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
data = pd.read_csv("stock_inputs.csv")

# Define label function (Buy = 2, Hold = 1, Sell = 0)
def label_output(row):
    if row['Price Change %'] > 2:
        return 2  # Buy
    elif row['Price Change %'] < -2:
        return 0  # Sell
    else:
        return 1  # Hold

# Apply label encoding
data['Label'] = data.apply(label_output, axis=1)

# Features and target
X = data[['Price Change %', 'Volume Change %', 'MA Trend', 'RSI']]
y = data['Label']

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build the neural model (ANFIS-like)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output: 3 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1)

# Save the trained model
model.save("anfis_model.h5")
print("✅ Model saved as 'anfis_model.h5'")
