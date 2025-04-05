from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io

import fuzzy_logic  # Your fuzzy logic system
import anfis_predict  # Your ANFIS prediction module

app = Flask(__name__)

def parse_volume(value):
    value = str(value).strip().upper().replace(',', '')
    try:
        if value.endswith("M"):
            return float(value[:-1]) * 1_000_000
        elif value.endswith("B"):
            return float(value[:-1]) * 1_000_000_000
        elif value.endswith("K"):
            return float(value[:-1]) * 1_000
        else:
            return float(value)
    except:
        raise ValueError(f"Invalid volume format: {value}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        method = request.form.get('method', 'both')  # Get selected prediction method
        file = request.files['stock_file']

        if file.filename == '':
            return "No file selected!", 400

        try:
            # Read CSV file
            df = pd.read_csv(file)

            # Ensure required columns
            required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(df.columns):
                return "Invalid CSV format!", 400

            # Convert and sort dates
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values(by='Date')

            # Save for plotting
            df.to_csv("static/last_stock_data.csv", index=False)

            # Extract latest row
            latest = df.iloc[-1]
            close = float(latest["Close"])
            volume = parse_volume(latest["Volume"])
            high = float(latest["High"])
            low = float(latest["Low"])

            # Predict based on selected method
            prediction_fuzzy = fuzzy_logic.predict(close, volume, high, low) if method in ['fuzzy', 'both'] else None
            prediction_anfis = anfis_predict.predict(close, volume, high, low) if method in ['anfis', 'both'] else None

            return render_template(
                "result.html",
                stock_name=stock_name,
                prediction_fuzzy=prediction_fuzzy,
                prediction_anfis=prediction_anfis
            )

        except Exception as e:
            return f"Error processing file: {str(e)}", 500

    return render_template("index.html")

@app.route('/plot.png')
def plot():
    try:
        df = pd.read_csv("static/last_stock_data.csv")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Date"], df["Close"], label="Stock Closing Price", color='blue', marker='o', linestyle='-')
        ax.set_title("📈 Stock Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.legend()
        plt.xticks(rotation=45)

        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches="tight")
        img.seek(0)
        return send_file(img, mimetype='image/png')

    except Exception as e:
        return f"Error generating plot: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
