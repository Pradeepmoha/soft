from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import fuzzy_logic  # Import your fuzzy logic processing

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        file = request.files['stock_file']

        if file.filename == '':
            return "No file selected!", 400
        
        try:
            # Read CSV file with date parsing
            df = pd.read_csv(file)
            
            # Ensure CSV contains required columns
            required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(df.columns):
                return "Invalid CSV format!", 400

            # Convert Date column with day-first format
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')

            # Drop any rows with invalid dates
            df = df.dropna(subset=['Date'])

            # Sort data by date
            df = df.sort_values(by='Date')

            # Extract last row for fuzzy logic prediction
            latest = df.iloc[-1]
            prediction = fuzzy_logic.predict(latest["Close"], latest["Volume"], latest["High"], latest["Low"])

            # Save processed data for graphing
            df.to_csv("static/last_stock_data.csv", index=False)

            return render_template("result.html", stock_name=stock_name, prediction=prediction)

        except Exception as e:
            return f"Error processing file: {str(e)}", 500

    return render_template("index.html")

@app.route('/plot.png')
def plot():
    try:
        df = pd.read_csv("static/last_stock_data.csv")

        # Ensure Date column is properly parsed
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
