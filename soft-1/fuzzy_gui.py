import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import Counter

# Fuzzy Logic System
price_change = ctrl.Antecedent(np.arange(-10, 11, 1), 'price_change')
volume_change = ctrl.Antecedent(np.arange(-100, 101, 10), 'volume_change')
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
ma_trend = ctrl.Antecedent(np.arange(-5, 6, 1), 'ma_trend')
action = ctrl.Consequent(np.arange(-1, 2, 1), 'action')

price_change['negative'] = fuzz.trapmf(price_change.universe, [-10, -10, -5, 0])
price_change['stable'] = fuzz.trimf(price_change.universe, [-2, 0, 2])
price_change['positive'] = fuzz.trapmf(price_change.universe, [0, 5, 10, 10])

volume_change['low'] = fuzz.trapmf(volume_change.universe, [-100, -100, -40, 0])
volume_change['medium'] = fuzz.trimf(volume_change.universe, [-20, 0, 20])
volume_change['high'] = fuzz.trapmf(volume_change.universe, [0, 40, 100, 100])

rsi['oversold'] = fuzz.trapmf(rsi.universe, [0, 0, 30, 40])
rsi['neutral'] = fuzz.trimf(rsi.universe, [35, 50, 65])
rsi['overbought'] = fuzz.trapmf(rsi.universe, [60, 70, 100, 100])

ma_trend['falling'] = fuzz.trapmf(ma_trend.universe, [-5, -5, -2, 0])
ma_trend['flat'] = fuzz.trimf(ma_trend.universe, [-1, 0, 1])
ma_trend['rising'] = fuzz.trapmf(ma_trend.universe, [0, 2, 5, 5])

action['sell'] = fuzz.trimf(action.universe, [-1, -1, 0])
action['hold'] = fuzz.trimf(action.universe, [-0.5, 0, 0.5])
action['buy'] = fuzz.trimf(action.universe, [0, 1, 1])

rules = [
    ctrl.Rule(price_change['positive'] & rsi['overbought'], action['sell']),
    ctrl.Rule(price_change['negative'] & rsi['oversold'], action['buy']),
    ctrl.Rule(price_change['stable'] & rsi['neutral'], action['hold']),
    ctrl.Rule(price_change['positive'] & volume_change['high'], action['sell']),
    ctrl.Rule(price_change['negative'] & volume_change['high'], action['buy']),
    ctrl.Rule(ma_trend['rising'] & rsi['neutral'], action['buy']),
    ctrl.Rule(ma_trend['falling'] & rsi['neutral'], action['sell']),
    ctrl.Rule(ma_trend['rising'] & price_change['positive'], action['buy']),
    ctrl.Rule(ma_trend['falling'] & price_change['negative'], action['sell']),
    ctrl.Rule(volume_change['low'] & rsi['overbought'], action['sell']),
    ctrl.Rule(volume_change['low'] & rsi['oversold'], action['buy']),
    ctrl.Rule(price_change['stable'] & ma_trend['flat'], action['hold']),
    ctrl.Rule(price_change['positive'] & rsi['neutral'] & volume_change['medium'], action['hold']),
    ctrl.Rule(price_change['negative'] & rsi['neutral'] & volume_change['medium'], action['hold']),
    ctrl.Rule(ma_trend['rising'] & rsi['oversold'], action['buy']),
    ctrl.Rule(ma_trend['falling'] & rsi['overbought'], action['sell']),
]

stock_ctrl = ctrl.ControlSystem(rules)
stock_sim = ctrl.ControlSystemSimulation(stock_ctrl)

# Track predictions for pie chart
prediction_history = []

# GUI Functions
def predict_action():
    try:
        p = float(price_entry.get())
        v = float(volume_entry.get())
        r = float(rsi_entry.get())
        m = float(ma_entry.get())

        stock_sim.input['price_change'] = p
        stock_sim.input['volume_change'] = v
        stock_sim.input['rsi'] = r
        stock_sim.input['ma_trend'] = m
        stock_sim.compute()

        result = stock_sim.output['action']
        if result > 0.3:
            msg = "Recommendation: BUY"
            prediction_history.append("Buy")
        elif result < -0.3:
            msg = "Recommendation: SELL"
            prediction_history.append("Sell")
        else:
            msg = "Recommendation: HOLD"
            prediction_history.append("Hold")

        result_label.config(text=msg)
    except Exception as e:
        messagebox.showerror("Input Error", str(e))

def show_charts():
    chart_window = tk.Toplevel(root)
    chart_window.title("Stock Prediction Charts")

    # RSI Line Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Download data and compute RSI
    data = yf.download("AAPL", start="2024-01-01", end="2024-04-01")
    data['Change'] = data['Close'].diff()

    def compute_rsi(data, window=14):
        delta = data['Change']
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = compute_rsi(data)

    axs[0].plot(data.index, data['RSI'], color='purple')
    axs[0].axhline(70, color='red', linestyle='--')
    axs[0].axhline(30, color='green', linestyle='--')
    axs[0].set_title("RSI Trend")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("RSI")

    # Pie chart
    counts = Counter(prediction_history)
    axs[1].pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140, colors=['green', 'blue', 'red'])
    axs[1].set_title("Buy/Sell/Hold Distribution")

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# GUI Layout
root = tk.Tk()
root.title("Stock Prediction Using Fuzzy Logic")
root.geometry("400x450")

tk.Label(root, text="Price Change (%)").pack()
price_entry = tk.Entry(root)
price_entry.pack()

tk.Label(root, text="Volume Change (%)").pack()
volume_entry = tk.Entry(root)
volume_entry.pack()

tk.Label(root, text="RSI (0 - 100)").pack()
rsi_entry = tk.Entry(root)
rsi_entry.pack()

tk.Label(root, text="MA Trend (5D MA - 10D MA)").pack()
ma_entry = tk.Entry(root)
ma_entry.pack()

tk.Button(root, text="Predict", command=predict_action, bg="blue", fg="white").pack(pady=10)
result_label = tk.Label(root, text="", font=('Arial', 14, 'bold'))
result_label.pack(pady=10)

tk.Button(root, text="Show Charts", command=show_charts, bg="green", fg="white").pack(pady=10)

root.mainloop()
