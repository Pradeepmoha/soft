import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

# Load input data
data = pd.read_csv("stock_inputs.csv")

# Setup fuzzy variables
price_change = ctrl.Antecedent(np.arange(-10, 11, 1), 'price_change')
volume_change = ctrl.Antecedent(np.arange(-100, 101, 10), 'volume_change')
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
ma_trend = ctrl.Antecedent(np.arange(-5, 6, 1), 'ma_trend')
action = ctrl.Consequent(np.arange(-1, 2, 1), 'action')

# Membership functions
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

# Fuzzy rules
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

# Create fuzzy system
stock_ctrl = ctrl.ControlSystem(rules)

# Run for multiple rows
recommendations = []

for index, row in data.iterrows():
    if row.isnull().any():
        print(f"⚠️ Skipping row {index} due to missing values.")
        recommendations.append("Error")
        continue

    stock_sim = ctrl.ControlSystemSimulation(stock_ctrl)  # Reset for each row

    try:
        stock_sim.input['price_change'] = row['Price Change %']
        stock_sim.input['volume_change'] = row['Volume Change %']
        stock_sim.input['rsi'] = row['RSI']
        stock_sim.input['ma_trend'] = row['MA Trend']
        stock_sim.compute()

        output = stock_sim.output['action']
        rec = "Buy" if output > 0.3 else "Sell" if output < -0.3 else "Hold"
        recommendations.append(rec)

    except Exception as e:
        print(f"❌ Error in row {index}: {e}")
        recommendations.append("Error")

# Save output
data['Recommendation'] = recommendations
data.to_csv("fuzzy_output.csv", index=False)
print("\n✅ Prediction Completed. Sample Output:")
print(data[['Price Change %', 'Volume Change %', 'RSI', 'MA Trend', 'Recommendation']].head())
