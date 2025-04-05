import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define input variables
price_change = ctrl.Antecedent(np.arange(-10, 11, 1), 'price_change')
volume_change = ctrl.Antecedent(np.arange(-100, 101, 10), 'volume_change')
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
ma_trend = ctrl.Antecedent(np.arange(-5, 6, 1), 'ma_trend')

# Define output variable
action = ctrl.Consequent(np.arange(-1, 2, 0.01), 'action')  # High resolution

# Membership functions (Input)
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

# Membership functions (Output)
action['sell'] = fuzz.trimf(action.universe, [-1, -1, 0])
action['hold'] = fuzz.trimf(action.universe, [-0.5, 0, 0.5])
action['buy'] = fuzz.trimf(action.universe, [0, 1, 1])

# Define fuzzy rules
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

# Build Mamdani Inference System
stock_ctrl = ctrl.ControlSystem(rules)

# Function to build a fresh simulation every time
def predict(close, volume, high, low):
    # Compute derived indicators
    price_diff = ((close - low) / low) * 100
    volume_diff = ((volume - 1e6) / 1e6) * 100
    ma = (high + low + close) / 3
    trend = ((close - ma) / ma) * 10  # Scale for -5 to +5 range
    rsi_value = 50  # You can update this if you have actual RSI calculation

    # Clip values to the fuzzy universe ranges
    price_diff = np.clip(price_diff, -10, 10)
    volume_diff = np.clip(volume_diff, -100, 100)
    trend = np.clip(trend, -5, 5)
    rsi_value = np.clip(rsi_value, 0, 100)

    # Initialize new simulation instance
    sim = ctrl.ControlSystemSimulation(stock_ctrl)

    sim.input['price_change'] = price_diff
    sim.input['volume_change'] = volume_diff
    sim.input['rsi'] = rsi_value
    sim.input['ma_trend'] = trend

    sim.compute()
    fuzzy_output = sim.output['action']

    if fuzzy_output > 0.3:
        return "Buy"
    elif fuzzy_output < -0.3:
        return "Sell"
    else:
        return "Hold"
