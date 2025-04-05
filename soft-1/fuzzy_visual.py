import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define fuzzy variables
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
ma_trend = ctrl.Antecedent(np.arange(-5, 6, 1), 'ma_trend')
action = ctrl.Consequent(np.arange(-1, 2, 1), 'action')

# Membership functions
rsi['oversold'] = fuzz.trapmf(rsi.universe, [0, 0, 30, 40])
rsi['neutral'] = fuzz.trimf(rsi.universe, [35, 50, 65])
rsi['overbought'] = fuzz.trapmf(rsi.universe, [60, 70, 100, 100])

ma_trend['falling'] = fuzz.trapmf(ma_trend.universe, [-5, -5, -2, 0])
ma_trend['flat'] = fuzz.trimf(ma_trend.universe, [-1, 0, 1])
ma_trend['rising'] = fuzz.trapmf(ma_trend.universe, [0, 2, 5, 5])

action['sell'] = fuzz.trimf(action.universe, [-1, -1, 0])
action['hold'] = fuzz.trimf(action.universe, [-0.5, 0, 0.5])
action['buy'] = fuzz.trimf(action.universe, [0, 1, 1])

# Rules (only based on rsi and ma_trend)
rules = [
    ctrl.Rule(rsi['overbought'] & ma_trend['falling'], action['sell']),
    ctrl.Rule(rsi['oversold'] & ma_trend['rising'], action['buy']),
    ctrl.Rule(rsi['neutral'] & ma_trend['flat'], action['hold']),
    ctrl.Rule(rsi['neutral'] & ma_trend['rising'], action['buy']),
    ctrl.Rule(rsi['neutral'] & ma_trend['falling'], action['sell']),
]

# Control system
stock_ctrl = ctrl.ControlSystem(rules)
stock_sim = ctrl.ControlSystemSimulation(stock_ctrl)

# Surface plot: RSI vs MA Trend
rsi_range = np.arange(0, 101, 5)
ma_range = np.arange(-5, 6, 1)
X, Y = np.meshgrid(rsi_range, ma_range)
Z = np.zeros_like(X, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        try:
            stock_sim.input['rsi'] = X[i, j]
            stock_sim.input['ma_trend'] = Y[i, j]
            stock_sim.compute()
            Z[i, j] = stock_sim.output['action']
        except:
            Z[i, j] = np.nan

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.5)
ax.set_xlabel('RSI')
ax.set_ylabel('MA Trend')
ax.set_zlabel('Action Output')
ax.set_title('Fuzzy Inference Surface: RSI vs MA Trend')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
