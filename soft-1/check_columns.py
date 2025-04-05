import pandas as pd

df = pd.read_csv("stock_inputs.csv")
print("Your CSV columns are:", df.columns.tolist())
