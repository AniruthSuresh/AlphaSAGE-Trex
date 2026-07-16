import pandas as pd
import numpy as np

# Mock df like Qlib output
idx = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=3), ['stock1', 'stock2', 'stock3']], names=['datetime', 'instrument'])
df = pd.DataFrame(np.random.randn(9, 6), index=idx, columns=['$open', '$close', '$high', '$low', '$volume', '$vwap'])

# Introduce NaNs to simulate real data
df.loc[(df.index.get_level_values(0) == '2020-01-01') & (df.index.get_level_values(1) == 'stock2'), '$open'] = np.nan

print("Original shape:", df.shape)

try:
    df1 = df.stack().unstack(level=1)
    values = df1.values
    values.reshape((-1, 6, values.shape[-1]))
    print("stack() works")
except Exception as e:
    print("stack() failed:", e)

try:
    df2 = df.stack(dropna=False).unstack(level=1)
    values2 = df2.values
    values2.reshape((-1, 6, values2.shape[-1]))
    print("stack(dropna=False) works")
except Exception as e:
    print("stack(dropna=False) failed:", e)
