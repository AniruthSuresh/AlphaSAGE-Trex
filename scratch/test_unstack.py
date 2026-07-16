import pandas as pd
import numpy as np

idx = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=3), ['stock1', 'stock2', 'stock3']], names=['datetime', 'instrument'])
df = pd.DataFrame(np.random.randn(9, 6), index=idx, columns=['$open', '$close', '$high', '$low', '$volume', '$vwap'])
features = df.columns.tolist()

# The original way but with dropna=False
df1 = df.stack(dropna=False).unstack(level=1)
dates1 = df1.index.levels[0]
stock_ids1 = df1.columns
values1 = df1.values
res1 = values1.reshape((-1, len(features), values1.shape[-1]))

# The unstack way
df2 = df.unstack(level=1)
dates2 = df2.index
stock_ids2 = df2.columns.levels[1]
values2 = df2.values
res2 = values2.reshape((-1, len(features), len(stock_ids2)))

print("Same result?", np.allclose(res1, res2))
print("dates match?", list(dates1) == list(dates2))
print("stock_ids match?", list(stock_ids1) == list(stock_ids2))
