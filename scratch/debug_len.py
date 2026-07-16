import qlib
from qlib.config import REG_CN
qlib.init(provider_uri='./data/qlib_data/cn_data_rolling', region=REG_CN)
from src.alphagen_qlib.stock_data import StockData
from alphagen.data.expression import Feature, FeatureType, Ref

data = StockData(instrument='csi300', start_time='2005-01-01', end_time='2017-12-31', qlib_path='./data/qlib_data/cn_data_rolling')
print(f"data.data len: {len(data.data)}")
print(f"data.max_backtrack: {data.max_backtrack_days}")
print(f"data.max_future: {data.max_future_days}")
print(f"data.n_days: {data.n_days}")
print(f"data._dates len: {len(data._dates)}")

close = Feature(FeatureType.CLOSE)
target = Ref(close, -1) / close - 1
print(f"target.evaluate(data) len: {len(target.evaluate(data))}")
