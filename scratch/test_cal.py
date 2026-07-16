import qlib
from qlib.data import D
from qlib.config import REG_CN
qlib.init(provider_uri='./data/qlib_data/cn_data_rolling', region=REG_CN)
cal = D.calendar(freq='day')
print("Calendar length:", len(cal))
print("First day:", cal[0])
print("Last day:", cal[-1])
