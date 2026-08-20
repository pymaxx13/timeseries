import darts
from darts import TimeSeries
import pandas as pd
import numpy as np
import inspect

df = pd.DataFrame({'val': np.arange(10)}, index=pd.date_range('2020-01-01', periods=10))
ts = TimeSeries.from_dataframe(df)

print("TimeSeries dir:")
print([m for m in dir(ts) if 'slice' in m or 'drop' in m])

print("\nTimeSeries specific slice methods:")
if hasattr(ts, 'slice'):
    print("slice:", inspect.signature(ts.slice))
if hasattr(ts, 'drop_after'):
    print("drop_after:", inspect.signature(ts.drop_after))
if hasattr(ts, 'drop_before'):
    print("drop_before:", inspect.signature(ts.drop_before))
