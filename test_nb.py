import darts
from darts import TimeSeries
from darts.utils.likelihood_models import NegativeBinomialLikelihood
from darts.models import NBEATSModel
import pandas as pd
import numpy as np

# Create continuous non-integer data
df = pd.DataFrame({'val': [0.1, 0.5, 1.2, 3.4, 2.1, 0.8, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]}, index=pd.date_range('2020-01-01', periods=12))
ts = TimeSeries.from_dataframe(df)

model = NBEATSModel(input_chunk_length=3, output_chunk_length=2, n_epochs=1, likelihood=NegativeBinomialLikelihood())

try:
    model.fit(ts)
    print("Fit succeeded on continuous data!")
except Exception as e:
    print(f"Fit failed on continuous data: {type(e).__name__} - {e}")
