import inspect
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule

print("\n\n=== PLForecastingModule.set_predict_parameters ===")
try:
    print(inspect.getsource(PLForecastingModule.set_predict_parameters))
except Exception as e:
    print(e)
