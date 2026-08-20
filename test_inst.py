import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

try:
    from darts.models import Chronos2Model, TimesFM2p5Model
    from darts.utils.likelihood_models import QuantileRegression

    print("Instantiating Chronos2Model with n_epochs=30...")
    m1 = Chronos2Model(input_chunk_length=24, output_chunk_length=24, n_epochs=30, likelihood=QuantileRegression())
    print("Success Chronos!")

except Exception as e:
    print(f"Error Chronos: {e}")

try:
    print("Instantiating TimesFM2p5Model with n_epochs=30...")
    m2 = TimesFM2p5Model(input_chunk_length=24, output_chunk_length=24, n_epochs=30, likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]))
    print("Success TimesFM!")
except Exception as e:
    print(f"Error TimesFM: {e}")
