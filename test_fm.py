import inspect
import sys
import traceback

print("Testing Darts models support...")
try:
    import darts.models
    models_to_test = ["Chronos2Model", "PatchTSTModel", "TimesFM2p5Model", "TiRExModel", "NeuralForecastModel", "KANForecastModel"]
    for m in models_to_test:
        if hasattr(darts.models, m):
            model_class = getattr(darts.models, m)
            try:
                sig = inspect.signature(model_class.__init__)
                params = sig.parameters
                print(f"\n--- {m} ---")
                print(f"n_epochs supported: {'n_epochs' in params}")
                print(f"likelihood supported: {'likelihood' in params}")
                print(f"All params: {[k for k in params.keys()]}")
            except Exception as e:
                print(f"Error inspecting {m}: {e}")
        else:
            print(f"\n--- {m} --- Not found in darts.models")
except Exception as e:
    traceback.print_exc()

