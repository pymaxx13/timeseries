import sys
import traceback
sys.path.append(r'C:\Users\Anusha\engression\engression-ts')

try:
    from engressionts.models.darts_original.patchtst_fm_model import PatchTSTFMModel
    from engressionts.models.darts_original.tirex_model import TiRExModel
    from engressionts.models.darts_original.chronos2_model import Chronos2Model
    from engressionts.models.darts_original.timesfm2p5_model import TimesFM2p5Model
    
    import inspect
    models = [PatchTSTFMModel, TiRExModel, Chronos2Model, TimesFM2p5Model]
    for model_class in models:
        sig = inspect.signature(model_class.__init__)
        params = sig.parameters
        print(f"\n--- {model_class.__name__} ---")
        print(f"n_epochs supported: {'n_epochs' in params}")
        print(f"likelihood supported: {'likelihood' in params}")
        print(f"All params: {[k for k in params.keys()]}")
except Exception as e:
    traceback.print_exc()
