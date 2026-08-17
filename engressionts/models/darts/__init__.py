import logging

logger = logging.getLogger(__name__)

try:
    from engressionts.models.darts.block_rnn_model import EnBlockRNNModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnBlockRNNModel: {e}")

try:
    from engressionts.models.darts.chronos2_model import EnChronos2Model
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnChronos2Model: {e}")

try:
    from engressionts.models.darts.dllinear_model import EnDLinearModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnDLinearModel: {e}")

try:
    from engressionts.models.darts.nbeats import EnBEATSModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnBEATSModel: {e}")

try:
    from engressionts.models.darts.nhits import EnHiTSModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnHiTSModel: {e}")

try:
    from engressionts.models.darts.nlinear_model import EnNLinearModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnNLinearModel: {e}")

try:
    from engressionts.models.darts.rnn_model import EnRNNModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnRNNModel: {e}")

try:
    from engressionts.models.darts.tcn_model import EnTCNModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTCNModel: {e}")

try:
    from engressionts.models.darts.tft_model import EnTFTModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTFTModel: {e}")

try:
    from engressionts.models.darts.tide_model import EnTiDEModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTiDEModel: {e}")

try:
    from engressionts.models.darts.tsmixer_model import EnTSMixerModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTSMixerModel: {e}")

try:
    from engressionts.models.darts.transformer import EnTransformerModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTransformerModel: {e}")

try:
    from engressionts.models.darts.enpatchtst_fm_model import EnPatchTSTFMModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnPatchTSTFMModel: {e}")

try:
    from engressionts.models.darts.entirex_model import EnTiRexModel
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTiRexModel: {e}")

try:
    from engressionts.models.darts.entimesfm2p5_model import EnTimesFM2p5Model
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import EnTimesFM2p5Model: {e}")