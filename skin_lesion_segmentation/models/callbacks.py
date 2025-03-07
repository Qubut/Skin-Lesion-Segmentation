import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
