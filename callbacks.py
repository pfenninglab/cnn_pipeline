import os.path

import numpy as np
import tensorflow as tf
import wandb

import constants
import dataset
import models

# wandb.run.dir when wandb mode == "disabled"
WANDB_RUN_DIR_DISABLED = '/tmp'

class LRLogger(tf.keras.callbacks.Callback):
    """Log learning rate at the end of each epoch.
    Adapted from https://stackoverflow.com/a/68911117
    """
    def __init__(self, optimizer):
        super(LRLogger, self).__init__()
        self.optimizer = optimizer
        self.optimizer_attributes = self._get_optimizer_attributes()

    def on_epoch_begin(self, epoch, logs):
        if epoch == 0:
            self._log()

    def on_epoch_end(self, epoch, logs):
        self._log()

    def _get_optimizer_attributes(self):
        """Get all numerical attributes of this optimizer."""
        attrs = [k for k, v in self.optimizer.get_config().items() if isinstance(v, float)]
        return attrs

    def _log(self):
        data = {"lr": self.optimizer.learning_rate(self.optimizer.iterations)}
        data.update({f"optim.{k}": getattr(self.optimizer, k) for k in self.optimizer_attributes})
        wandb.log(data, commit=False)

def get_early_stopping_callbacks(config):
    if config.get('early_stopping_callbacks') is None:
        return []
    return [
        tf.keras.callbacks.EarlyStopping(**kwargs)
        for kwargs in config.early_stopping_callbacks]

class AdditionalValidationLogger(tf.keras.callbacks.Callback):
    """Log additional validation set metrics after each epoch."""
    def __init__(self, additional_val):
        super(AdditionalValidationLogger, self).__init__()
        self.additional_val = additional_val

    def on_epoch_end(self, epoch, logs):
        results = self.additional_val.evaluate()
        print({k: f"{v:.4}" for k, v in results.items()})
        wandb.log(results)

def get_additional_validation_callback(config, model):
    additional_val = models.get_additional_validation(config, model)
    if additional_val is None:
        return None
    return AdditionalValidationLogger(additional_val)

def get_model_checkpoint_callback():
    """Save latest model after each epoch."""
    run_dir = wandb.run.dir
    if run_dir == WANDB_RUN_DIR_DISABLED:
        return None
    filepath = os.path.join(run_dir, 'model-latest.h5')
    return tf.keras.callbacks.ModelCheckpoint(filepath)
