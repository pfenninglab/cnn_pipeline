import os.path

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import wandb
from wandb.keras import WandbCallback

import constants
import dataset
import models
import datetime
# wandb.run.dir when wandb mode == "disabled"
WANDB_RUN_DIR_DISABLED = '/tmp'

def get_training_callbacks(config, model, steps_per_epoch, disable_momentum=False):
    callback_fns = get_early_stopping_callbacks(config) + [
        WandbCallback(),
        OptimizerLogger(model.optimizer),
        get_additional_validation_callback(config, model),
        get_model_checkpoint_callback()]
    if not disable_momentum:
        callback_fns.append(get_momentum_callback(steps_per_epoch, config))
    return [cb for cb in callback_fns if cb is not None]    

class OptimizerLogger(tf.keras.callbacks.Callback):
    """Log learning rate and optimizer values at the end of each epoch.
    Adapted from https://stackoverflow.com/a/68911117
    """
    def __init__(self, optimizer):
        super(OptimizerLogger, self).__init__()
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
    #filepath = os.path.join(run_dir, 'model-latest.h5')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(run_dir, f'model_{timestamp}_epoch_{{epoch:02d}}.h5')
    #return tf.keras.callbacks.ModelCheckpoint(filepath)
    return tf.keras.callbacks.ModelCheckpoint(filepath,
                                            save_freq='epoch',  # Save the model at the end of every epoch
		                                    save_weights_only=False,  # Set to True to save only weights, False to save the whole model
		                                    verbose=1)

def get_momentum_callback(steps_per_epoch, config):
    if config.momentum_schedule == 'cyclic':
        cycle_period_epochs = config.num_epochs / config.lr_cyc_num_cycles
        # Number of iterations in half of a cycle
        step_size = steps_per_epoch * cycle_period_epochs / 2
        return CyclicMomentum(step_size, config.momentum_base, config.momentum_max)
    else:
        return None

class CyclicMomentum(tf.keras.callbacks.Callback):
    """Adapted from /projects/pfenninggroup/machineLearningForComputationalBiology/retina/scripts/zoonomia/step9c_keras_cnn.py"""
    def __init__(self, step_size, base_m, max_m):
      self.base_m = base_m
      self.max_m = max_m
      self.step_size = step_size
      self.clr_iterations = 0.
      self.cm_iterations = 0.
      self.trn_iterations = 0.
      self.history = {}

    def cm(self):
      # NOTE this might break for num_cycles > 1.
      cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
      if cycle == 2:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.max_m
      else:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.max_m - (self.max_m-self.base_m)*np.maximum(0,(1-x))

    def on_train_begin(self, logs={}):
      logs = logs or {}
      K.set_value(self.model.optimizer.momentum, self.cm())

    def on_batch_begin(self, batch, logs=None):
      logs = logs or {}
      self.trn_iterations += 1
      self.clr_iterations += 1
      self.history.setdefault('iterations', []).append(self.trn_iterations)
      self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))
      for k, v in logs.items():
        self.history.setdefault(k, []).append(v)
      K.set_value(self.model.optimizer.momentum, self.cm())
