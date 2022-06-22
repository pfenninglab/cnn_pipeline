import os.path

import numpy as np
import tensorflow as tf
import wandb

import constants
import dataset

WANDB_RUN_DIR_DISABLED = '/tmp'

class LRLogger(tf.keras.callbacks.Callback):
    """Log learning rate at the end of each epoch.
    Adapted from https://stackoverflow.com/a/68911117
    """
    def __init__(self, optimizer):
        super(LRLogger, self).__init__()
        self.optimizer = optimizer

    def on_epoch_begin(self, epoch, logs):
        if epoch == 0:
            self._log()

    def on_epoch_end(self, epoch, logs):
        self._log()

    def _log(self):
        wandb.log({"lr": self.optimizer.learning_rate(self.optimizer.iterations)}, commit=False)

def get_early_stopping_callbacks(config):
    if config.get('early_stopping_callbacks') is None:
        return []
    return [
        tf.keras.callbacks.EarlyStopping(**kwargs)
        for kwargs in config.early_stopping_callbacks]

class AdditionalValidation(tf.keras.callbacks.Callback):
    """Validate on additional validation sets.
    Adapted from https://stackoverflow.com/a/62902854
    """
    def __init__(self, val_datasets, metrics=None, batch_size=constants.DEFAULT_BATCH_SIZE):
        super(AdditionalValidation, self).__init__()
        self.val_datasets = val_datasets
        self.metrics = metrics or ['acc']
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs):
        results = {}
        for idx, val_data in enumerate(self.val_datasets):
            values = self.model.evaluate(
                x=val_data.dataset[0], y=val_data.dataset[1],
                batch_size=self.batch_size, return_dict=True, verbose=0)
            for metric in self.metrics:
                results[f'val_{idx + 1}_{metric}'] = values[metric]
        # Aggregate metrics with geometric mean
        for metric in self.metrics:
            num_values = len(self.val_datasets)
            values = [results[f'val_{idx + 1}_{metric}'] for idx in range(num_values)]
            # https://en.wikipedia.org/wiki/Geometric_mean
            results[f'val_*_{metric}_gm'] = np.power(np.product(values), 1 / num_values)
        wandb.log(results)

def get_additional_validation_callback(config):
    if config.get('additional_val_data_paths') is None:
        return None

    val_datasets = [
        dataset.SequenceTfDataset(paths, targets, targets_are_classes=config.targets_are_classes,
            # Use map_targets=False in case some datasets have only positive label
            endless=False, map_targets=False)
        for paths, targets in zip(config.additional_val_data_paths, config.additional_val_targets)
    ]
    metrics = ['acc'] if config.targets_are_classes else ['mean_squared_error']
    return AdditionalValidation(val_datasets, metrics=metrics, batch_size=config.batch_size)

def get_model_checkpoint_callback():
    run_dir = wandb.run.dir
    if run_dir == WANDB_RUN_DIR_DISABLED:
        return None
    filepath = os.path.join(run_dir, 'model-latest.h5')
    return tf.keras.callbacks.ModelCheckpoint(filepath)
