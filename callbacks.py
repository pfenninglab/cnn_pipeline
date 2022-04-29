import tensorflow as tf
import wandb


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
    return [
        tf.keras.callbacks.EarlyStopping(**kwargs)
        for kwargs in config.early_stopping_callbacks]
