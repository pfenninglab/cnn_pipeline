import tensorflow as tf
import wandb


# adapted from https://stackoverflow.com/a/68911117
class LRLogger(tf.keras.callbacks.Callback):
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
