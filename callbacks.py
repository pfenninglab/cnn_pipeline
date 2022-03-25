import tensorflow as tf
import wandb

wandb.init(project="mouse-sst")

# from https://stackoverflow.com/a/68911117
class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
      super(LRLogger, self).__init__()
      self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
      lr = self.optimizer.learning_rate(epoch)
      wandb.log({"lr": lr})