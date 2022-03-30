import callbacks
import dataset
import models
import lr_schedules
import utils

import wandb
from wandb.keras import WandbCallback

def train():
	wandb.init(project="mouse-sst")
	utils.validate_config(wandb.config)

	train_data = dataset.FastaTfDataset(wandb.config.train_data_paths, wandb.config.train_labels)
	val_data = dataset.FastaTfDataset(wandb.config.val_data_paths, wandb.config.val_labels)

	batch_size = wandb.config.batch_size
	steps_per_epoch = len(train_data.fc) // batch_size
	validation_steps = len(val_data.fc) // batch_size

	lr_schedule = lr_schedules.get_exp_lr_schedule(steps_per_epoch, wandb.config)
	model = models.get_model(train_data.fc.seq_shape, train_data.fc.num_classes, lr_schedule, wandb.config)

	model.fit(
		train_data.ds.batch(batch_size),
		epochs=wandb.config.num_epochs,
		steps_per_epoch=steps_per_epoch,
		validation_data=val_data.ds.batch(batch_size),
		validation_steps=validation_steps,
		callbacks=[WandbCallback(), callbacks.LRLogger(model.optimizer)])

	print("full validation:")
	val_data = dataset.FastaTfDataset(wandb.config.val_data_paths, wandb.config.val_labels, endless=False)
	model.evaluate(val_data.ds.batch(batch_size), callbacks=[WandbCallback()])


if __name__ == '__main__':
	train()
