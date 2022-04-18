import callbacks
import dataset
import models
import lr_schedules
import utils

import wandb
# TODO once finished debugging, remove mock_wandb
use_wandb = hasattr(wandb, 'init')
if not use_wandb:
	import mock_wandb
	wandb = mock_wandb.MockWandb()
	from mock_wandb import MockWandbCallback as WandbCallback
else:
	from wandb.keras import WandbCallback


def train(args):
	# Start `wandb`
	config, project = utils.get_config(args.config)
	wandb.init(config=config, project=project)
	utils.validate_config(wandb.config)

	# Get datasets
	train_data = dataset.FastaTfDataset(wandb.config.train_data_paths, wandb.config.train_labels)
	val_data = dataset.FastaTfDataset(wandb.config.val_data_paths, wandb.config.val_labels)

	# Get model
	batch_size, steps_per_epoch_train, steps_per_epoch_val = utils.get_step_size(
		wandb.config, train_data, val_data)
	lr_schedule = lr_schedules.get_lr_schedule(steps_per_epoch_train, wandb.config)
	model = models.get_model(
		train_data.fc.seq_shape, train_data.fc.num_classes, lr_schedule, wandb.config)

	# Train
	model.fit(
		train_data.ds.batch(batch_size),
		epochs=wandb.config.num_epochs,
		steps_per_epoch=steps_per_epoch_train,
		validation_data=val_data.ds.batch(batch_size),
		validation_steps=steps_per_epoch_val,
		callbacks=[WandbCallback(), callbacks.LRLogger(model.optimizer)])

	# Validate on full validation set
	print("full validation:")
	val_data = dataset.FastaTfDataset(wandb.config.val_data_paths, wandb.config.val_labels, endless=False)
	model.evaluate(val_data.ds.batch(batch_size), callbacks=[WandbCallback()])

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, required=True)
	return parser.parse_args()


if __name__ == '__main__':
	train(get_args())
