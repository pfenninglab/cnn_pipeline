"""clr_rangetest.py: Find learning rate range for cyclic learning rate schedule.
"""


### from train.py

import callbacks
import dataset
import models
import lr_schedules
import utils

import wandb
from wandb.keras import WandbCallback
import tensorflow.keras.optimizers


def range_test(args):
	# Start `wandb`
	config, project = utils.get_config(args.config)
	wandb.init(config=config, project=project, mode='disabled')
	utils.validate_config(wandb.config)

	# Get datasets
	train_data = dataset.SequenceTfDataset(
		wandb.config.train_data_paths, wandb.config.train_targets,
		targets_are_classes=wandb.config.targets_are_classes, endless=True,
		batch_size=wandb.config.batch_size,
		reverse_complement=wandb.config.use_reverse_complement)
	val_data = dataset.SequenceTfDataset(
		wandb.config.val_data_paths, wandb.config.val_targets,
		targets_are_classes=wandb.config.targets_are_classes,
		endless=False,
		batch_size=wandb.config.batch_size,
		reverse_complement=wandb.config.use_reverse_complement)

	utils.validate_datasets([train_data, val_data])

	# Get model
	steps_per_epoch_train, steps_per_epoch_val = utils.get_step_size(
		wandb.config, train_data, val_data)
	lr_schedule = lr_schedules.get_lr_schedule(steps_per_epoch_train, wandb.config)
	model = models.get_model(
		train_data.seq_shape, train_data.num_classes, train_data.class_to_idx_mapping, lr_schedule, wandb.config)

	# LR range test requires SGD optimizer
	model.compile(optimizer=tensorflow.keras.optimizers.SGD(), loss=model.loss, metrics=model.metrics)




	### from lr range finder

	from clr import LRFinder

	lr_finder = LRFinder(len(train_data), wandb.config.batch_size,
	                     minimum_lr=args.minlr, maximum_lr=args.maxlr,
	                     lr_scale='exp',
	                     save_dir='lr_find/',
	                     validation_data=val_data.dataset,
	                     loss_smoothing_beta=0.0,
	                     validation_sample_rate=8)

	model.fit(train_data.dataset,
	          epochs=1,
	          steps_per_epoch=steps_per_epoch_train,
	          callbacks=[lr_finder])

	# adapted from lr_finder.plot_schedule()

	import matplotlib.pyplot as plt

	# clipping: find the smallest and largest index where the loss is less than 4 times the min
	min_idx = min(enumerate(lr_finder.losses),
		key=lambda x: x[0] if x[1] < 4*min(lr_finder.losses) else float('inf'))[0]
	max_idx = max(enumerate(lr_finder.losses),
		key=lambda x: x[0] if x[1] < 4*min(lr_finder.losses) else float('-inf'))[0]

	plt.plot(lr_finder.lrs[min_idx:max_idx], lr_finder.losses[min_idx:max_idx])
	plt.title('Learning rate vs Loss')
	plt.xlabel('log_10(learning rate)')
	plt.ylabel('loss')
	plt.savefig('lr_find/lr_loss.png', dpi=200)

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, required=True)
	parser.add_argument('-minlr', type=float, default=1e-6, help='Minimum learning rate to test, default 1e-6.')
	parser.add_argument('-maxlr', type=float, default=50., help='Maximum learning rate to test, default 50.')
	# parse_known_args() allows hyperparameters to be passed in during sweeps
	args, _ = parser.parse_known_args()
	return args


if __name__ == '__main__':
	range_test(get_args())