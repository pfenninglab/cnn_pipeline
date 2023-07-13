"""clr_rangetest.py: Find learning rate range for cyclic learning rate schedule.

Usage:
1. Fill out config-base.yaml with your train & validation data paths and model architecture.

2. Run the range test (takes about 30 minutes on default dataset):

sbatch -n 1 -p pfen3 --gres gpu:1 --wrap "\
source activate keras2-tf27; \
python clr_rangetest.py -config config-base.yaml"

Parameters:
-config: CNN pipeline config yaml file, e.g. config-base.yaml
-minlr: Minimum LR in the search. Default 1e-6.
-maxlr: Maximum LR in the search. Default 50.

3. The output lr_find/lr_loss.png is a plot of loss vs learning rate.
Inspect the plot and use this to interpret it:
https://github.com/titu1994/keras-one-cycle/tree/master#interpreting-the-plot

4. The bounds you should use for the cyclic LR are:
lr_max: the number you get from interpreting the plot, e.g. 10^(-1.7)
lr_init: lr_max / 20, e.g. 5^(-2.7)
"""


### from train.py

import callbacks
from clr import LRFinder
import dataset
import models
import lr_schedules
import utils

import matplotlib.pyplot as plt
import tensorflow.keras.optimizers
import wandb
from wandb.keras import WandbCallback


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

	# Plot loss vs log_10(learning_rate)
	# adapted from lr_finder.plot_schedule()
	# clipping: find the smallest and largest index where the loss is less than 4 times the min
	min_idx = min(enumerate(lr_finder.losses),
		key=lambda x: x[0] if x[1] < 4*min(lr_finder.losses) else float('inf'))[0]
	max_idx = max(enumerate(lr_finder.losses),
		key=lambda x: x[0] if x[1] < 4*min(lr_finder.losses) else float('-inf'))[0]

	plt.plot(lr_finder.lrs[min_idx:max_idx], lr_finder.losses[min_idx:max_idx])
	plt.title('Loss vs Learning Rate')
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