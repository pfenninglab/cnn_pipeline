"""train.py: Train a model.

Usage:
- Single training run, from interactive session: python train.py -config config-base.yaml
- Single training run, on slurm: sbatch train.sb config-base.yaml
- Hyperparameter sweep, on slurm: see README.md
"""

import callbacks
import dataset
import models
import lr_schedules
import os
import utils

import wandb
from wandb.keras import WandbCallback

# improve connection with wandb, from: https://github.com/wandb/wandb/issues/3326#issuecomment-1065328606
os.environ["WANDB_START_METHOD"] = "thread"


def train(args):
	# Start `wandb`
	config, project = utils.get_config(args.config)

	# Configure wandb with directory and name from config
	wandb_dir = config.get('dir', os.getcwd())  # Use current directory if not specified
	wandb_name = config.get('name', None)  # Use None if not specified - wandb will generate random name
	wandb_id = utils.generate_unique_id(wandb_name) # create a unique run ID that includes the name if not none

	wandb.init(
		config=config, 
		project=project, 
		dir=wandb_dir, 
		name=wandb_name, 
		id=wandb_id, 
		mode=args.wandb_mode)
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
		endless=not wandb.config.use_exact_val_metrics,
		batch_size=wandb.config.batch_size,
		reverse_complement=wandb.config.use_reverse_complement)

	utils.validate_datasets([train_data, val_data])

	# Get training details
	steps_per_epoch_train, steps_per_epoch_val = utils.get_step_size(
		wandb.config, train_data, val_data)
	class_weight = utils.get_class_weight(wandb.config, train_data)
	
	# Get model
	lr_schedule = lr_schedules.get_lr_schedule(steps_per_epoch_train, wandb.config)
	model = models.get_model(
		train_data.seq_shape, train_data.num_classes, train_data.class_to_idx_mapping, lr_schedule, wandb.config)

	# Train
	callback_fns = callbacks.get_training_callbacks(wandb.config, model, steps_per_epoch_train)
	model.fit(
		train_data.dataset,
		epochs=wandb.config.num_epochs,
		steps_per_epoch=steps_per_epoch_train,
		validation_data=val_data.dataset,
		validation_steps=steps_per_epoch_val,
		callbacks=callback_fns,
		class_weight=class_weight)

	# CLR tail
	if wandb.config.lr_schedule == 'cyclic':
		finetune_clr_tail(steps_per_epoch_train, steps_per_epoch_val, class_weight, train_data, val_data, model, wandb.config)


def finetune_clr_tail(steps_per_epoch_train, steps_per_epoch_val, class_weight, train_data, val_data, model, config):
	"""Train with a linear LR decay at the end of a one-cycle LR schedule.
	E.g. the final linear segment illustrated at https://raw.githubusercontent.com/titu1994/keras-one-cycle/master/images/one_cycle_lr.png
	"""
	lr_schedule = lr_schedules.get_linear_lr_schedule(steps_per_epoch_train, config.clr_tail_epochs, config.lr_init, config.lr_init / 10)
	tail_config = dict(config).copy()
	if config.momentum_schedule == 'cyclic':
		tail_config['optimizer_args']['momentum'] = config.momentum_max
	optimizer = models.get_optimizer(lr_schedule, tail_config)
	metrics = models.get_metrics(train_data.num_classes, train_data.class_to_idx_mapping, config)
	model.compile(optimizer=optimizer, loss=model.loss, metrics=metrics)

	callback_fns = callbacks.get_training_callbacks(config, model, steps_per_epoch_train, disable_momentum=True)

	model.fit(
		train_data.dataset,
		epochs=config.clr_tail_epochs,
		steps_per_epoch=steps_per_epoch_train,
		validation_data=val_data.dataset,
		validation_steps=steps_per_epoch_val,
		callbacks=callback_fns,
		class_weight=class_weight)


def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, required=True)
	parser.add_argument('-wandb-mode', type=str)
	# parse_known_args() allows hyperparameters to be passed in during sweeps
	args, _ = parser.parse_known_args()
	return args


if __name__ == '__main__':
	train(get_args())
