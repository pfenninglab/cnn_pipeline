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
import utils

import wandb
from wandb.keras import WandbCallback


def train(args):
	# Start `wandb`
	config, project = utils.get_config(args.config)
	wandb.init(config=config, project=project, mode=args.wandb_mode)
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

	# Get model
	steps_per_epoch_train, steps_per_epoch_val = utils.get_step_size(
		wandb.config, train_data, val_data)
	lr_schedule = lr_schedules.get_lr_schedule(steps_per_epoch_train, wandb.config)
	model = models.get_model(
		train_data.seq_shape, train_data.num_classes, train_data.class_to_idx_mapping, lr_schedule, wandb.config)

	# Get callbacks
	callback_fns = callbacks.get_early_stopping_callbacks(wandb.config) + [
		WandbCallback(),
		callbacks.OptimizerLogger(model.optimizer),
		callbacks.get_additional_validation_callback(wandb.config, model),
		callbacks.get_model_checkpoint_callback(),
		callbacks.get_momentum_callback(steps_per_epoch_train, wandb.config)
	]
	callback_fns = [cb for cb in callback_fns if cb is not None]

	# Get class weights
	class_weight = utils.get_class_weight(wandb.config, train_data)

	# Train
	model.fit(
		train_data.dataset,
		epochs=wandb.config.num_epochs,
		steps_per_epoch=steps_per_epoch_train,
		validation_data=val_data.dataset,
		validation_steps=steps_per_epoch_val,
		callbacks=callback_fns,
		class_weight=class_weight)


	# CLR tail

	lr_schedule = lr_schedules.get_linear_lr_schedule(steps_per_epoch_train, 4, wandb.config.lr_init, wandb.config.lr_init / 10)

	tail_config = dict(wandb.config).copy()
	if wandb.config.momentum_schedule == 'cyclic':
		tail_config['optimizer_args']['momentum'] = wandb.config.momentum_max
	optimizer = models.get_optimizer(lr_schedule, tail_config)
	metrics = models.get_metrics(train_data.num_classes, train_data.class_to_idx_mapping, wandb.config)
	model.compile(optimizer=optimizer, loss=model.loss, metrics=metrics)

	callback_fns = callbacks.get_early_stopping_callbacks(wandb.config) + [
		WandbCallback(),
		callbacks.OptimizerLogger(model.optimizer),
		callbacks.get_additional_validation_callback(wandb.config, model),
		callbacks.get_model_checkpoint_callback(),
		#callbacks.get_momentum_callback(steps_per_epoch_train, wandb.config)
	]
	callback_fns = [cb for cb in callback_fns if cb is not None]

	model.fit(
		train_data.dataset,
		epochs=4,
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
