import wandb

import callbacks
import dataset
import lr_schedules
import models
import train
import utils

# args
MODEL_UNCERTAINTY = True


# train.py

# Start `wandb`
config, project = utils.get_config('config-base.yaml')
wandb.init(config=config, project=project, mode='disabled')

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

# Get training details
steps_per_epoch_train, steps_per_epoch_val = utils.get_step_size(
	wandb.config, train_data, val_data)
class_weight = utils.get_class_weight(wandb.config, train_data)

# Get model
lr_schedule = lr_schedules.get_lr_schedule(steps_per_epoch_train, wandb.config)
model = models.get_model(
	train_data.seq_shape, train_data.num_classes, train_data.class_to_idx_mapping, lr_schedule, wandb.config,
	model_uncertainty=MODEL_UNCERTAINTY)

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


