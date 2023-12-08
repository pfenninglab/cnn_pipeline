
import importlib

import wandb

import dataset
import lr_schedules
import models
import train
import utils

importlib.reload(dataset)
importlib.reload(lr_schedules)
importlib.reload(models)
importlib.reload(train)
importlib.reload(utils)


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







### testing

import tensorflow as tf
res = model.loss(tf.constant([1, 1, 1, 1, 1]), tf.constant([[0.3, 0.7, -3.0], [0.3, 0.7, -2.0], [0.3, 0.7, -1.0], [0.3, 0.7, 0.0], [0.3, 0.7, 1.0]]))
print(res)
