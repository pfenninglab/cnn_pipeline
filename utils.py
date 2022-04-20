import os

import wandb
import yaml

CONFIG_EXPECTED_KEYS = {
	'project': str,
	'train_data_paths': list,
	'train_labels': list,
	'val_data_paths': list,
	'val_labels': list,
	'batch_size': int,
	'num_epochs': int,
	'metric_pos_label': int,
	'optimizer': str,
	'lr_schedule': str,
	'lr_cyc_scale_fn': str,
	'lr_init': float,
	'lr_max': float,
	'l2_reg': float,
	'lr_exp_decay_per_epoch': float,
	'lr_cyc_num_cycles': float,
	'dropout_rate': float,
	'num_conv_layers': int,
	'conv_filters': int,
	'conv_width': int,
	'conv_stride': int,
	'max_pool_size': int,
	'max_pool_stride': int,
	'num_dense_layers': int,
	'dense_filters': int,
	'shap_num_bg': int,
	'shap_num_fg': int,
	'shap_pos_label': int
}

def validate_config(config_dict):
	# check required keys
	for k, t in CONFIG_EXPECTED_KEYS.items():
		assert k in config_dict

	# check types
	for k, t in CONFIG_EXPECTED_KEYS.items():
		assert isinstance(config_dict[k], t), (
			f"Invalid type in config! key {k}, expected {t}, got {type(config_dict[k])}")

	# check lengths
	for paths, labels in [
		(config_dict['train_data_paths'], config_dict['train_labels']),
		(config_dict['val_data_paths'], config_dict['val_labels'])]:
		assert len(paths) == len(labels)

def get_config(yaml_path):
	with open(yaml_path, "r") as f:
	    config = yaml.safe_load(f)
	config = {k: v['value'] for k, v in config.items()}
	project = config['project']
	return config, project

def get_step_size(wandb_config, train_data, val_data):
	batch_size = wandb_config.batch_size
	steps_per_epoch_train = len(train_data.fc) // batch_size
	steps_per_epoch_val = len(val_data.fc) // batch_size
	return batch_size, steps_per_epoch_train, steps_per_epoch_val
