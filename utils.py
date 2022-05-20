import os

import wandb
import yaml

CONFIG_EXPECTED_KEYS = {
	'project': str,
	'train_data_paths': list,
	'train_targets': list,
	'val_data_paths': list,
	'val_targets': list,
	'batch_size': int,
	'num_epochs': int,
	'metric_pos_label': [int, str],
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
	'shap_pos_label': int,
	'modisco_normalization': str
}

def validate_config(config_dict):
	# check required keys
	for k, t in CONFIG_EXPECTED_KEYS.items():
		assert k in config_dict

	# check types
	for k, t in CONFIG_EXPECTED_KEYS.items():
		if not isinstance(t, list):
			t = [t]
		assert any(isinstance(config_dict[k], tt) for tt in t), (
			f"Invalid type in config! key {k}, expected {t}, got {type(config_dict[k])}")

	# check lengths
	for paths, targets in [
		(config_dict['train_data_paths'], config_dict['train_targets']),
		(config_dict['val_data_paths'], config_dict['val_targets'])]:
		assert len(paths) == len(targets)

def get_config(yaml_path):
	with open(yaml_path, "r") as f:
	    config = yaml.safe_load(f)
	config = {k: v['value'] for k, v in config.items()}
	project = config['project']
	return config, project

def get_step_size(wandb_config, train_data, val_data):
	batch_size = wandb_config.batch_size
	steps_per_epoch_train = len(train_data) // batch_size
	steps_per_epoch_val = len(val_data) // batch_size
	return batch_size, steps_per_epoch_train, steps_per_epoch_val

def validate_datasets(datasets):
	"""
	Args:
		datasets (list of dataset.SequenceTfDataset)
	"""
	# Check that classes and class mappings are all the same
	if datasets[0].targets_are_classes:
		class_to_idx_mapping = None
		idx_to_class_mapping = None
		for dataset in datasets:
			class_to_idx_mapping = class_to_idx_mapping or dataset.class_to_idx_mapping
			if dataset.class_to_idx_mapping != class_to_idx_mapping:
				raise ValueError(f"Inconsistent class_to_idx_mapping: {class_to_idx_mapping}, {dataset.class_to_idx_mapping}")
			idx_to_class_mapping = idx_to_class_mapping or dataset.idx_to_class_mapping
			if dataset.idx_to_class_mapping != idx_to_class_mapping:
				raise ValueError(f"Inconsistent idx_to_class_mapping: {idx_to_class_mapping}, {dataset.idx_to_class_mapping}")









