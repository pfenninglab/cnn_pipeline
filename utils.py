import os

import wandb
import yaml

from models import LAYERWISE_PARAMS_CONV, LAYERWISE_PARAMS_DENSE

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
	'l2_reg_conv': [float, list],
	'l2_reg_dense': [float, list],
	'l2_reg_final': float,
	'lr_exp_decay_per_epoch': float,
	'lr_cyc_num_cycles': float,
	'dropout_rate_conv': [float, list],
	'dropout_rate_dense': [float, list],
	'num_conv_layers': int,
	'conv_filters': [int, list],
	'conv_width': [int, list],
	'conv_stride': [int, list],
	'max_pool_size': int,
	'max_pool_stride': int,
	'num_dense_layers': int,
	'dense_filters': [int, list],
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

	# check layer-wise parameters
	check_layerwise_params(config_dict, 'num_conv_layers', LAYERWISE_PARAMS_CONV)
	check_layerwise_params(config_dict, 'num_dense_layers', LAYERWISE_PARAMS_DENSE)

def check_layerwise_params(config_dict, layer_key, layerwise_params):
	"""For each parameter that is set per-layer, check that enough values are passed.
	E.g. if num_conv_layers == 3, then conv_filters needs to have at least 3 values.
	"""
	num_layers = config_dict[layer_key]
	for param in layerwise_params:
		if isinstance(config_dict[param], list) and len(config_dict[param]) < num_layers:
			raise ValueError(f"Not enough layer-wise params for parameter {param}: need at least {layer_key} = {num_layers}, got {config_dict[param]}")

def get_config(yaml_path):
	with open(yaml_path, "r") as f:
	    config = yaml.safe_load(f)
	config = {k: v['value'] for k, v in config.items()}
	project = config['project']
	return config, project

def get_step_size(config, train_data, val_data):
	batch_size = config.batch_size
	steps_per_epoch_train = len(train_data) // batch_size
	# if val_data.endless == True, then the validation set is a generator and we need the epoch size
	# if val_data.enless == False, then the validation set is a numpy array and the batching happens automatically
	steps_per_epoch_val = len(val_data) // batch_size if val_data.endless else None
	return steps_per_epoch_train, steps_per_epoch_val

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

def get_class_weight(config, train_data):
	if config.get('class_weight') in [None, 'none']:
		return None
	if config.get('targets_are_classes') == False:
		raise ValueError("Targets are not classes (`targets_are_classes == False`), but a class_weight scheme is provided. Please check config.")

	if config.get('class_weight') == 'reciprocal':
		# Chai's balancing method
		return {class_idx: 1 / count * len(train_data) / train_data.num_classes
			for class_idx, count in train_data.class_counts.items()}

	elif config.get('class_weight') == 'proportional':
		# Irene's balancing method
		weights = {}
		for class_idx in train_data.class_counts.keys():
			# The weight for class i is the fraction of all the data that is *not* in class i
			weights[class_idx] = sum(count/len(train_data) for idx, count in train_data.class_counts.items() if idx != class_idx)
		return weights

	else:
		raise ValueError(f"Unsupported class_weight: `{config.get('class_weight')}`")