

CONFIG_EXPECTED_KEYS = {
	'train_data_paths': list,
	'train_labels': list,
	'val_data_paths': list,
	'val_labels': list,
	'batch_size': int,
	'num_epochs': int,
	'optimizer': str,
	'lr_init': float,
	'lr_max': float,
	'l2_reg': float,
	'dropout_rate': float,
	'num_conv_layers': int,
	'conv_filters': int,
	'conv_width': int,
	'conv_stride': int,
	'max_pool_size': int,
	'max_pool_stride': int,
	'num_dense_layers': int,
	'dense_filters': int
}
def validate_config(config_dict):
	# check required keys
	for k, t in CONFIG_EXPECTED_KEYS.items():
		assert k in config_dict

	# check types
	for k, t in CONFIG_EXPECTED_KEYS.items():
		assert isinstance(config_dict[k], t), f"key {k}, expected {t}, got {type(config_dict[k])}"

	# check lengths
	for paths, labels in [
		(config_dict['train_data_paths'], config_dict['train_labels']),
		(config_dict['val_data_paths'], config_dict['val_labels'])]:
		assert len(paths) == len(labels)