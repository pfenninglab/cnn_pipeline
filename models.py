
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

import constants
import dataset
from metrics import MulticlassMetric
import lr_schedules


OPTIMIZER_MAPPING = {
	'sgd': SGD,
	'adam': Adam
}
# TN, TP, FN, and FP, mainly for debugging
USE_CONFUSION_METRICS = False

LAYERWISE_PARAMS_CONV = ['conv_filters', 'conv_width', 'conv_stride']
LAYERWISE_PARAMS_DENSE = ['dense_filters']


def get_model(input_shape, num_classes, class_to_idx_mapping, lr_schedule, config):
	model = get_model_architecture(input_shape, num_classes, config)
	optimizer = get_optimizer(lr_schedule, config)
	metrics = get_metrics(num_classes, class_to_idx_mapping, config)

	loss = 'mean_squared_error' if num_classes is None else 'sparse_categorical_crossentropy' 
	model.compile(loss=loss,
		optimizer=optimizer,
		metrics=metrics)

	return model

def get_model_architecture(input_shape, num_classes, config):
	inputs = keras.Input(shape=input_shape)
	x = inputs

	config = _get_layerwise_params(config, 'num_conv_layers', LAYERWISE_PARAMS_CONV)
	config = _get_layerwise_params(config, 'num_dense_layers', LAYERWISE_PARAMS_DENSE)
	
	for (conv_filters, conv_width, conv_stride, _) in zip(
		config['conv_filters'], config['conv_width'], config['conv_stride'], range(config['num_conv_layers'])):
		x = layers.Conv1D(filters=conv_filters, kernel_size=conv_width, activation='relu', strides=conv_stride, kernel_regularizer=l2(l=config['l2_reg']))(x)
		x = layers.Dropout(rate=config['dropout_rate'])(x)

	x = layers.MaxPooling1D(
			pool_size=config['max_pool_size'],
			strides=config['max_pool_stride'],
			# NOTE we use padding='same' so that no input data gets discarded
			padding='same')(x)
	x = layers.Flatten()(x)

	for (dense_filters, _) in zip(config['dense_filters'], range(config['num_dense_layers'])):
		x = layers.Dense(units=dense_filters, activation='relu', kernel_regularizer=l2(l=config['l2_reg']))(x)
		x = layers.Dropout(rate=config['dropout_rate'])(x)

	if num_classes is None:
		num_output_units = 1
		activation = None
	elif isinstance(num_classes, int):
		num_output_units = num_classes
		activation = "softmax"
	else:
		raise ValueError(f"Invalid num_classes: {num_classes}")
	outputs = layers.Dense(num_output_units, activation=activation,
		kernel_regularizer=l2(l=config['l2_reg']))(x)

	return keras.Model(inputs=inputs, outputs=outputs)

def _get_layerwise_params(config, num_layer_key, params):
	for param in params:
		if not isinstance(config[param], list):
			config.update({param: [config[param]] * config[num_layer_key]}, allow_val_change=True)
		elif len(config[param]) < config[num_layer_key]:
			raise ValueError(f"Not enough layer-wise params for parameter {param}: need at least {num_layer_key} = {config[num_layer_key]}, got {config_dict[param]}")
	return config

def get_optimizer(lr_schedule, config):
	return OPTIMIZER_MAPPING[config['optimizer'].lower()](learning_rate=lr_schedule)

def get_metrics(num_classes, class_to_idx_mapping, config):
	if num_classes is None:
		# regression
		metrics = [MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()]
	else:
		# classification
		pos_label = class_to_idx_mapping[config.metric_pos_label]
		metrics = [SparseCategoricalAccuracy(name='acc'),
			MulticlassMetric('AUC', name='auroc', pos_label=pos_label, curve='ROC'),
			MulticlassMetric('AUC', name='auprc', pos_label=pos_label, curve='PR'),
			MulticlassMetric('Precision',  name='precision', pos_label=pos_label),
			MulticlassMetric('Recall', name='sensitivity', pos_label=pos_label),
			MulticlassMetric('F1Score', name='f1', pos_label=pos_label, make_dense=True, num_classes=num_classes)]
		if num_classes == 2:
			# This is a binary classification problem, so "negative" metrics apply
			neg_label = [idx for idx in class_to_idx_mapping.values() if idx != pos_label][0]
			metrics.extend([
				MulticlassMetric('Precision', name='npv', pos_label=neg_label),
				MulticlassMetric('Recall', name='specificity', pos_label=neg_label),
				MulticlassMetric('AUC', name='npvsc', pos_label=neg_label, curve='PR')])
		if USE_CONFUSION_METRICS:
			metrics.extend([
				MulticlassMetric('TruePositives', name='conf_TP', pos_label=pos_label),
				MulticlassMetric('TrueNegatives', name='conf_TN', pos_label=pos_label),
				MulticlassMetric('FalsePositives', name='conf_FP', pos_label=pos_label),
				MulticlassMetric('FalseNegatives', name='conf_FN', pos_label=pos_label)])

	return metrics

def load_model(model_path):
	"""Load a model .h5 file.

	Args:
		model_path (str): path to model .h5 file
	"""
	# These are all the custom_objects that *could* be needed to load the model,
	# even if some of them don't end up getting used. This approach might become
	# unwieldy as more features get added. If that starts to happen, consider
	# changing it so that each object knows its own `custom_objects` entries,
	# and construct this dict dynamically before load.
	custom_objects = {
		"MulticlassMetric": MulticlassMetric,
		"scale_fn": lr_schedules.ClrScaleFn.scale_fn
	}
	return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def validate(config, model):
	"""Evaluate model on main eval set, and any additional eval sets.

	import models, wandb
	wandb.init(config='config-base.yaml', mode='disabled')
	res = models.validate(wandb.config, <path to model .h5>)
	"""
	# Load model from path, if necessary
	if isinstance(model, str):
		model = load_model(model)

	# Evaluate on main validation set
	val_data = dataset.SequenceTfDataset(
		config.val_data_paths, config.val_targets,
		targets_are_classes=config.targets_are_classes, endless=False)
	res = model.evaluate(x=val_data.dataset[0], y=val_data.dataset[1],
		batch_size=config.batch_size, return_dict=True, verbose=0)

	# Evaluate on additional validation sets
	additional_val = get_additional_validation(config, model)
	if additional_val is not None:
		res.update(additional_val.evaluate())

	return res

class AdditionalValidation:
    """Validate on additional validation sets.
    Adapted from https://stackoverflow.com/a/62902854
    """
    def __init__(self, model, val_datasets, metrics=None, batch_size=constants.DEFAULT_BATCH_SIZE):
        self.model = model
        self.val_datasets = val_datasets
        self.metrics = metrics or ['acc']
        self.batch_size = batch_size

    def evaluate(self):
        results = {}
        for idx, val_data in enumerate(self.val_datasets):
            values = self.model.evaluate(
                x=val_data.dataset[0], y=val_data.dataset[1],
                batch_size=self.batch_size, return_dict=True, verbose=0)
            for metric in self.metrics:
                results[f'val_{idx + 1}_{metric}'] = values[metric]
        # Aggregate metrics with geometric mean
        for metric in self.metrics:
            num_values = len(self.val_datasets)
            values = [results[f'val_{idx + 1}_{metric}'] for idx in range(num_values)]
            # https://en.wikipedia.org/wiki/Geometric_mean
            results[f'val_*_{metric}_gm'] = np.power(np.product(values), 1 / num_values)
        return results

def get_additional_validation(config, model):
    if config.get('additional_val_data_paths') is None:
        return None

    val_datasets = [
        dataset.SequenceTfDataset(paths, targets, targets_are_classes=config.targets_are_classes,
            # Use map_targets=False in case some datasets have only positive label
            endless=False, map_targets=False)
        for paths, targets in zip(config.additional_val_data_paths, config.additional_val_targets)
    ]
    metrics = ['acc'] if config.targets_are_classes else ['mean_squared_error']
    return AdditionalValidation(model, val_datasets, metrics=metrics, batch_size=config.batch_size)
