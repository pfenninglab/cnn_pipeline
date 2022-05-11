
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.metrics import AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives

from metrics import MulticlassMetric
import metrics
import lr_schedules


OPTIMIZER_MAPPING = {
	'sgd': SGD,
	'adam': Adam
}
USE_CONFUSION_METRICS = False


def get_model(input_shape, num_classes, class_to_idx_mapping, lr_schedule, config):
	model = get_model_architecture(input_shape, num_classes, config)
	optimizer = get_optimizer(lr_schedule, config)
	metrics = get_metrics(class_to_idx_mapping, config)

	loss = 'mean_squared_error' if num_classes is None else 'sparse_categorical_crossentropy' 
	model.compile(loss=loss,
		optimizer=optimizer,
		metrics=metrics)

	return model

def get_model_architecture(input_shape, num_classes, config):
	inputs = keras.Input(shape=input_shape)
	x = inputs
	
	for _ in range(config['num_conv_layers']):
		x = layers.Conv1D(filters=config['conv_filters'], kernel_size=config['conv_width'], activation='relu', strides=config['conv_stride'], kernel_regularizer=l2(l=config['l2_reg']))(x)
		x = layers.Dropout(rate=config['dropout_rate'])(x)

	x = layers.MaxPooling1D(
			pool_size=config['max_pool_size'],
			strides=config['max_pool_stride'],
			# NOTE we use padding='same' so that no input data gets discarded
			padding='same')(x)
	x = layers.Flatten()(x)

	for _ in range(config['num_dense_layers']):
		x = layers.Dense(units=config['dense_filters'], activation='relu', kernel_regularizer=l2(l=config['l2_reg']))(x)
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

def get_optimizer(lr_schedule, config):
	return OPTIMIZER_MAPPING[config['optimizer'].lower()](learning_rate=lr_schedule)

def get_metrics(class_to_idx_mapping, config):
	pos_label = class_to_idx_mapping[config.metric_pos_label]

	metrics = [SparseCategoricalAccuracy(name='acc'),
		MulticlassMetric('AUC', name='auroc', pos_label=pos_label, curve='ROC'),
		MulticlassMetric('AUC', name='auprc', pos_label=pos_label, curve='PR')]
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
		"MulticlassMetric": metrics.MulticlassMetric,
		"scale_fn": lr_schedules.ClrScaleFn.scale_fn
	}
	return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
