
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.metrics import AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives

from metrics import get_multiclass_metric, MulticlassAUC


OPTIMIZER_MAPPING = {
	'sgd': SGD,
	'adam': Adam
}
USE_CONFUSION_METRICS = False


def get_model(input_shape, num_classes, lr_schedule, config):
	model = get_model_architecture(input_shape, num_classes, config)
	optimizer = get_optimizer(lr_schedule, config)
	metrics = get_metrics()
	model.compile(loss='sparse_categorical_crossentropy',
		optimizer=optimizer,
		metrics=metrics)

	return model

def get_model_architecture(input_shape, num_classes, config):
	inputs = keras.Input(shape=input_shape)
	x = inputs
	
	for _ in range(config['num_conv_layers']):
		x = layers.Conv1D(filters=config['conv_filters'], kernel_size=config['conv_width'], activation='relu', strides=config['conv_stride'], kernel_regularizer=l2(l=config['l2_reg']))(x)
		x = layers.Dropout(rate=config['dropout_rate'])(x)

	x = layers.MaxPooling1D(pool_size=config['max_pool_size'], strides=config['max_pool_stride'])(x)
	x = layers.Flatten()(x)

	for _ in range(config['num_dense_layers']):
		x = layers.Dense(units=config['dense_filters'], activation='relu', kernel_regularizer=l2(l=config['l2_reg']))(x)
		x = layers.Dropout(rate=config['dropout_rate'])(x)

	outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2(l=config['l2_reg']))(x)

	return keras.Model(inputs=inputs, outputs=outputs)

def get_optimizer(lr_schedule, config):
	return OPTIMIZER_MAPPING[config['optimizer'].lower()](learning_rate=lr_schedule)

def get_metrics():

	metrics = [SparseCategoricalAccuracy(name='acc'),
		MulticlassAUC(name='auroc', pos_label=1, curve='ROC'),
		MulticlassAUC(name='auprc', pos_label=1, curve='PR')]
	if USE_CONFUSION_METRICS:
		metrics.extend([
			get_multiclass_metric(TruePositives, name='conf_TP', pos_label=1),
			get_multiclass_metric(TrueNegatives, name='conf_TN', pos_label=1),
			get_multiclass_metric(FalsePositives, name='conf_FP', pos_label=1),
			get_multiclass_metric(FalseNegatives, name='conf_FN', pos_label=1)])

	return metrics