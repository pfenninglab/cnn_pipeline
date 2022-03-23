
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CONFIG = {
	'num_conv_layers': 2,
	'conv_filters': 300,
	'conv_width': 7,
	'conv_stride': 1,
    'dropout_rate': 0.3,
    'max_pool_size': 26,
    'max_pool_stride': 26,
    'dense_filters': 300
}

def get_model(input_shape, num_classes, config):
	inputs = keras.Input(shape=input_shape)
	x = inputs
	
	for _ in range(config['num_conv_layers']):

		x = layers.Conv1D(filters=config['conv_filters'], kernel_size=config['conv_width'], activation='relu', strides=config['conv_stride'])(x)
		x = layers.Dropout(rate=config['dropout_rate'])(x)

	x = layers.MaxPooling1D(pool_size=config['max_pool_size'], strides=config['max_pool_stride'])(x)
	x = layers.Flatten()(x)
	x = layers.Dense(units=config['dense_filters'], activation='relu')(x)
	x = layers.Dropout(rate=config['dropout_rate'])(x)

	outputs = layers.Dense(num_classes, activation="softmax")(x)

	model = keras.Model(inputs=inputs, outputs=outputs)

	return model