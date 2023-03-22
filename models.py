
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm

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

LAYERWISE_PARAMS_CONV = ['conv_filters', 'conv_width', 'conv_stride', 'dropout_rate_conv', 'l2_reg_conv']
LAYERWISE_PARAMS_DENSE = ['dense_filters', 'dropout_rate_dense', 'l2_reg_dense']


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
	"""Get 1-dimensional CNN model architecture.
	Properties:
		- Inputs are 1-hot encoded sequences of shape [sequence_len, encoding_dim]
			- encoding_dim = 4 for DNA sequences (A, C, G, T)
		- Outputs are either:
			- float tensor of shape [num_classes], non-negative and summing to 1, if num_classes >= 2 (classification)
			- float tensor of shape [1], taking values in (-inf, inf), if num_classes is None (regression)
	"""
	# Get config dicts for kernel and bias initializers
	kernel_initializer_cfg = _get_initializer_cfg(config, 'kernel_initializer')
	bias_initializer_cfg = _get_initializer_cfg(config, 'bias_initializer')

	# Inputs
	inputs = keras.Input(shape=input_shape)
	x = inputs

	# Convolutional stack
	for layer_num in range(config['num_conv_layers']):
		layer_config = _get_layer_config(config, layer_num, LAYERWISE_PARAMS_CONV)
		x = layers.Conv1D(
				filters=layer_config['conv_filters'],
				kernel_size=layer_config['conv_width'],
				activation='relu',
				strides=layer_config['conv_stride'],
				kernel_regularizer=l2(l=layer_config['l2_reg_conv']),
				kernel_initializer=keras.initializers.get(kernel_initializer_cfg),
				bias_initializer=keras.initializers.get(bias_initializer_cfg))(x)
		x = layers.Dropout(rate=layer_config['dropout_rate_conv'])(x)

	# Max-pooling layer
	x = layers.MaxPooling1D(
			pool_size=config['max_pool_size'],
			strides=config['max_pool_stride'],
			# NOTE we use padding='same' so that no input data gets discarded
			padding='same')(x)
	x = layers.Flatten()(x)

	# Dense stack
	for layer_num in range(config['num_dense_layers']):
		layer_config = _get_layer_config(config, layer_num, LAYERWISE_PARAMS_DENSE)
		x = layers.Dense(
				units=layer_config['dense_filters'],
				activation='relu',
				kernel_regularizer=l2(l=layer_config['l2_reg_dense']),
				kernel_initializer=keras.initializers.get(kernel_initializer_cfg),
				bias_initializer=keras.initializers.get(bias_initializer_cfg))(x)
		x = layers.Dropout(rate=layer_config['dropout_rate_dense'])(x)

	# Final (output) layer
	if num_classes is None:
		num_output_units = 1
		activation = None
	elif isinstance(num_classes, int):
		num_output_units = num_classes
		activation = "softmax"
	else:
		raise ValueError(f"Invalid num_classes: {num_classes}")
	outputs = layers.Dense(num_output_units, activation=activation,
		kernel_regularizer=l2(l=config['l2_reg_final']))(x)

	return keras.Model(inputs=inputs, outputs=outputs)

# TODO move Bayesian layers to a new layers module

# From https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# The prior is not trainable; we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian.
# The learnable parameters for this distribution are the means, variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def get_dense_layer(bayesian=False, train_size=None, **kwargs):
	"""Return either keras.layers.Dense or tfp.layers.DenseVariational layer instance.
	Args:
		bayesian (bool):
			if False, then return keras Dense layer.
			if True, then return tfp DenseVariational layer with Gaussian prior and posterior.
		train_size (int): number of examples in the training set, needed for bayesian == True.

	Returns: keras.layers.Layer
	"""
	if bayesian:
		return tfp.layers.DenseVariational(
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=(1 / train_size),
            **kwargs
        )


def _get_layer_config(config, layer_num, keys):
	"""Get the config values that apply at this layer.
	If a config value is set as a list, then this returns the element from that list at this layer.
	If a config value is set as a single value, then this returns that value.

	E.g. if config contains {
		'conv_filters': [300, 400, 500],
		'conv_width': 7
	}
	then _get_layer_config(config, 1) contains {
		'conv_filters': 400, # because the 1-th element of [300, 400, 500] is 400
		'conv_width': 7      # because 7 is a constant config value
	}

	Args:
	    config (wandb.config)
	    layer_num (int)
	    keys (list of str): only get config for these keys
	"""
	layer_config = {}
	for k in keys:
		v = config[k]
		if isinstance(v, list):
			if layer_num >= len(v):
				raise ValueError(f"Not enough layer-wise params for parameter {k}, got {v}. Please check that this parameter has enough values for the number of layers, or use a constant value.")
			layer_config[k] = v[layer_num]
		else:
			layer_config[k] = v
	return layer_config

def _get_initializer_cfg(config, key):
	"""Create config dict for tf.keras.initializers.get()"""
	# Default
	init_cfg = {'class_name': 'glorot_uniform' if key == 'kernel_initializer' else 'zeros',
				'config': {}}

	data = config.get(key)
	if not data:
		return init_cfg

	identifier = data.get('identifier')
	if identifier:
		init_cfg['class_name'] = identifier
	args = data.get('args')
	if args:
		init_cfg['config'] = args
	return init_cfg

def get_optimizer(lr_schedule, config):
	args = config.get('optimizer_args') or {}
	if lr_schedule is not None:
		args['learning_rate'] = lr_schedule
	optimizer = OPTIMIZER_MAPPING[config['optimizer'].lower()](**args)
	return optimizer

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
		targets_are_classes=config.targets_are_classes, endless=False,
		reverse_complement=config.use_reverse_complement)
	res = model.evaluate(x=val_data.dataset[0], y=val_data.dataset[1],
		batch_size=config.batch_size, return_dict=True, verbose=0)

	# Evaluate on additional validation sets
	additional_val = get_additional_validation(config, model)
	if additional_val is not None:
		res.update(additional_val.evaluate())

	return res

def get_activations(model, in_files, in_genomes=None, out_file=None, layer_name=None, use_reverse_complement=True,
	write_csv=False, score_column=None, batch_size=constants.DEFAULT_BATCH_SIZE):
	"""Use the model to predict on all sequences, and save the activations.

	Args:
		model (keras model or str)
		in_files (str or list of str): paths to input .fa, .bed, or .narrowPeak files.
		in_genomes (str or list of str): paths to corresponding genome .fa files,
			if in_files are .bed or .narrowPeak. You must pass the same number of
			in_genomes as in_files.
		out_file (str): path to output file, .npy or .csv
		layer_name (str): layer of model to get activations from. Default is the output layer.
		use_reverse_complement (bool): if True, then evaluate on reverse complement sequences as well.
			The order of the output predictions is then:
			pred(example_1), pred(revcomp(example_1)), ..., pred(example_n), pred(revcomp(example_n))
		write_csv (bool): whether to write activations to csv
			if False, then activations will be saved as a numpy array, dimension [num_examples, dim_1, ..., dim_n]
			if True, then activations will be saved as rows in a csv. This can only be used with
			a layer whose output shape is rank 2, i.e. a layer with output shape (None, N).
		score_column (int): index of rank-1 activations to write to csv rows.
			For example, if you want the predicted probabilities out of a binary classifier, use
			layer_name=None (get activations of output layer) and
			score_column=1 (get score from output unit for class 1).
			if score_column is None, then all units of activation will be written as a row.
	"""
	# Load model from path, if necessary
	if isinstance(model, str):
		model = load_model(model)

	# Check layer shape
	if layer_name is None:
		out_layer = model.layers[-1]
	else:
		out_layer = model.get_layer(layer_name)
	out_shape = out_layer.output_shape
	if write_csv and len(out_shape) != 2:
		raise ValueError(f"Wrong layer shape for write_csv. Required shape is rank 2, i.e. [None, N], got layer {layer_name} with shape {out_shape}")
	if (score_column is not None):
		if not isinstance(score_column, int):
			raise ValueError(f"Invalid type for score_column, expected int, got {type(score_column)}")
		if score_column >= out_shape[1]:
			raise ValueError(f"Invalid score_column, got {score_column} but layer shape is {out_shape}")

	# Get model to evaluate
	if layer_name is not None:
		model = keras.Model(inputs=model.inputs, outputs=out_layer.output)

	# Get dataset
	if isinstance(in_files, str):
		in_files = [in_files]
	if isinstance(in_genomes, str):
		in_genomes = [in_genomes]
	if in_genomes is not None:
		source_files = [
			{"genome": in_genome, "intervals": in_file}
			for (in_file, in_genome) in zip(in_files, in_genomes)]
	else:
		# in_files are .fa files
		source_files = in_files
	# Only the input sequences will be used, target is fake
	data = dataset.SequenceTfDataset(
		source_files, [0], targets_are_classes=True, endless=False, reverse_complement=use_reverse_complement)

	# Generate predictions
	print("Predicting...")
	predictions = model.predict(data.dataset[0], batch_size=batch_size, verbose=1)

	# Write to file
	if out_file is not None:
		print("Saving...")
		if write_csv:
			if score_column is None:
				# Write entire activation as row
				lines = predictions
			else:
				# Extract single value
				lines = predictions[:, score_column]
			np.savetxt(out_file, lines, delimiter='\t', fmt='%.8e')
		else:
			np.save(out_file, predictions)

	return predictions

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
        for idx, val_data in tqdm(enumerate(self.val_datasets), total=len(self.val_datasets)):
            values = self.model.evaluate(
                x=val_data.dataset[0], y=val_data.dataset[1],
                batch_size=self.batch_size, return_dict=True, verbose=0)
            for metric in self.metrics:
                if metric in values:
                    results[f'val_{idx + 1}_{metric}'] = values[metric]
        # Aggregate metrics with geometric mean
        for metric in self.metrics:
            num_values = len(self.val_datasets)
            try:
	            values = [results[f'val_{idx + 1}_{metric}'] for idx in range(num_values)]
	            # https://en.wikipedia.org/wiki/Geometric_mean
	            results[f'val_*_{metric}_gm'] = np.power(np.product(values), 1 / num_values)
            except KeyError as e:
            	# this metric was not calculated, skip it
	            pass
        return results

def get_additional_validation(config, model):
    """Get AdditionalValidation with datasets and metrics based on config."""
    if config.get('additional_val_data_paths') is None:
        return None

    val_datasets = [
        dataset.SequenceTfDataset(paths, targets, targets_are_classes=config.targets_are_classes,
            # Use map_targets=False in case some datasets have only positive label
            endless=False, map_targets=False, reverse_complement=config.use_reverse_complement)
        for paths, targets in zip(config.additional_val_data_paths, config.additional_val_targets)
    ]
    if config.targets_are_classes:
    	metrics = ['acc', 'auroc', 'auprc', 'precision', 'sensitivity', 'f1', 'npv', 'specificity', 'npvsc']
    else:
    	metrics = ['mean_squared_error']
    return AdditionalValidation(model, val_datasets, metrics=metrics, batch_size=config.batch_size)


def enable_dropout(model):
	"""Turn on all Dropout layers during inference.

	Args:
		model (keras.models.Model)

	Returns: keras.models.Model
	"""
	model_config = model.get_config()
	orig_weights = model.get_weights()
	for layer in model_config['layers']:
		if layer.get('class_name') == 'Dropout':
			layer['inbound_nodes'][0][0][-1]['training'] = True
	# If this line fails in the future, we might need to add the custom_objects argument as in load_model()
	model = keras.Model.from_config(model_config)
	model.set_weights(orig_weights)
	return model

def predict_with_uncertainty(model, inputs, batch_size=constants.DEFAULT_BATCH_SIZE, num_trials=64, return_trials=False):
	"""Predict multiple times with Dropout enabled, and report aggregate results.
	This is a Dropout-based approximation to using a Bayesian neural network.

	Args:
		model (keras.models.Model)
		inputs (np.ndarray): shape [num_examples, sequence_len, 4]
		batch_size (int): batch size for prediction
		num_trials (int): number of times to run the model on each input
		return_trials (bool):
			if True, then return the raw outputs of the model for each trial,
				in addition to aggregate results.
			if False, then return the aggregate results only.

	Returns:
		res (dict): Aggregated outputs of the model. Keys are
			"mean", "std", "skew", "kurtosis", all have shape [num_examples, num_classes]
			Optionally "trials" which are all the raw outputs of the model, shape [num_trials, num_examples, num_classes]
	"""
	model = enable_dropout(model)
	trials = np.array([model.predict(inputs, batch_size=batch_size) for _ in range(num_trials)])
	res = {
		"mean": np.mean(trials, axis=0),
		"std": np.std(trials, axis=0),
		"skew": scipy.stats.skew(trials, axis=0),
		"kurtosis": scipy.stats.kurtosis(trials, axis=0)
	}
	if return_trials:
		res['trials'] = trials
	return res
