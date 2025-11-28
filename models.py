
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

LOSS_MAPPING_REGRESSION = {
	# regression loss functions
	'mean_squared_error': 'mean_squared_error',
	'mean_absolute_error': 'mean_absolute_error',
	'mean_absolute_percentage_error': 'mean_absolute_percentage_error',
	'huber': keras.losses.Huber()
}
LOSS_MAPPING_CLASSIFICATION = {
	'sparse_categorical_crossentropy': 'sparse_categorical_crossentropy'
}
OPTIMIZER_MAPPING = {
	'sgd': SGD,
	'adam': Adam
}
# TN, TP, FN, and FP, mainly for debugging
USE_CONFUSION_METRICS = False

LAYERWISE_PARAMS_CONV = ['conv_filters', 'conv_width', 'conv_stride', 'dropout_rate_conv', 'l2_reg_conv']
LAYERWISE_PARAMS_DENSE = ['dense_filters', 'dropout_rate_dense', 'l2_reg_dense']


def get_model(input_shape, num_classes, class_to_idx_mapping, lr_schedule, config):
	if config.get('model_checkpoint') in [None, 'none']:
		model = get_model_architecture(input_shape, num_classes, config)
	else:
		model = load_model(config.model_checkpoint)
	optimizer = get_optimizer(lr_schedule, config)
	metrics = get_metrics(num_classes, class_to_idx_mapping, config)

	# choose loss function
	loss_str = config.get('loss_function')
	if num_classes is None:
		# regression
		if loss_str in [None, 'none']:
			loss_str = 'mean_squared_error'
		try:
			loss = LOSS_MAPPING_REGRESSION[loss_str]
		except KeyError:
			raise KeyError(f"Invalid loss function for regression problem: {loss_str}. Try 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', or 'huber'")
	else:
		# classification
		if loss_str in [None, 'none']:
			loss_str = 'sparse_categorical_crossentropy'
		try:
			loss = LOSS_MAPPING_CLASSIFICATION[loss_str]
		except KeyError:
			raise KeyError(f"Invalid loss function for classification problem: {loss_str}. Try 'sparse_categorical_crossentropy'")

	model.compile(loss=loss,
		optimizer=optimizer,
		metrics=metrics)

	return model

def get_model_architecture(input_shape, num_classes, config):
    """Get 1D CNN (optionally with Transformer) model architecture."""
    # initializer configs
    kernel_initializer_cfg = _get_initializer_cfg(config, 'kernel_initializer')
    bias_initializer_cfg = _get_initializer_cfg(config, 'bias_initializer')

    # 是否启用 Transformer 分支
    use_transformer = bool(config.get('use_transformer', False))

    # Inputs
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # 如果要用 transformer，则先从 one-hot 算一个原始 mask
    if use_transformer:
        mask = layers.Lambda(make_mask_from_onehot, name="input_mask")(inputs)
    else:
        mask = None

    # Convolutional stack
    for layer_num in range(config['num_conv_layers']):
        layer_config = _get_layer_config(config, layer_num, LAYERWISE_PARAMS_CONV)
        conv_filters = layer_config['conv_filters']
        conv_width = layer_config['conv_width']
        conv_stride = layer_config['conv_stride']

        x = layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_width,
            activation='relu',
            strides=conv_stride,
            kernel_regularizer=l2(l=layer_config['l2_reg_conv']),
            kernel_initializer=keras.initializers.get(kernel_initializer_cfg),
            bias_initializer=keras.initializers.get(bias_initializer_cfg),
        )(x)
        x = layers.Dropout(rate=layer_config['dropout_rate_conv'])(x)

        # 同步更新 mask 的长度（按 Conv1D 的 valid 卷积规则）
        if use_transformer:
            mask = layers.Lambda(
                lambda m, k=conv_width, s=conv_stride: conv_mask_1d(
                    m, kernel_size=k, stride=s, padding='VALID'
                ),
                name=f"mask_conv{layer_num}",
            )(mask)

    if not use_transformer:
        # ===== 纯 CNN 路径：保持和原来完全一样 =====
        x = layers.MaxPooling1D(
            pool_size=config['max_pool_size'],
            strides=config['max_pool_stride'],
            padding='same',  # 你原来的注释：不丢数据
        )(x)
        x = layers.Flatten()(x)

    else:
        # ===== CNN + RoPE Transformer 路径 =====
        pool_size = int(config['max_pool_size'])
        pool_stride = int(config['max_pool_stride'])

        # 1) MaxPooling1D（和原来一样的超参，但单独命名）
        x = layers.MaxPooling1D(
            pool_size=pool_size,
            strides=pool_stride,
            padding='same',
            name="maxpool",
        )(x)  # x: [B, T, C]

        # 2) 对 mask 做同样的 pooling，确保长度 T 完全对齐
        mask = layers.Lambda(
            lambda m, p=pool_size, s=pool_stride: pool_mask_1d(
                m, pool_size=p, stride=s, padding='SAME'
            ),
            name="mask_pooled",
        )(mask)  # mask: [B, T]

        # 3) 投影到 d_model，作为 Transformer 的 token embedding
        d_model = int(config.get('transformer_d_model', 256))
        x = layers.Dense(d_model, activation=None, name="proj_to_dmodel")(x)  # [B, T, d_model]

        # 4) Transformer 堆叠
        num_tr_layers = int(config.get('num_transformer_layers', 1))
        num_heads = int(config.get('transformer_num_heads', 4))
        key_dim = int(config.get('transformer_key_dim', d_model // num_heads))
        ff_dim = int(config.get('transformer_ff_dim', 4 * d_model))
        dropout_attn = float(config.get('transformer_dropout', 0.1))

        for layer_idx in range(num_tr_layers):
            x = transformer_block(
                x,
                mask=mask,
                num_heads=num_heads,
                key_dim=key_dim,
                ff_dim=ff_dim,
                dropout=dropout_attn,
                name_prefix=f"tr{layer_idx}",
            )

        # 5) 把 padding 位置显式清零（尽管 mask 也已经在 attention 里用过）
        x = layers.Multiply(name="apply_mask")([x, mask[..., tf.newaxis]])

        # 6) 全局 pooling：max + avg 拼接
        x_max = layers.GlobalMaxPooling1D(name="global_max_pool")(x)        # [B, d_model]
        x_avg = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)   # [B, d_model]
        x = layers.Concatenate(name="pool_concat")([x_max, x_avg])         # [B, 2*d_model]

    # Dense stack（原样保留）
    for layer_num in range(config['num_dense_layers']):
        layer_config = _get_layer_config(config, layer_num, LAYERWISE_PARAMS_DENSE)
        x = layers.Dense(
            units=layer_config['dense_filters'],
            activation='relu',
            kernel_regularizer=l2(l=layer_config['l2_reg_dense']),
            kernel_initializer=keras.initializers.get(kernel_initializer_cfg),
            bias_initializer=keras.initializers.get(bias_initializer_cfg),
        )(x)
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

    outputs = layers.Dense(
        num_output_units,
        activation=activation,
        kernel_regularizer=l2(l=config['l2_reg_final']),
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)

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
	write_csv=False, score_column=None, batch_size=constants.DEFAULT_BATCH_SIZE, bayesian=False):
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
		score_column (int, string, or None): which unit of the layer to get the score from.
			if int: select the unit with that index. for example:
				choose 0 to get the first unit (e.g. single-output regression)
				choose 1 to get the second unit (e.g. probability of positive class for binary classification)
			if 'all': return all the scores for this layer (e.g. intermediate layer activations)
			if None: behavior is based on the layer_name:
				if layer_name is None (output layer), then get the last unit in the output layer
				if layer_name is an intermediate layer, then equivalent to 'all'
		bayesian (bool): whether to run model in Bayesian inference mode.
			if False, then output fixed predictions
			if True, then output Bayesian predictions (N=64 trials) for each input
	"""
	score_column_all = 'all'

	# Check combination of inputs
	if write_csv and (score_column == score_column_all) and bayesian:
		raise IOError('Invalid argument combination. If doing Bayesian inference and writing to csv, then choose a single score_column, or pass score_column=None for default behavior.')
	if write_csv and (layer_name is not None) and bayesian:
		raise IOError('Invalid argument combination. If doing Bayesian inference and getting inner layer activations, then there are too many dimensions to write to .csv file. Omit --write_csv to save to .npy file instead.')

	# Load model from path, if necessary
	if isinstance(model, str):
		model = load_model(model)

	# Convert score column to numeric, if possible
	try:
		# "1" -> 1
		# 1 -> 1
		score_column = int(score_column)
	except:
		# None -> None
		# "all" -> "all"
		pass

	# Get output layer and score column
	if layer_name is None:
		out_layer = model.layers[-1]
		# Apply score_column default
		if score_column is None:
			# The last unit in the output layer
			score_column = out_layer.output_shape[1] - 1
	else:
		out_layer = model.get_layer(layer_name)
		# Apply score_column default
		if score_column is None:
			# The entire layer
			score_column = 'all'
	out_shape = out_layer.output_shape
	if write_csv and len(out_shape) != 2:
		raise ValueError(f"Wrong layer shape for write_csv. Required shape is rank 2, i.e. [None, N], got layer {layer_name} with shape {out_shape}")
	
	# Check score column
	if isinstance(score_column, int):
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
		source_files, [0 for _ in source_files], targets_are_classes=True, endless=False, reverse_complement=use_reverse_complement)

	# Generate predictions
	print("Predicting...")
	if bayesian:
		predictions = predict_with_uncertainty(model, data.dataset[0], batch_size=batch_size, num_trials=64, return_trials=True)
		# [num_examples, num_bayesian_trials, num_classes]
		predictions = predictions['trials']
	else:
		# [num_examplesl, num_classes]
		predictions = model.predict(data.dataset[0], batch_size=batch_size, verbose=1)

	# Write to file
	if out_file is not None:
		print("Saving...")
		if write_csv:
			if score_column == score_column_all:
				# Write entire activation as row
				lines = predictions
			else:
				# Extract single value
				if bayesian:
					# [num_examples, num_bayesian_trials]
					lines = predictions[:, :, score_column]
				else:
					# [num_examples,]
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
        metrics = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
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
			Optionally "trials" which are all the raw outputs of the model, shape [num_examples, num_trials, num_classes]
	"""
	model = enable_dropout(model)
	trials = np.array([model.predict(inputs, batch_size=batch_size) for _ in tqdm(range(num_trials))])
	res = {
		"mean": np.mean(trials, axis=0),
		"std": np.std(trials, axis=0),
		"skew": scipy.stats.skew(trials, axis=0),
		"kurtosis": scipy.stats.kurtosis(trials, axis=0)
	}
	if return_trials:
		# Swap axes so that dimensions are [num_examples, num_trials, num_classes]
		res['trials'] = np.swapaxes(trials, 0, 1)
	return res

########################################
# Mask helpers
########################################

def make_mask_from_onehot(x):
    """
    x: [B, L, C]，one-hot 序列，pad 部分全 0
    返回: [B, L]，真实碱基=1，padding=0
    """
    mask = tf.reduce_sum(tf.abs(x), axis=-1) > 0  # bool
    return tf.cast(mask, tf.float32)


def conv_mask_1d(mask, kernel_size, stride=1, padding='VALID'):
    """
    按照 Conv1D 的 kernel_size / stride / padding 规则，下采样 mask。
    思路：
      - 对 mask 做一次 1D conv（kernel 全 1）
      - 如果一个 output 位置看到的 window 里所有 mask==1，则该位置 mask_out=1，否则=0
    mask: [B, L_in]
    返回: [B, L_out]
    """
    mask = tf.cast(mask, tf.float32)
    # [B, L, 1]
    m = mask[..., tf.newaxis]
    # 卷积核 [kernel_size, in_channels=1, out_channels=1]
    kernel = tf.ones((kernel_size, 1, 1), dtype=tf.float32)
    conv = tf.nn.conv1d(m, kernel, stride=stride, padding=padding)  # [B, L_out, 1]
    conv = tf.squeeze(conv, axis=-1)  # [B, L_out]
    # 只有当这个 window 全是 1 时，sum == kernel_size
    full = tf.equal(conv, float(kernel_size))
    return tf.cast(full, tf.float32)


def pool_mask_1d(mask, pool_size, stride, padding='SAME'):
    """
    按照 MaxPooling1D 的 pool_size / stride / padding 规则，下采样 mask。
    逻辑：window 里只要有一个有效位置，就认为 pooled 位置有效。
    mask: [B, L_in]
    返回: [B, L_out]
    """
    mask = tf.cast(mask, tf.float32)
    m = mask[..., tf.newaxis]  # [B, L, 1]
    pooled = tf.nn.max_pool1d(
        m,
        ksize=pool_size,
        strides=stride,
        padding=padding,
        data_format="NWC",
    )  # [B, L_out, 1]
    pooled = tf.squeeze(pooled, axis=-1)  # [B, L_out]
    return pooled


########################################
# RoPE Multi-Head Self-Attention
########################################

class RotaryMultiHeadSelfAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, rope_base=10000.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.rope_base = rope_base
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "rope_base": self.rope_base,
            "dropout": self.dropout,
        })
        return config
	
    def build(self, input_shape):
        d_model = input_shape[-1]
        assert d_model == self.num_heads * self.key_dim, \
            f"d_model ({d_model}) must = num_heads ({self.num_heads}) * key_dim ({self.key_dim})"

        self.qkv_dense = layers.Dense(3 * d_model, use_bias=False)
        self.out_dense = layers.Dense(d_model, use_bias=False)
        self.attn_dropout = layers.Dropout(self.dropout)
        super().build(input_shape)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([-x2, x1], axis=-1)

    def _compute_rope_angles(self, seq_len, dim):
        half_dim = dim // 2
        inv_freq = 1.0 / (self.rope_base ** (tf.range(0, half_dim, 1.0) / half_dim))
        positions = tf.cast(tf.range(seq_len), tf.float32)  # [L]
        freqs = tf.einsum('i,j->ij', positions, inv_freq)   # [L, half_dim]
        emb = tf.concat([freqs, freqs], axis=-1)            # [L, dim]
        cos = tf.cos(emb)[tf.newaxis, tf.newaxis, ...]      # [1,1,L,dim]
        sin = tf.sin(emb)[tf.newaxis, tf.newaxis, ...]
        return cos, sin

    def _apply_rope(self, x, cos, sin):
        # x: [B, H, L, D]
        return (x * cos) + (self._rotate_half(x) * sin)

    def call(self, x, mask=None, training=None):
        """
        x: [B, L, d_model]
        mask: [B, L]，1 = 有效，0 = padding
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        d_model = x.shape[-1]

        # qkv projection
        qkv = self.qkv_dense(x)  # [B, L, 3*d_model]
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.key_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]          # each [B, H, L, D]

        # apply RoPE to q, k
        cos, sin = self._compute_rope_angles(seq_len, self.key_dim)  # [1,1,L,D]
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # scaled dot-product attention
        scale = tf.math.rsqrt(tf.cast(self.key_dim, tf.float32))
        attn_scores = tf.einsum('bhqd,bhkd->bhqk', q, k) * scale  # [B,H,L,L]

        if mask is not None:
            # mask 作用在 key 维度：不允许 attend 到 padding 位置
            m = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)  # [B,1,1,L]
            attn_scores += (1.0 - m) * -1e9

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        context = tf.einsum('bhqk,bhkd->bhqd', attn_weights, v)  # [B,H,L,D]
        context = tf.transpose(context, [0, 2, 1, 3])            # [B,L,H,D]
        context = tf.reshape(context, [batch_size, seq_len, d_model])  # [B,L,d_model]

        out = self.out_dense(context)
        return out


def transformer_block(x, mask, num_heads=4, key_dim=64, ff_dim=256,
                      dropout=0.1, name_prefix="tr"):
    d_model = x.shape[-1]

    # self-attention with RoPE
    attn_out = RotaryMultiHeadSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f"{name_prefix}_rope_mha"
    )(x, mask=mask)
    x = layers.Add(name=f"{name_prefix}_attn_add")([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_attn_ln")(x)

    # position-wise FFN
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(d_model),
    ], name=f"{name_prefix}_ffn")
    ffn_out = ffn(x)
    ffn_out = layers.Dropout(dropout, name=f"{name_prefix}_ffn_dropout")(ffn_out)
    x = layers.Add(name=f"{name_prefix}_ffn_add")([x, ffn_out])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ffn_ln")(x)
    return x


def build_cnn_rope_model(
    seq_len=None,       # None = 支持变长；如果你现在还是 500 就填 500 也行
    num_channels=4,
    num_filters=500,    # 和你之前 CNN 一样
    d_model=256,        # Transformer 的通道数
    num_heads=4,
    num_transformer_layers=1,
    ff_dim=1024,
    dense_units=300,
    output_activation="linear",  # 回归可以用 "linear"
):
    # 1) 输入 + mask
    inputs = keras.Input(shape=(seq_len, num_channels), name="seq")  # (L,4)
    mask = layers.Lambda(make_mask_from_onehot, name="seq_mask")(inputs)  # [B,L]

    # 2) CNN 前端（基本保留你原来的结构）
    x = layers.Conv1D(num_filters, 11, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.Dropout(0.1, name="conv1_dropout")(x)

    x = layers.Conv1D(num_filters, 11, padding="same", activation="relu", name="conv2")(x)
    x = layers.Dropout(0.1, name="conv2_dropout")(x)

    # 3) MaxPooling + mask pooling（适当减少 pool_size，比如 4）
    pool_size = 4
    stride = 4
    x = layers.MaxPooling1D(pool_size=pool_size, strides=stride,
                            padding="same", name="maxpool")(x)  # [B,T,500]

    mask_pooled = layers.Lambda(
        lambda m: pool_mask(m, pool_size, stride),
        name="mask_pooled"
    )(mask)  # [B,T]

    # 4) 投影到 d_model，作为 Transformer 的 token 表示
    x = layers.Dense(d_model, activation=None, name="proj_to_dmodel")(x)

    # 5) RoPE Transformer 层（可以 1–2 层）
    for i in range(num_transformer_layers):
        x = transformer_block(
            x,
            mask=mask_pooled,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout=0.1,
            name_prefix=f"tr{i}"
        )

    # 选做：把 pad 位置显式置零（即使通常也很小）
    x = layers.Multiply(name="apply_mask")([x, mask_pooled[..., tf.newaxis]])

    # 6) Global pooling 替代 Flatten，length-agnostic
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # 7) Dense 头，尽量接近你原来的
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dense1_dropout")(x)

    x = layers.Dense(dense_units, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.5, name="dense2_dropout")(x)

    outputs = layers.Dense(1, activation=output_activation, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_rope_transformer")
    return model
