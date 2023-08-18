import tensorflow as tf

def get_lr_schedule(steps_per_epoch, config):
	if config.lr_schedule == 'exponential':
		return get_exp_lr_schedule(steps_per_epoch, config)
	elif config.lr_schedule == 'cyclic':
		cycle_period_epochs = config.num_epochs / config.lr_cyc_num_cycles
		return get_clr_schedule(cycle_period_epochs, steps_per_epoch, config)
	else:
		raise ValueError("Invalid learning rate schedule")

class ClrScaleFn:
	def __init__(self, scale_fn_type):
		self.scale_fn_type = scale_fn_type

	def scale_fn(self, x):
		if self.scale_fn_type == 'triangular2':
			return 1/(2.**(x-1))
		elif self.scale_fn_type == 'triangular':
			return 1
		else:
			raise NotImplementedError(f"scale_fn type not implemented: {self.scale_fn_type}")

def get_clr_schedule(cycle_period_epochs, steps_per_epoch, config):
	"""Cyclic learning rate schedule.
	See Smith, 2015: https://arxiv.org/abs/1506.01186

	Args:
		cycle_period_epochs (float): number of epochs for each full cycle
		steps_per_epoch (int): number of training steps per epoch,
			most likely len(train_dataset) // batch_size
		config (dict): hyperparameter config, containing:
			lr_init: initial learning rate, the min value of each cycle
			lr_max: maximum learning rate, the max value of each cycle
	"""
	import tensorflow_addons as tfa
	return tfa.optimizers.CyclicalLearningRate(
		initial_learning_rate=config.lr_init,
	    maximal_learning_rate=config.lr_max,
	    scale_fn=ClrScaleFn(config.lr_cyc_scale_fn).scale_fn,
	    step_size=steps_per_epoch * cycle_period_epochs / 2
	)

def get_exp_lr_schedule(steps_per_epoch, config):
	"""Exponential decay.

	Args:
		steps_per_epoch (int): number of training steps per epoch,
			most likely len(train_dataset) // batch_size
		config (dict): hyperparameter config, containing:
			lr_exp_decay_per_epoch (float in [0, 1]): fraction that the learning
				rate should exponentially decrease over 1 epoch.
				e.g. if decay_per_epoch = 0.9, then lr_{n+1} = 0.9 * lr_n,
				where lr_n and lr_{n+1} are the learning rates before and
				after epoch n, respectively.
	"""
	return tf.keras.optimizers.schedules.ExponentialDecay(
    	initial_learning_rate=config.lr_init,
    	decay_steps=steps_per_epoch,
    	decay_rate=config.lr_exp_decay_per_epoch
	)

def get_linear_lr_schedule(steps_per_epoch, num_epochs, lr_max, lr_min):
	"""Linear decay.
	Currently intended only for the "tail" part at the end of the cyclic learning rate schedule,
	not intended to be an option for the main learning rate schedule.
	This is why this function does not conform to the API for CLR and exp lr schedules.
	"""
	return tf.keras.optimizers.schedules.PolynomialDecay(
	    lr_max,
	    steps_per_epoch * num_epochs,
	    lr_min,
	    power=1)
