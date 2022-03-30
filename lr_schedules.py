import tensorflow as tf
import tensorflow_addons as tfa


def get_clr_schedule(cycle_period_epochs, config):
	"""Cyclic learning rate schedule.
	TODO currently step_size is misconfigured
	"""
	return tfa.optimizers.CyclicalLearningRate(
		initial_learning_rate=config.lr_init,
	    maximal_learning_rate=config.lr_max,
	    scale_fn=lambda x: 1/(2.**(x-1)),
	    step_size=cycle_period_epochs / 2
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
	
