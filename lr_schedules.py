import tensorflow as tf
import tensorflow_addons as tfa


def get_clr_schedule(cycle_period_epochs, config):
	return tfa.optimizers.CyclicalLearningRate(
		initial_learning_rate=config['lr_init'],
	    maximal_learning_rate=config['lr_max'],
	    scale_fn=lambda x: 1/(2.**(x-1)),
	    step_size=cycle_period_epochs / 2
	)

def get_exp_lr_schedule(decay_per_epoch, config):
	return tf.keras.optimizers.schedules.ExponentialDecay(
    	initial_learning_rate=config['lr_init'],
    	decay_steps=1,
    	decay_rate=decay_per_epoch
	)
	
