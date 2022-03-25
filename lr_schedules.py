import tensorflow as tf
import tensorflow_addons as tfa


def get_clr_schedule(step_size, config):
	return tfa.optimizers.CyclicalLearningRate(
		initial_learning_rate=config['lr_init'],
	    maximal_learning_rate=config['lr_max'],
	    scale_fn=lambda x: 1/(2.**(x-1)),
	    step_size=step_size
	)
	
