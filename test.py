from models import load_model
import dataset
from tensorflow.keras.models import Model
import numpy as np
import scipy.stats

print('loading model')
model = load_model('/home/csestili/models/mouse_sst_lgrjv1ri.h5')
print('done')

print('loading data')
val_data = dataset.SequenceTfDataset(
        ['/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_PV/FinalModelData/mouse_PV_pos_VAL.fa'], [1],
        targets_are_classes=True,
        endless=False,
        batch_size=32,
        reverse_complement=True)

sample = val_data.dataset[0][:10]
print('done')

print('eval with no dropout')
print(model(sample))
print('done')


def enable_dropout(model):
	"""Turn on all Dropout layers during inference.

	Args:
		model (keras.models.Model)
	"""
	model_config = model.get_config()
	orig_weights = model.get_weights()
	for layer in model_config['layers']:
		if layer.get('class_name') == 'Dropout':
			layer['inbound_nodes'][0][0][-1]['training'] = True
	# If this line fails in the future, we might need to add the custom_objects argument as in load_model()
	model = Model.from_config(model_config)
	model.set_weights(orig_weights)
	return model

model = enable_dropout(model)

def predict_with_uncertainty(model, inputs, num_trials=128):
	model = enable_dropout(model)
	trials = np.array([model(inputs) for _ in range(num_trials)])
	res = {
		"trials": trials,
		"mean": np.mean(trials, axis=0),
		"std": np.std(trials, axis=0),
		"skew": scipy.stats.skew(trials, axis=0),
		"kurtosis": scipy.stats.kurtosis(trials, axis=0)
	}
	return res

res = predict_with_uncertainty(model, sample)
print(res)

'''
no dropout
[[0.7808571  0.21914288]
 [0.6140988  0.3859012 ]]

dropout mean
[[0.7485059 , 0.25149402],
 [0.59578687, 0.40421322]]

dropout std
[[0.10021471, 0.10021472],
 [0.12408201, 0.12408199]]


'''