import models
import dataset
import numpy as np
import os





for label, part in [(0, 'neg'), (1, 'pos')]:
	fname = f'mouse_SST_{part}_VAL.fa'
	in_path = os.path.join('/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/', fname)

	print('loading data')
	val_data = dataset.SequenceTfDataset(
        [in_path], [label],
        targets_are_classes=True,
        endless=False,
        reverse_complement=True)
	sample = val_data.dataset[0]

	print('loading model')
	model = models.load_model('/home/csestili/models/mouse_sst_lgrjv1ri.h5')

	print('single prediction')
	outputs = os.path.join('/home/csestili/output/bnn_test/standard/', f'{part}.npy')
	os.makedirs('/home/csestili/output/bnn_test/standard/', exist_ok=True)
	if not os.path.exists(outputs):
		res = model.predict(sample)
		np.save(outputs, res)
		print(f'saved {outputs}, shape {res.shape}')

	print('bayesian prediction')
	outputs = os.path.join('/home/csestili/output/bnn_test/bayesian/', f'{part}.npy')
	os.makedirs('/home/csestili/output/bnn_test/bayesian/', exist_ok=True)
	if not os.path.exists(outputs):
		res = models.predict_with_uncertainty(model, sample, return_trials=True)
		np.save(outputs, res['trials'])
		print(f"saved {outputs}, shape {res['trials'].shape}")






# print('loading model')
# model = models.load_model('/home/csestili/models/mouse_sst_lgrjv1ri.h5')
# print('done')


# print('loading data')
# val_data = dataset.SequenceTfDataset(
#         ['/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_PV/FinalModelData/mouse_PV_neg_VAL.fa'], [0],
#         targets_are_classes=True,
#         endless=False,
#         reverse_complement=True)

# print(val_data.dataset[0].shape)
# sample = val_data.dataset[0][:1000]
# print('done')

# print('eval with no dropout')
# print(model.predict(sample))
# print()

# print('eval with dropout')
# res = models.predict_with_uncertainty(model, sample, return_trials=True)
# print(res)
