from models import load_model, MulticlassMetric, lr_schedules
import dataset
from tensorflow.keras.models import Model
import numpy as np

print('loading model')
model = load_model('wandb/run-20230120_152325-1u0587np/files/model-best.h5')
print('done')

print('loading data')
val_data = dataset.SequenceTfDataset(
        ['/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_PV/FinalModelData/mouse_PV_pos_VAL.fa'], [1],
        targets_are_classes=True,
        endless=False,
        batch_size=32,
        reverse_complement=True)

lil_sample = val_data.dataset[0][:10]
print('done')

print('eval with no dropout')
for _ in range(4):
    print(model(lil_sample))
print('done')

model_config = model.get_config()
orig_weights = model.get_weights()

# TODO do I need custom_objects?
custom_objects = {
	"MulticlassMetric": MulticlassMetric,
	"scale_fn": lr_schedules.ClrScaleFn.scale_fn
}

for layer in model_config['layers']:
	if layer.get('class_name') == 'Dropout':
		layer['inbound_nodes'][0][0][-1]['training'] = True
model = Model.from_config(model_config, custom_objects=custom_objects)
model.set_weights(orig_weights)

print('eval with reloaded model, dropout on')
res = []
for _ in range(64):
    res.append(model(lil_sample))
res = np.array(res)
print('mean')
print(np.mean(res, axis=0))
print('std')
print(np.std(res, axis=0))
print('done')

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