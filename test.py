import models
import dataset
import numpy as np

print('loading model')
model = models.load_model('/home/csestili/models/mouse_sst_lgrjv1ri.h5')
print('done')

print('loading data')
val_data = dataset.SequenceTfDataset(
        ['/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_PV/FinalModelData/mouse_PV_neg_VAL.fa'], [0],
        targets_are_classes=True,
        endless=False,
        reverse_complement=True)

print(val_data.dataset[0].shape)
sample = val_data.dataset[0]
print('done')

print('eval with no dropout')
print(model.predict(sample))
print()

print('eval with dropout')
res = models.predict_with_uncertainty(model, sample)
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