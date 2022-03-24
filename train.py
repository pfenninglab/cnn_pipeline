import dataset
import models

def train():
	data_dir = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/"
	import os
	train_paths = [os.path.join(data_dir, fname) for fname in ["mouse_SST_neg_TRAIN.fa", "mouse_SST_pos_TRAIN.fa"]]
	val_paths = [os.path.join(data_dir, fname) for fname in ["mouse_SST_neg_VAL.fa", "mouse_SST_pos_VAL.fa"]]
	train_data = dataset.FastaTfDataset(train_paths, [0, 1])
	val_data = dataset.FastaTfDataset(val_paths, [0, 1])

	model = models.get_model(train_data.fc.seq_shape, train_data.fc.num_classes, models.CONFIG)

	batch_size = 512
	steps_per_epoch = 10 #len(train_data.fc) // batch_size
	validation_steps = len(val_data.fc) // batch_size
	model.fit(
		train_data.ds.batch(batch_size),
		epochs=2,
		steps_per_epoch=steps_per_epoch,
		validation_data=val_data.ds.batch(batch_size),
		validation_steps=validation_steps)

	print("full val:")
	val_data = dataset.FastaTfDataset(val_paths, [0, 1], endless=False)
	model.evaluate(val_data.ds.batch(batch_size))
