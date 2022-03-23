import dataset
import model

def train():
	data_dir = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/"
	import os
	train_paths = [os.path.join(data_dir, fname) for fname in ["mouse_SST_neg_TRAIN.fa", "mouse_SST_pos_TRAIN.fa"]]
	val_paths = [os.path.join(data_dir, fname) for fname in ["mouse_SST_neg_VAL.fa", "mouse_SST_pos_VAL.fa"]]
	train_data = dataset.FastaTfDataset(train_paths, [0, 1], endless=True)
	val_data = dataset.FastaTfDataset(val_paths, [0, 1], endless=False)

	cnn = model.get_model(train_data.fc.seq_shape, train_data.fc.num_classes, model.CONFIG)
	cnn.summary()

	cnn.fit(train_data.ds.batch(32), epochs=1)