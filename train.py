import callbacks
import dataset
import models
import lr_schedules

import wandb
from wandb.keras import WandbCallback

def train():
	wandb.init(project="mouse-sst")

	data_dir = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/"
	import os
	train_paths = [os.path.join(data_dir, fname) for fname in ["mouse_SST_neg_TRAIN.fa", "mouse_SST_pos_TRAIN.fa"]]
	val_paths = [os.path.join(data_dir, fname) for fname in ["mouse_SST_neg_VAL.fa", "mouse_SST_pos_VAL.fa"]]
	train_data = dataset.FastaTfDataset(train_paths, [0, 1])
	val_data = dataset.FastaTfDataset(val_paths, [0, 1])

	batch_size = 512
	steps_per_epoch = len(train_data.fc) // batch_size
	validation_steps = len(val_data.fc) // batch_size
	num_epochs = 24

	#lr_schedule = lr_schedules.get_clr_schedule(num_epochs, models.CONFIG)
	lr_schedule = lr_schedules.get_exp_lr_schedule(0.9, models.CONFIG)
	model = models.get_model(train_data.fc.seq_shape, train_data.fc.num_classes, lr_schedule, models.CONFIG)
	model.fit(
		train_data.ds.batch(batch_size),
		epochs=num_epochs,
		steps_per_epoch=steps_per_epoch,
		validation_data=val_data.ds.batch(batch_size),
		validation_steps=validation_steps,
		callbacks=[WandbCallback(), callbacks.LRLogger(model.optimizer)])

	print("full validation:")
	val_data = dataset.FastaTfDataset(val_paths, [0, 1], endless=False)
	model.evaluate(val_data.ds.batch(batch_size),
		callbacks=[WandbCallback()])


if __name__ == '__main__':
	train()
