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

	config = models.CONFIG
	batch_size = config['batch_size']
	steps_per_epoch = len(train_data.fc) // batch_size
	validation_steps = len(val_data.fc) // batch_size

	lr_schedule = lr_schedules.get_exp_lr_schedule(steps_per_epoch, 0.94, config)
	model = models.get_model(train_data.fc.seq_shape, train_data.fc.num_classes, lr_schedule, config)
	model.fit(
		train_data.ds.batch(batch_size),
		epochs=config['num_epochs'],
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
