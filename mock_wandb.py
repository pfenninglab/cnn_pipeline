"""mock_wandb.py: mock `wandb` object to be used if not in current env
NOTE this is intended only for debugging, do not use in real training runs!
"""

class MockWandb:
	def __init__(self):
		self.config = MockWandbConfig()

	def init(self, **kwargs):
		pass

class MockWandbConfig:
	def __init__(self):
		self.train_data_paths = [
			"/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_neg_TRAIN.fa",
			"/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_TRAIN.fa"
		]
		self.train_labels = [0, 1]
		self.val_data_paths = [
			"/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_neg_VAL.fa",
			"/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.fa"
		]
		self.val_labels = [0, 1]
		self.batch_size = 512
		self.num_epochs = 0
		self.optimizer = 'adam'
		self.lr_schedule = 'exponential'
		self.lr_cyc_scale_fn = 'triangular2'
		self.lr_init = 1.e-4
		self.lr_exp_decay_per_epoch = 0.97
		self.l2_reg = 1.e-4
		self.dropout_rate = 0.3
		self.num_conv_layers = 2
		self.conv_filters = 300
		self.conv_width = 7
		self.conv_stride = 1
		self.max_pool_size = 26
		self.max_pool_stride = 26
		self.num_dense_layers = 1
		self.dense_filters = 300

	def __getitem__(self, key):
		return getattr(self, key)

	def update(self, data, allow_val_change):
		for k, v in data.items():
			setattr(self, k, v)

class MockWandbCallback:
	def __init__(self):
		pass

	def __call__(self):
		pass