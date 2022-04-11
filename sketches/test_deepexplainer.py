import shap
import wandb
import numpy as np

import models
import dataset
import utils


def test_deepexplainer(model_path):
	model = models.load_model(model_path)

	# Start `wandb`
	config, project = utils.get_config("config-mouse-sst.yaml")
	wandb.init(config=config, project=project, mode="disabled")
	utils.validate_config(wandb.config)

	# Get datasets
	train_data = dataset.FastaTfDataset(wandb.config.train_data_paths, wandb.config.train_labels)
	val_data = dataset.FastaTfDataset(wandb.config.val_data_paths, wandb.config.val_labels)

	bg = np.array([itm[0] for itm in train_data.ds.take(10).as_numpy_iterator()])
	fg = np.array([itm[0] for itm in val_data.ds.take(10).as_numpy_iterator()])

	explainer = shap.DeepExplainer(model, bg)
	values = explainer.shap_values(fg)

	return values
