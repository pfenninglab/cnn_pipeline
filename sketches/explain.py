# allow importing from one directory up
import sys
sys.path.append('..')

import shap
import modisco
import wandb
import numpy as np
import tensorflow as tf

import models
import dataset
import utils


# TODO convert constants to config settings
NUM_BG = 100
NUM_FG = 20
MODEL_PATH = "/home/csestili/repos/mouse_sst/wandb/run-20220408_112918-eg1rb9tq/files/model-best.h5"


def explain():
    init()
    bg, fg = get_data(NUM_BG, NUM_FG)
    shap_values = get_deepshap_scores(MODEL_PATH, bg, fg)
    modisco_results = get_modisco_results(shap_values, fg)
    return modisco_results

def init():
	config, project = utils.get_config("../config-mouse-sst.yaml")
	wandb.init(config=config, project=project, mode="disabled")
	utils.validate_config(wandb.config)

def get_data(num_bg, num_fg):
	train_data = dataset.FastaTfDataset(wandb.config.train_data_paths, wandb.config.train_labels)
	val_data = dataset.FastaTfDataset(wandb.config.val_data_paths, wandb.config.val_labels)

	bg = np.array([itm[0] for itm in train_data.ds.take(num_bg).as_numpy_iterator()])
	fg = np.array([itm[0] for itm in val_data.ds.take(num_fg).as_numpy_iterator()])

	return bg, fg

def get_deepshap_scores(model_path, bg, fg):
	model = models.load_model(model_path)

	explainer = shap.DeepExplainer(model, bg)
	shap_values = explainer.shap_values(fg)

	return shap_values

def get_modisco_results(shap_values, fg):
	# NOTE currently these scores are NOT normalized
	normed_impscores = shap_values[0]
	normed_hyp_impscores = shap_values[0]

	seqlets_to_patterns_factory = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
        trim_to_window_size=11,
        initial_flank_to_add=3,
        final_flank_to_add=3,
        kmer_len=7, num_gaps=1,
        num_mismatches=1)
	workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        sliding_window_size=11,
        flank_size=3,
        seqlets_to_patterns_factory=seqlets_to_patterns_factory)
	tfmodisco_results = workflow(
	                task_names=["task0"],
	                contrib_scores={'task0': normed_impscores},
	                hypothetical_contribs={'task0': normed_hyp_impscores},
	                one_hot=fg)

	return tfmodisco_results


if __name__ == '__main__':
	explain()