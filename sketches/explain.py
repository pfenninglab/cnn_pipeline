"""explain.py: Get SHAP importance scores and TF-MoDISco motifs.

NOTE this must be run in an environment with tensorflow 2.4.1.
Tested with a conda env created from ../keras2-tf24.yml.
"""

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


def explain(args):
	init(args)
	bg, fg = get_data()
	shap_values = get_deepshap_scores(bg, fg)
	modisco_results = get_modisco_results(shap_values, fg)
	return modisco_results

def init(args):
	config, project = utils.get_config(args.config)
	wandb.init(config=config, project=project, mode="disabled")
	utils.validate_config(wandb.config)

def get_data():
	# Background is training set
	bg_data = dataset.SequenceTfDataset(wandb.config.train_data_paths, wandb.config.train_labels,
		targets_are_classes=True)
	# Foreground is positive examples from validation set
	fg_idx = np.array(wandb.config.val_labels) == wandb.config.shap_pos_label
	fg_data = dataset.SequenceTfDataset(
		list(np.array(wandb.config.val_data_paths)[fg_idx]),
		list(np.array(wandb.config.val_labels)[fg_idx]),
		targets_are_classes=True)

	bg, _ = bg_data.get_subset_as_arrays(wandb.config.shap_num_bg)
	fg, _ = fg_data.get_subset_as_arrays(wandb.config.shap_num_fg)

	return bg, fg

def get_deepshap_scores(bg, fg):
	model = models.load_model(wandb.config.interp_model_path)
	explainer = shap.DeepExplainer(model, bg)
	shap_values = explainer.shap_values(fg)

	return shap_values

class ModiscoNormalization:
	VALID_NORMALIZATION_TYPES = ('none', 'gkm_explain', 'pointwise')

	def __init__(self, normalization_type):
		normalization_type = normalization_type.lower()
		if normalization_type not in self.VALID_NORMALIZATION_TYPES:
			raise ValueError(f"Invalid normalization type. Allowed values are {self.VALID_NORMALIZATION_TYPES}, got {normalization_type}")
		self.normalization_type = normalization_type

	def __call__(self, hyp_impscores, sequences):
		"""
		Args:
			hyp_impscores [num_sequences, seq_len, 4]: hypothetical importance scores
			sequences [num_sequences, seq_len, 4]: 1-hot encoded actual sequences

		Returns:
			normed_impscores [num_sequences, seq_len, 4]
			normed_hyp_impscores [num_sequences, seq_len, 4]
		"""
		if self.normalization_type == 'none':
			return self.identity_normalization(hyp_impscores, sequences)
		elif self.normalization_type == 'gkm_explain':
			return self.gkm_explain_normalization(hyp_impscores, sequences)
		elif self.normalization_type == 'pointwise':
			return self.pointwise_normalization(hyp_impscores, sequences)
		else:
			raise NotImplementedError(self.normalization_type)

	@staticmethod
	def identity_normalization(hyp_impscores, sequences):
		normed_hyp_impscores = hyp_impscores
		normed_impscores = normed_hyp_impscores * sequences
		return normed_impscores, normed_hyp_impscores

	@staticmethod
	def gkm_explain_normalization(hyp_impscores, sequences):
		# implements equations (27) and (28) from A.3 of GkmExplain supplementary material
		# https://academic.oup.com/bioinformatics/article/35/14/i173/5529147#supplementary-data

		# actual_scores is f_h(S_x, i, S_x^i)
		actual_scores = np.expand_dims(np.sum(hyp_impscores * sequences, axis=-1), axis=-1)
		numerator = hyp_impscores * actual_scores
		denominator = np.expand_dims(np.sum(hyp_impscores * (hyp_impscores * actual_scores > 0), axis=-1), axis=-1)

		# numerical fix: it is possible for some entries in the denominator to be 0!
		# however, this happens only when the actual score f_h(S_x, i, S_x^i) is 0.
		# therefore, the correct normalized score at this position should be 0 as well.
		# we will set the denominator equal to 1 at this position, because
		# 0 / 1 = 0 yields the correct normalized answer in this special case.
		denominator = denominator + np.ones_like(denominator) * (denominator == 0)

		normed_hyp_impscores = numerator / denominator
		normed_impscores = normed_hyp_impscores * sequences

		return normed_impscores, normed_hyp_impscores

	@staticmethod
	def pointwise_normalization(hyp_impscores, sequences):
		# adapted from https://github.com/kundajelab/tfmodisco/blob/master/examples/H1ESC_Nanog_gkmsvm/TF%20MoDISco%20Nanog.ipynb
		# TODO vectorize
		normed_hyp_impscores = np.array([x - np.mean(x, axis=-1)[:,None] for x in hyp_impscores])
		normed_impscores = normed_hyp_impscores * sequences
		return normed_impscores, normed_hyp_impscores


def get_modisco_results(shap_values, fg):
	# Get normalized importance scores
	hyp_imp_scores = shap_values[wandb.config.shap_pos_label]
	normalization = ModiscoNormalization(wandb.config.modisco_normalization)
	normed_impscores, normed_hyp_impscores = normalization(hyp_imp_scores, fg)

	# Run TF-MoDISco
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

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, required=True)
	return parser.parse_args()


if __name__ == '__main__':
	explain(get_args())