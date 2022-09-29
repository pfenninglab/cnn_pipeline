import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

import visualization
import models


# multispecies PV model
MODEL_PATH = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/multispecies_PV/models/FINAL_modelmultiPVi.h5'
LAYER_NAME = 'activation_1'
# # mouse-only PV model
# MODEL_PATH = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_PV/models/FINAL_modelPV3e.h5'
# # [...] -> maxpool -> flatten -> dense(300) -> activation_9 -> [...]
# LAYER_NAME = 'activation_9'



DATA_DIR = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/multispecies_PV/FinalModelData'
FG_POS_PATH = os.path.join(DATA_DIR, 'mouse_PV_pos_VAL.fa')
FG_NEG_PATH_FIT = os.path.join(DATA_DIR, 'mouse_PV_neg_VAL.fa')
# mouse non-enhancer orthologs of human enhancers
FG_NEG_PATH_TX = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/evaluations/PV/Eval4_mm10.fa'
BG_POS_PATH = os.path.join(DATA_DIR, 'human_PV_pos_VAL.fa')
BG_NEG_PATH_FIT = os.path.join(DATA_DIR, 'human_PV_neg_VAL.fa')
# human non-enhancer orthologs of mouse enhancers
BG_NEG_PATH_TX = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/evaluations/PV/Eval2_hg38.fa'
ACTIVATIONS_DIR = '/home/csestili/data/tacit_viz/mouse_pv/'

REDUCER_TYPE = 'pca' # 'pca' or 'umap'
if REDUCER_TYPE not in ['pca', 'umap']:
	raise NotImplementedError()

PV_DATA_DIR = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/multispecies_PV/FinalModelData'
PV_EVAL_DIR = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/evaluations/PV/'
DATA_MAPPING = {
	'mouse_neg_val': os.path.join(PV_DATA_DIR, 'mouse_PV_neg_VAL.fa'),
	'combined_pos_val': os.path.join(PV_DATA_DIR, 'combined_PV_pos_VAL.fa'),
	'combined_neg_val': os.path.join(PV_DATA_DIR, 'combined_PV_neg_VAL.fa'),
	'human_pos_train': os.path.join(PV_DATA_DIR, 'human_PV_pos_TRAIN.fa'),
	'human_pos_val': os.path.join(PV_DATA_DIR, 'human_PV_pos_VAL.fa'),
	'human_pos_test': os.path.join(PV_DATA_DIR, 'human_PV_pos_TEST.fa'),
	'mouse_pos_train': os.path.join(PV_DATA_DIR, 'mouse_PV_pos_TRAIN.fa'),
	'mouse_pos_val': os.path.join(PV_DATA_DIR, 'mouse_PV_pos_VAL.fa'),
	'mouse_pos_test': os.path.join(PV_DATA_DIR, 'mouse_PV_pos_TEST.fa'),
	'human_neg_neoe_all': os.path.join(PV_EVAL_DIR, 'Eval4_mm10.fa'),
	'mouse_neg_neoe_all': os.path.join(PV_EVAL_DIR, 'Eval2_hg38.fa')
}
VISUALIZATION_MAPPING = {
	#multispecies
	'fit_data': ['combined_pos_val', 'combined_neg_val'],
	# #mouse-only
	# 'fit_data': ['mouse_pos_val', 'mouse_neg_val'],
	'plot_data': [
		{
			'title': 'Positives',
			'groups': [
				{'id': 2, 'name': 'Human +', 'sets': ['human_pos_train', 'human_pos_val', 'human_pos_test']},
				{'id': 3, 'name': 'Mouse +', 'sets': ['mouse_pos_train', 'mouse_pos_val', 'mouse_pos_test']}
			]
		},
		{
			'title': 'Human',
			'groups': [
				{'id': 0, 'name': 'Human - (neoe)', 'sets': ['human_neg_neoe_all']},
				{'id': 2, 'name': 'Human +', 'sets': ['human_pos_train', 'human_pos_val', 'human_pos_test']}
			]
		},
		{
			'title': 'Mouse',
			'groups': [
				{'id': 1, 'name': 'Mouse - (neoe)', 'sets': ['mouse_neg_neoe_all']},
				{'id': 3, 'name': 'Mouse +', 'sets': ['mouse_pos_train', 'mouse_pos_val', 'mouse_pos_test']}
			]
		}
	]
}

def main():

	# Compile list of datasets that need activations
	datasets = VISUALIZATION_MAPPING['fit_data'] + [name for plot_spec in VISUALIZATION_MAPPING['plot_data'] for group in plot_spec['groups'] for name in group['sets']]
	datasets = list(set(datasets))

	# Get activations for each dataset
	print("Getting activations...")
	activations = {}
	if not os.path.exists(ACTIVATIONS_DIR):
		os.makedirs(ACTIVATIONS_DIR)
	model = None
	for name in datasets:
		activations_path = os.path.join(ACTIVATIONS_DIR, f"{name}_activations.npy")
		if not os.path.exists(activations_path):
			if model is None:
				model = models.load_model(MODEL_PATH)
			path = DATA_MAPPING[name]
			activations[name] = models.get_activations(
				model, path, out_file=activations_path, layer_name=LAYER_NAME)
		else:
			activations[name] = np.load(activations_path)

	# Fit reducer
	reducer_path = os.path.join(ACTIVATIONS_DIR, f"{REDUCER_TYPE}_reducer.pkl")
	if not os.path.exists(reducer_path):
		fit_data = np.concatenate(
			[activations[name] for name in VISUALIZATION_MAPPING['fit_data']],
			axis=0)
		fit_fn = visualization.umap_fit if REDUCER_TYPE == 'umap' else visualization.pca_fit
		reducer = fit_fn(fit_data, reducer_outfile=reducer_path)
	else:
		with open(reducer_path, 'rb') as f:
			reducer = pickle.load(f)

	# Transform and plot
	for plot_spec in VISUALIZATION_MAPPING['plot_data']:
		# Get the relevant activations
		transform_data = [activations[name] for group in plot_spec['groups'] for name in group['sets']]
		transform_data = np.concatenate(transform_data, axis=0)
		transform_labels = [
			group['id'] for group in plot_spec['groups']
			for name in group['sets']
			for _ in range(len(activations[name]))]
		transform_labels = np.array(transform_labels)

		# # Add 1 instance from each class for legend colors
		# extra_points = np.stack(tuple(activations[set_name][0] for set_name in ['fg_pos', 'fg_neg_tx', 'bg_pos', 'bg_neg_tx']))
		# extra_labels = np.array([0, 1, 2, 3])
		# transform_data = np.concatenate((transform_data, extra_points), axis=0)
		# transform_labels = np.concatenate((transform_labels, extra_labels), axis=0)

		# Shuffle for visibility of all classes
		rng = np.random.default_rng()
		combined = np.concatenate((transform_data, np.expand_dims(transform_labels, axis=1)), axis=1)
		rng.shuffle(combined)
		transform_data, transform_labels = combined[:, :-1], combined[:, -1]

		# Transform
		print("Getting transformed samples...")
		title = plot_spec['title']
		transform_outfile = os.path.join(ACTIVATIONS_DIR, f"{title}_{REDUCER_TYPE}.npy")
		transform_label_outfile = os.path.join(ACTIVATIONS_DIR, f"{title}_{REDUCER_TYPE}_labels.npy")
		if not os.path.exists(transform_outfile):
			transformed = visualization.transform(reducer, transform_data, transform_outfile=transform_outfile)
			np.save(transform_label_outfile, transform_labels)
		else:
			transformed = np.load(transform_outfile)
			transform_labels = np.load(transform_label_outfile)

		# Rank-Sum test on whether the first PC separates
		sample1 = (transform_labels == plot_spec['groups'][0]['id']).nonzero()[0]
		sample2 = (transform_labels == plot_spec['groups'][1]['id']).nonzero()[0]
		ranksum_result = scipy.stats.ranksums(transformed[sample1, 0], transformed[sample2, 0])
		print(f"{title}: {ranksum_result}")

		# Visualize
		plot_outfile = os.path.join(ACTIVATIONS_DIR, f"{title}_{REDUCER_TYPE}.png")
		label_mapping = {group['id']: group['name'] for group in plot_spec['groups']}
		fig, axs = visualization.scatter(transformed, plot_outfile, transform_labels=transform_labels,
			label_mapping=label_mapping, scatter_kwargs={"s": 1.5, "alpha": 0.7}, add_histogram=True)
		fig.suptitle(f"{title}")
		axs[0].set_title("First 2 PCs")
		axs[0].set_xlabel("PC 1")
		axs[0].set_ylabel("PC 2")
		axs[1].set_title("First PC")
		axs[1].set_xlabel(f"PC 1\nstatistic = {ranksum_result.statistic}\np = {ranksum_result.pvalue}")
		axs[1].set_ylabel("Density")
		plt.savefig(plot_outfile, dpi=300)

if __name__ == '__main__':
	main()