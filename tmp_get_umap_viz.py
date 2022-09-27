import os
import pickle

import numpy as np
from tqdm import tqdm

import visualization
import models


# multispecies PV model
MODEL_PATH = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/multispecies_PV/models/FINAL_modelmultiPVi.h5'
LAYER_NAME = 'activation_1'
# mouse-only PV model
# MODEL_PATH = '/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_PV/models/FINAL_modelPV3e.h5'
# [...] -> maxpool -> flatten -> dense(300) -> activation_9 -> [...]
#LAYER_NAME = 'activation_9'



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

def main():
	# Get activations
	print("Getting activations...")
	activations = {}
	if not os.path.exists(ACTIVATIONS_DIR):
		os.makedirs(ACTIVATIONS_DIR)
	model = None
	for (path, name) in tqdm([
		(FG_POS_PATH, 'fg_pos'), (FG_NEG_PATH_FIT, 'fg_neg_fit'), (FG_NEG_PATH_TX, 'fg_neg_tx'),
		(BG_POS_PATH, 'bg_pos'), (BG_NEG_PATH_FIT, 'bg_neg_fit'), (BG_NEG_PATH_TX, 'bg_neg_tx')]):
		activations_path = os.path.join(ACTIVATIONS_DIR, f"{name}.npy")
		if not os.path.exists(activations_path):
			if model is None:
				model = models.load_model(MODEL_PATH)
			activations[name] = models.get_activations(
				model, path, out_file=activations_path, layer_name=LAYER_NAME)
		else:
			activations[name] = np.load(activations_path)

	# Fit reducer on the val data seen during training
	reducer_path = os.path.join(ACTIVATIONS_DIR, f"{REDUCER_TYPE}_reducer.pkl")
	if not os.path.exists(reducer_path):
		fit_data = np.concatenate((
			activations['fg_pos'], activations['fg_neg_fit'],
			activations['bg_pos'], activations['bg_neg_fit']), axis=0)
		fit_fn = visualization.umap_fit if REDUCER_TYPE == 'umap' else visualization.pca_fit
		reducer = fit_fn(fit_data, reducer_outfile=reducer_path)
	else:
		with open(reducer_path, 'rb') as f:
			reducer = pickle.load(f)

	# Transform and visualize
	# labels for all plots
	for (set_a, label_a, set_b, label_b, name) in [
		('fg_pos', 0, 'bg_pos', 2, f'{REDUCER_TYPE}_positives'),
		('fg_pos', 0, 'fg_neg_tx', 1, f'{REDUCER_TYPE}_fg'),
		('bg_pos', 2, 'bg_neg_tx', 3, f'{REDUCER_TYPE}_bg')]:

		# Get the relevant activations
		transform_data = np.concatenate((activations[set_a], activations[set_b]), axis=0)
		transform_labels = np.array(
			[label_a] * len(activations[set_a]) +
			[label_b] * len(activations[set_b]))
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

		# Transform and visualize
		transform_outfile = os.path.join(ACTIVATIONS_DIR, f"{name}.npy")
		plot_outfile = os.path.join(ACTIVATIONS_DIR, f"{name}.png")
		label_mapping = {0: 'mouse PV +', 1: 'mouse PV - (neoe)', 2: 'human PV +', 3: 'human PV - (neoe)'}
		if not os.path.exists(transform_outfile):
			transformed = visualization.transform(reducer, transform_data, transform_outfile=transform_outfile)
		else:
			transformed = np.load(transform_outfile)
		visualization.scatter(transformed, plot_outfile, transform_labels=transform_labels,
			label_mapping=label_mapping, scatter_kwargs={"s": 1.5}, add_histogram=True)

if __name__ == '__main__':
	main()