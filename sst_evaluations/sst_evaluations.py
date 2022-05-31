# allow importing from one directory up
import sys
sys.path.append('..')
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

import dataset, models

model_paths = [
	("/home/csestili/repos/mouse_sst_clean/wandb/run-20220525_113317-pzz35myd/files/model-best.h5", "best auprc"),
	("/home/csestili/repos/mouse_sst_clean/wandb/run-20220525_080410-dho6horg/files/model-best.h5", "best sensitivity with precision > 0.5"),
	("/home/csestili/repos/mouse_sst_clean/wandb/run-20220502_160831-8joj4ofy/files/model-best.h5", "earlier sweep best auprc")
]

eval_dir = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/evaluations/SST/"
eval_files = [
	("Eval1_mm10_VAL.fa", 1),
	("Eval2_hg38_VAL.fa", 0),
	("Eval3_hg38_VAL.fa", 1),
	("Eval4_mm10_VAL.fa", 0),
	("Eval5_mm10_VAL.fa", 1),
	("Eval6_mm10_VAL.fa", 1),
	("Eval7_mm10_VAL.fa", 0),
	("Eval8_mm10_VAL.fa", 1),
	("Eval9_mm10_VAL.fa", 1),
	("Eval10_mm10_VAL.fa", 0)
]

out_path = "/home/csestili/repos/mouse_sst/sst_evaluations/results.csv"

def do_all_evals():
	results = []
	for model_path, desc in tqdm(model_paths, desc="model"):
		row = {
			'model_path': model_path,
			'desc': desc
		}
		model = models.load_model(model_path)
		for idx, (eval_file, expected_label) in tqdm(enumerate(eval_files), total=len(eval_files), desc="eval set", leave=False):
			eval_fa_path = os.path.join(eval_dir, eval_file)
			res = validate(model, eval_fa_path, expected_label)

			row[f'eval_{idx + 1}'] = res['acc']

		results.append(row)

	results = pd.DataFrame(results).round(4)
	columns = [f'eval_{idx + 1}' for idx in range(10)] + ['model_path', 'desc']
	results.to_csv(out_path, index=False, columns=columns, sep="\t")

def validate(model, eval_fa, expected_label):
	val_data = dataset.SequenceTfDataset([eval_fa], [expected_label], targets_are_classes=True, endless=False, map_targets=False)
	y = val_data.dataset[1]
	return model.evaluate(x=val_data.dataset[0], y=y, batch_size=512, return_dict=True, verbose=0)

if __name__ == '__main__':
	do_all_evals()