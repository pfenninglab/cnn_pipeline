"""validate.py: Evaluate a trained model on a collection of validation sets.

Usage: python -m scripts.validate -config <path to config .yaml> -model <path to model .h5>
"""
import pprint

import wandb
import pandas as pd

import models

def validate(config_path, model_path, out_csv):
	wandb.init(config=config_path, mode='disabled')
	res = models.validate(wandb.config, model_path)
	if out_csv is not None:
		# 2 columns: metric_name, metric_value
		df = pd.DataFrame({k: [v] for k, v in res.items()}).transpose()
		df.to_csv(out_csv, header=False)
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint({k: f"{v:0.4}" for k, v in res.items()})
	return res

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, required=True, help='Path to config .yaml file')
	parser.add_argument('-model', type=str, required=True, help='Path to trained model .h5 file')
	parser.add_argument('-csv', type=str, help='(Optional) Path to save results as a .csv file')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	validate(args.config, args.model, args.csv)