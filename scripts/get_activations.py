"""get_activations.py: Get activations from a trained model's intermediate or output layers.

Usage: python scripts/get_activations.py \
	-model <path to model .h5>
	-in_file <path to input .fa, .bed, or .narrowPeak file> \
	[-in_genome <path to genome .fa file, if in_file is .bed or .narrowPeak>] \
	-out_file <path to output file, .npy or .csv> \
	[-layer_name <layer name to get activations from, e.g. 'flatten'>. default is output layer] \
	[--no_reverse_complement, don't evaluate on reverse complement sequences] \
	[--write_csv, write activations as .csv file instead of .npy] \
	[-score_column <output unit to extract score in the csv, e.g. 1>. default writes whole activation as a row]

To get a numpy array of activations from an intermediate layer:
	-layer_name <layer_name>
	[don't pass --write_csv]

To get a csv of probabilities for the positive class from a binary classifier:
	[don't pass -layer_name]
	--write_csv
	-score_column 1
"""
import argparse

from models import get_activations

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, required=True)
	parser.add_argument('-in_file', type=str, required=True)
	parser.add_argument('-in_genome', type=str, required=False)
	parser.add_argument('-out_file', type=str, required=True)
	parser.add_argument('-layer_name', type=str, required=False)
	parser.add_argument('--no_reverse_complement', action='store_true')
	parser.add_argument('--write_csv', action='store_true')
	parser.add_argument('-score_column', type=int, required=False)
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	get_activations(args.model, args.in_file,
		in_genome=args.in_genome,
		out_file=args.out_file,
		layer_name=args.layer_name,
		use_reverse_complement=not args.no_reverse_complement,
		write_csv=args.write_csv,
		score_column=args.score_column)