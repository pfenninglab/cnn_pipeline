"""get_activations.py: Get activations from a trained model's intermediate or output layers.

Usage: python scripts/get_activations.py \
	-model <path to model .h5>
	-in_files <paths to input .fa, .bed, or .narrowPeak file> \
	[-in_genomes <paths to genome .fa file, if in_file is .bed or .narrowPeak>] \
	-out_file <path to output file, .npy or .csv> \
	[-layer_name <layer name to get activations from, e.g. 'flatten'>. default is output layer] \
	[--no_reverse_complement, don't evaluate on reverse complement sequences] \
	[--write_csv, write activations as .csv file instead of .npy] \
	[-score_column <output unit to extract score in the csv, e.g. 1>. default writes whole activation as a row]

Examples:

1. Model is a binary classifier, output .csv file of probabilities for the positive class:
	[don't pass -layer_name]
	--write_csv
	-score_column 1
	(optional: --bayesian to get Bayesian predictions)

2. Model is a regression model, output .csv file of predicted values:
	[don't pass -layer_name]
	--write_csv
	-score_column 0
	(optional: --bayesian to get Bayesian predictions)

3. Model is classification or regression, output .npy file of inner-layer activations:
	-layer_name <layer_name>
	[don't pass --write_csv]
"""
import argparse

from models import get_activations

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, required=True)
	parser.add_argument('-in_files', nargs='+')
	parser.add_argument('-in_genomes', nargs='*')
	parser.add_argument('-out_file', type=str, required=True)
	parser.add_argument('-layer_name', type=str, required=False)
	parser.add_argument('--no_reverse_complement', action='store_true')
	parser.add_argument('--write_csv', action='store_true')
	parser.add_argument('-score_column', type=int, required=False)
	parser.add_argument('--bayesian', action='store_true')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	if args.in_genomes is not None:
		if len(args.in_files) != len(args.in_genomes):
			raise IOError(f'Different number of in_files and in_genomes! Got {len(args.in_files)} input files and {len(args.in_genomes)} genomes.')
	get_activations(args.model, args.in_files,
		in_genomes=args.in_genomes,
		out_file=args.out_file,
		layer_name=args.layer_name,
		use_reverse_complement=not args.no_reverse_complement,
		write_csv=args.write_csv,
		score_column=args.score_column,
		bayesian=args.bayesian)