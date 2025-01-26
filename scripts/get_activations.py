"""get_activations.py: Get activations from a trained model's intermediate or output layers.
Usage: python scripts/get_activations.py \
    -model <path to model .h5>
    -in_files <paths to input .fa, .bed, or .narrowPeak file> \
    [-in_genomes <paths to genome .fa file, if in_file is .bed or .narrowPeak>] \
    -out_file <path to output file, .npy or .csv> \
    [-layer_name <layer name to get activations from, e.g. 'flatten'>. default is output layer] \
    [--no_reverse_complement, don't evaluate on reverse complement sequences] \
    [--write_csv, write activations as .csv file instead of .npy] \
    [-score_column <output unit to extract score, e.g. 1>. use 'all' to write all units] \
    [--bayesian, do Bayesian inference with N=64 trials] \
    [--name, include sequence names in CSV output] \
    [--aggregate <average|logit_average>, combine sequence and reverse complement predictions]

Examples:
1. Binary classifier with sequence names and logit-averaged predictions:
    [don't pass -layer_name]
    --write_csv
    --name
    --aggregate logit_average
    -score_column 1

2. Regression model with sequence predictions:
    [don't pass -layer_name]
    --write_csv
    -score_column 0
    (optional: --bayesian)

3. Inner-layer activations:
    -layer_name <layer_name>
    -score_column all
    [don't pass --write_csv]
"""

import argparse

from models import get_activations

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-in_files', required=True, nargs='+')
    parser.add_argument('-in_genomes', nargs='+')
    parser.add_argument('-out_file')
    parser.add_argument('-layer_name')
    parser.add_argument('--no_reverse_complement', action='store_false', dest='use_reverse_complement')
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('-score_column')
    parser.add_argument('--bayesian', action='store_true')
    parser.add_argument('--name', action='store_true', help='Include sequence names in CSV output')
    parser.add_argument('--aggregate', choices=['average', 'logit_average'],
                       help='How to combine sequence and reverse complement predictions')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    get_activations(args.model, args.in_files, args.in_genomes, args.out_file,
                   args.layer_name, args.use_reverse_complement, args.write_csv,
                   args.score_column, bayesian=args.bayesian, include_names=args.name,
                   aggregate_method=args.aggregate)