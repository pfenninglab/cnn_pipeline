"""preprocessing.py: Data preprocessing

Usage:
python preprocessing.py expand_peaks -i <input bed file> -o <output bed file> -l 501
"""

import numpy as np
import pandas as pd


CENTERING_OPTIONS = ['summit', 'endpoints']


def main(args):
	if args.function == 'expand_peaks':
		expand_peaks(args.in_file, args.out_file, args.length, centering=args.centering)
	else:
		raise ValueError(f"Invalid args: {args}")

def expand_peaks(bed_file, out_file, length, centering='summit'):
	"""Standardize bed file peaks to a uniform length.
	Adapted from expand_peaks.py by Calvin Chen.

	Actions:
	- Expand peaks to the same length
		- If centering == 'summit', then preserve the original summit locations
		- If centering == 'endpoints', then preserve the original interval centers
	- Drop duplicate peaks

	Args:
		bed_file (str)
		out_file (str)
		length (int)
		centering (str): how to calculate the center of the new peaks
			- if 'summit', then use the summit
			- if 'endpoints', then use the original center
	"""
	if out_file == bed_file:
		raise ValueError("Don't overwrite old bed file")

	if centering not in CENTERING_OPTIONS:
		raise ValueError(f"Invalid centering option `{centering}`, valid options are {CENTERING_OPTIONS}")

	bed = pd.read_csv(bed_file, delim_whitespace=True, header=None)
	has_summit_column = len(bed.columns) >= 10

	if centering == 'summit' and not has_summit_column:
		raise ValueError(f"Cannot use centering='summit' because bed file does not have summit column. "
						  "Check bed file or use centering='endpoints' if there is no summit column.")

	# Get summit index, to recompute summit after recentering
	if has_summit_column:
		# bed[1]: left endpoint index
		# bed[9]: summit offset
		# midpoint = summit index
		summit_idx = bed[1] + bed[9]		

	# Expand peaks to the same length
	if centering == 'summit':
		midpoint = summit_idx
	elif centering == 'endpoints':
		# bed[1]: left endpoint index
		# bed[2]: right endpoint index
		# midpoint = geometric midpoint of left and right, i.e. mean
		midpoint = (bed[1] + bed[2]) / 2
	else:
		raise ValueError(f"Invalid centering option `{centering}`, valid options are {CENTERING_OPTIONS}")

	# Recompute endpoints
	bed[1] = np.floor(midpoint - length / 2).astype(int)
	bed[2] = bed[1] + length

	if has_summit_column:
		# Recompute summit
		bed[9] = summit_idx - bed[1]

	# Drop duplicate peaks
	bed = bed.drop_duplicates(subset=[0,1,2], keep='first', inplace=False)

	bed.to_csv(out_file, index=False, sep="\t", header=None)

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('function')
	parser.add_argument('--in_file', '-i')
	parser.add_argument('--out_file', '-o')
	parser.add_argument('--length', '-l', type=int)
	parser.add_argument('--centering', '-c', default='summit')
	return parser.parse_args()

if __name__ == '__main__':
	main(get_args())
