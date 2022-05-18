"""preprocessing.py: Data preprocessing

Usage:
python preprocessing.py expand_peaks -b <input bed file> -o <output bed file> -w 501
"""

import numpy as np
import pandas as pd


def main(args):
	if args.function == 'expand_peaks':
		expand_peaks(args.bed_file, args.out_file, args.width)
	else:
		raise ValueError(f"Invalid args: {args}")

def expand_peaks(bed_file, out_file, width):
    """Standardize bed file peaks to a uniform width.
    Adapted from expand_peaks.py by Calvin Chen.

    Actions:
    - Drop duplicate peaks
    - Expand peaks to the same width, keeping the center at the original place

    Args:
    	bed_file (str)
    	out_file (str)
    	width (int)
    """
    if out_file == bed_file:
    	raise ValueError("Don't overwrite old bed file")

    import pandas as pd
    bed = pd.read_csv(bed_file, sep="\t", header=None)

    # Drop duplicate peaks
    bed= bed.drop_duplicates(subset=[0,1,2], keep='first', inplace=False)

    # Expand peaks to the same width
    midpoint = (bed[1] + bed[2]) / 2
    bed[1] = np.floor(midpoint - width / 2).astype(int)
    bed[2] = bed[1] + width

    bed.to_csv(out_file, index=False, sep="\t", header=None)

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('function')
	parser.add_argument('--bed_file', '-b')
	parser.add_argument('--out_file', '-o')
	parser.add_argument('--width', '-w', type=int)
	return parser.parse_args()

if __name__ == '__main__':
	main(get_args())
