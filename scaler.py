"""scaler.py: transform narrowPeak data to zero mean, unit variance


testing:
DATA_DIR=/projects/pfenninggroup/heather/ad_ml_mpra/data/AD_ML_MPRA/expand_peaks_500bp_hg38/
FIT_FILES="$DATA_DIR/THP1_LPSIFNG_4hrs.idr.optimal_peak.narrowPeak.train.bed $DATA_DIR/THP1_LPSIFNG_4hrs.idr.optimal_peak.narrowPeak.train.Neg.bed"
TRANSFORM_FILES=$FIT_FILES
TRANSFORM_OUT_DIR=/home/csestili/test_transform_out/
DATA_COLUMN=6
python scaler.py -fit_files $FIT_FILES -transform_files $TRANSFORM_FILES -transform_out_dir $TRANSFORM_OUT_DIR -data_column $DATA_COLUMN



"""

import numpy as np
import pandas as pd
import sklearn.preprocessing

import os
import pickle


def main(fit_files, scaler_file, transform_files, transform_out_dir, data_column):
	if (fit_files is None) and (scaler_file is None):
		raise IOError('Must pass either fit files or scaler file!')

	if fit_files and scaler_file:
		raise IOError('Must pass fit files or scaler file, not both!')

	if fit_files:
		# Fit the scaler and save it
		scaler = get_scaler(fit_files, data_column)
		save_scaler(scaler, fit_files, transform_out_dir, data_column)
	elif scaler_file:
		# Load the saved scaler
		scaler = load_scaler(scaler_file)['scaler']

	if transform_files:
		# Use the scaler to transform the files
		scale(transform_files, scaler, transform_out_dir, data_column)

def get_scaler(fit_files, data_column):
	fit_data = []
	for in_file in fit_files:
		df = load_narrowpeak(in_file)
		cur_data = df[data_column]
		fit_data.append(cur_data)

	fit_data = [x for d in fit_data for x in d]
	# reshape to (n_samples, n_features) = (n_samples, 1)
	fit_data = np.array(fit_data)[:, np.newaxis]
	scaler = sklearn.preprocessing.StandardScaler().fit(fit_data)
	return scaler

def save_scaler(scaler, fit_files, transform_out_dir, data_column):

	scaler_data = {
		'scaler': scaler,
		'fit_files': fit_files,
		'data_column': data_column
	}

	os.makedirs(transform_out_dir, exist_ok=True)
	with open(os.path.join(transform_out_dir, 'scaler_data.pkl'), 'wb') as handle:
	    pickle.dump(scaler_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_scaler(scaler_data_path):
	with open(scaler_data_path, 'rb') as handle:
		scaler_data = pickle.load(handle)
	return scaler_data

def scale(transform_files, scaler, transform_out_dir, data_column):
	for in_file in transform_files:
		fname = os.path.basename(in_file)
		out_file = os.path.join(transform_out_dir, fname)
		df = load_narrowpeak(in_file)
		# reshape to (n_samples, n_features) = (n_samples, 1)
		data = np.array(df[data_column])[:, np.newaxis]
		new_data = scaler.transform(data)
		# reshape to (n_samples,)
		df[data_column] = new_data[:, 0]
		save_narrowpeak(df, out_file)

def load_narrowpeak(in_file):
	df = pd.read_csv(in_file, delim_whitespace=True, header=None, na_values=['.'])
	return df

def save_narrowpeak(df, out_file):
	_make_parent_dir(out_file)
	df.to_csv(out_file, index=False, sep="\t", header=None, na_rep='.')

def _make_parent_dir(path):
	"""Make parent directory of path"""
	os.makedirs(os.path.abspath(os.path.dirname(path)), exist_ok=True)

def get_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-fit_files', nargs='+')
	parser.add_argument('-scaler_file', type=str)
	parser.add_argument('-transform_files', nargs='+')
	parser.add_argument('-transform_out_dir', type=str, required=True)
	parser.add_argument('-data_column', type=int, required=True)
	args, _ = parser.parse_known_args()
	return args

if __name__ == '__main__':
	args = get_args()
	main(args.fit_files, args.scaler_file, args.transform_files, args.transform_out_dir, args.data_column)