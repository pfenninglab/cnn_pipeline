# allow importing from one directory up
import sys
sys.path.append('../sketches')

import numpy as np

from explain import ModiscoNormalization


def test_gkm_explain_normalization():
	# test case inputs
	hyp_impscores = np.array([[ 1, -2,  3, -4], [ -5,  6,  -7,  8], [-9,  1,   2, -3]])
	sequences     = np.array([[ 0,  1,  0,  0], [  0,  0,   0,  1], [ 1,  0,   0,  0]])

	# expected values (computed by hand)
	numerator     = np.array([[-2,  4, -6,  8], [-40, 48, -56, 64], [81, -9, -18, 27]])
	denominator   = np.array([[ 0, -2,  0, -4], [  0,  6,   0,  8], [-9,  0,   0, -3]])
	denominator   = np.sum(denominator, axis=-1)[:, np.newaxis]
	expected_hyp_imp = numerator / denominator
	expected_imp = expected_hyp_imp * sequences

	# computed values
	normed_impscores, normed_hyp_impscores = ModiscoNormalization('gkm_explain')(hyp_impscores, sequences)

	for arr in [expected_hyp_imp, expected_imp, normed_impscores, normed_hyp_impscores]:
		assert arr.shape == (3, 4)
	assert np.allclose(expected_hyp_imp, normed_hyp_impscores)
	assert np.allclose(expected_imp, normed_impscores)

	# test that it works when the batch dimension is added
	# shape = (1, 3, 4)
	hyp_impscores = hyp_impscores[np.newaxis, :]
	sequences = sequences[np.newaxis, :]
	numerator = numerator[np.newaxis, :]
	denominator = denominator[np.newaxis, :]
	expected_hyp_imp = numerator / denominator
	expected_imp = expected_hyp_imp * sequences
	normed_impscores, normed_hyp_impscores = ModiscoNormalization('gkm_explain')(hyp_impscores, sequences)
	for arr in [expected_hyp_imp, expected_imp, normed_impscores, normed_hyp_impscores]:
		assert arr.shape == (1, 3, 4)	
	assert np.allclose(expected_hyp_imp, normed_hyp_impscores)
	assert np.allclose(expected_imp, normed_impscores)

	# test that it works with batch size of 2
	# shape = (2, 3, 4)
	hyp_impscores = np.repeat(hyp_impscores, 2, axis=0)
	sequences = np.repeat(sequences, 2, axis=0)
	numerator = np.repeat(numerator, 2, axis=0)
	denominator = np.repeat(denominator, 2, axis=0)
	expected_hyp_imp = numerator / denominator
	expected_imp = expected_hyp_imp * sequences
	normed_impscores, normed_hyp_impscores = ModiscoNormalization('gkm_explain')(hyp_impscores, sequences)
	for arr in [expected_hyp_imp, expected_imp, normed_impscores, normed_hyp_impscores]:
		assert arr.shape == (2, 3, 4)	
	assert np.allclose(expected_hyp_imp, normed_hyp_impscores)
	assert np.allclose(expected_imp, normed_impscores)


if __name__ == '__main__':
	test_gkm_explain_normalization()