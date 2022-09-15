"""visualization.py: Tools for visualizing data."""
import matplotlib.pyplot as plt
import numpy as np
import umap

def get_umap(fit_data, transform_data=None, transform_outfile=None, plot_outfile=None, umap_kwargs=None, scatter_kwargs=None):
	"""Fit a UMAP reducer, and optionally transform and plot.

	Args:
		fit_data (np array): Data to fit the UMAP transform.
		transform_data (np array): Data to transform after UMAP is fit.
			if not passed, then fit_data will be transformed.
		transform_outfile (str): Path to save transformed data, .npy.
		plot_outfile (str): Path to save plot of transformed data, .jpg or .png.
		scatter_kwargs (dict): Keyword arguments to matplotlib scatter().
		umap_kwargs (dict): Keyword arguments to create the UMAP reducer.
	"""
	# Fit UMAP transform
	umap_kwargs = umap_kwargs or {}
	reducer = umap.UMAP(**umap_kwargs)
	reducer.fit(fit_data)

	# Transform data
	if transform_data is None:
		transform_data = fit_data
	transformed = reducer.transform(transform_data)
	if transform_outfile is not None:
		np.save(transform_outfile, transformed)

	# Plot transformed data
	if plot_outfile is not None:
		scatter_kwargs = scatter_kwargs or {}
		plt.scatter(transform[:, 0], transform[:, 1], **scatter_kwargs)
		plt.savefig(plot_outfile)

	return reducer, transformed_data