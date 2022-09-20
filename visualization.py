"""visualization.py: Tools for visualizing data."""
import matplotlib.pyplot as plt
import numpy as np
import umap

def get_umap(fit_data, fit_labels=None, transform_data=None, transform_labels=None, transform_outfile=None, plot_outfile=None, umap_kwargs=None, scatter_kwargs=None):
	"""Fit a UMAP reducer, and optionally transform and plot.

	Args:
		fit_data (np array), shape (num_samples, num_features): Data to fit the UMAP transform.
		fit_labels (np array), shape (num_samples,): If given, then use labels for
			supervised dimension reduction.
		transform_data (np array), shape (num_samples_tx, num_features): Data to transform
			after UMAP is fit. If not passed, then fit_data will be transformed.
		transform_labels (np array), shape (num_samples_tx,): If given, then use these labels
			to color the points in the visualization.
		transform_outfile (str): Path to save transformed data, .npy.
		plot_outfile (str): Path to save plot of transformed data, .jpg or .png.
		scatter_kwargs (dict): Keyword arguments to matplotlib scatter().
		umap_kwargs (dict): Keyword arguments to create the UMAP reducer.
	"""
	# Fit UMAP transform
	umap_kwargs = umap_kwargs or {}
	reducer = umap.UMAP(**umap_kwargs)
	print("Fitting UMAP transform...")
	reducer.fit(fit_data, y=fit_labels)

	# Transform data
	if transform_data is None:
		transform_data = fit_data
	print("Transforming data...")
	transformed = reducer.transform(transform_data)
	if transform_outfile is not None:
		np.save(transform_outfile, transformed)

	# Plot transformed data
	if plot_outfile is not None:
		print("Plotting...")
		# Create default scatter_kwargs as an empty dict
		scatter_kwargs = scatter_kwargs or {}
		if transform_labels is not None:
			# Color the samples by their label
			scatter_kwargs['c'] = transform_labels
			scatter_kwargs['cmap'] = 'Spectral'
		plt.scatter(transformed[:, 0], transformed[:, 1], **scatter_kwargs)
		if transform_labels is not None:
			# Add color legend
			num_labels = len(set(transform_labels))
			plt.colorbar(boundaries=np.arange(num_labels + 1) - 0.5).set_ticks(np.arange(num_labels))			
		plt.savefig(plot_outfile)

	return reducer, transformed