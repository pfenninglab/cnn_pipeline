"""visualization.py: Tools for visualizing data."""
import pickle

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
	# TODO DRY

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
			scatter_kwargs['cmap'] = 'gist_rainbow'
		plt.scatter(transformed[:, 0], transformed[:, 1], **scatter_kwargs)
		if transform_labels is not None:
			# Add color legend
			num_labels = len(set(transform_labels))
			plt.colorbar(boundaries=np.arange(num_labels + 1) - 0.5).set_ticks(np.arange(num_labels))			
		plt.savefig(plot_outfile, dpi=300)

	return reducer, transformed

def umap_fit(fit_data, fit_labels=None, umap_kwargs=None, reducer_outfile=None):
	# Fit UMAP transform
	umap_kwargs = umap_kwargs or {}
	reducer = umap.UMAP(**umap_kwargs)
	print("Fitting UMAP transform...")
	reducer.fit(fit_data, y=fit_labels)

	# Save fit reducer object
	if reducer_outfile is not None:
		with open(reducer_outfile, 'wb') as f:
			pickle.dump(reducer, f)

	return reducer

def umap_transform(reducer, transform_data, transform_labels=None, label_mapping=None, transform_outfile=None, plot_outfile=None, scatter_kwargs=None):
	# Transform data
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
			scatter_kwargs['cmap'] = 'cool'
		plt.clf()
		plot = plt.scatter(transformed[:, 0], transformed[:, 1], **scatter_kwargs)
		if transform_labels is not None:
			# Convert numerical labels to string in the legend
			lines, labels = plot.legend_elements()
			if label_mapping is not None:
				# Assumes labels is sorted unique transform labels
				labels = [label_mapping[itm] for itm in sorted(set(transform_labels))]
			# Add color legend and format figure
			plt.legend(lines, labels, loc='lower right', prop={'size': 5})
			plt.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
		plt.savefig(plot_outfile, dpi=300)

	return transformed