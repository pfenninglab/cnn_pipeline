"""visualization.py: Tools for visualizing data."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn
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

def transform(reducer, transform_data, transform_outfile=None):
	print("Transforming data...")
	transformed = reducer.transform(transform_data)
	if transform_outfile is not None:
		np.save(transform_outfile, transformed)

	return transformed

def pca_fit(fit_data, pca_kwargs=None, reducer_outfile=None):
	# TODO DRY
	# Fit PCA transform
	pca_kwargs = pca_kwargs or {}
	reducer = sklearn.decomposition.PCA(**pca_kwargs)
	print("Fitting PCA transform...")
	reducer.fit(fit_data)
	print(f"PCA variance explained: {reducer.explained_variance_}")

	# Save fit reducer object
	if reducer_outfile is not None:
		with open(reducer_outfile, 'wb') as f:
			pickle.dump(reducer, f)

	return reducer

def scatter(points, plot_outfile, transform_labels=None, label_mapping=None, scatter_kwargs=None, add_histogram=False):
	print("Plotting...")
	# Create default scatter_kwargs as an empty dict
	scatter_kwargs = scatter_kwargs or {}
	cmap = plt.cm.get_cmap('cool')
	if transform_labels is not None:
		# Color the samples by their label
		scatter_kwargs['c'] = transform_labels
		scatter_kwargs['cmap'] = cmap

	plt.clf()
	nrows = 2 if add_histogram else 1 
	fig, axs = plt.subplots(nrows=nrows, sharex=True)

	# Scatter plot
	plot = axs[0].scatter(points[:, 0], points[:, 1], **scatter_kwargs)
	if transform_labels is not None:
		# Convert numerical labels to string in the legend
		lines, labels = plot.legend_elements()
		if label_mapping is not None:
			# Assumes labels is sorted unique transform labels
			labels = [label_mapping[itm] for itm in sorted(set(transform_labels))]
		# Add color legend and format figure
		axs[0].legend(lines, labels, loc='lower right', prop={'size': 5})
		axs[0].tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

	# Histogram
	if add_histogram:
		bins = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=32)
		values = np.unique(transform_labels)
		for value in values:
			hist_points = points[np.where(transform_labels == value)]
			color = cmap((value - np.min(values))/(np.max(values) - np.min(values)))
			axs[1].hist(hist_points[:, 0], label=label_mapping[value], alpha=0.5, density=True, bins=bins, color=color)
			axs[1].legend(loc='lower right', prop={'size': 5})
			axs[1].tick_params(labelleft=False, left=False)

	# Arrange plots so that they don't squeeze into each other
	plt.tight_layout(pad=5)
	plt.savefig(plot_outfile, dpi=300)
	return fig, axs