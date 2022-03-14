import os.path

from Bio import SeqIO
import numpy as np
from torch.utils.data import IterableDataset, ChainDataset



DATA_DIR = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/"

def get_fa_file(label, part):
	label = label.lower()
	part = part.upper()
	if label not in ["pos", "neg"]:
		raise ValueError("bad label")
	if part not in ["TRAIN", "VAL", "TEST"]:
		raise ValueError("bad part")

	fname = f"mouse_SST_{label}_{part}.fa"
	return os.path.join(DATA_DIR, fname)

class FaExampleIterator(IterableDataset):
	base_mapping = {
		'A': np.array([1, 0, 0, 0]),
		'C': np.array([0, 1, 0, 0]),
		'G': np.array([0, 0, 1, 0]),
		'T': np.array([0, 0, 0, 1])
	}
	label_mapping = {
		'neg': 0,
		'pos': 1
	}

	def __init__(self, label, part, random_skip_range:int=None):
		self.fa_file = get_fa_file(label, part)
		self.seqio_iter = SeqIO.parse(self.fa_file, "fasta")
		self.label = label.lower()
		self.part = part.upper()
		self.example_num = -1
		self.len = sum(1 for seq in SeqIO.parse(self.fa_file, "fasta") if not self._is_malformed(seq))
		# if random_skip_range is passed, then
		#     skip [0, ..., random_skip_range - 1]-many sequences on each iteration
		# otherwise, don't skip sequences
		self.random_skip_range = random_skip_range or 1

	def __next__(self):
		seq = None
		skip = 0
		if self.random_skip_range > 1:
			skip = np.random.randint(self.random_skip_range)
		while ((seq is None) or self._is_malformed(seq) or (skip > 0)):
			seq = next(self.seqio_iter)
			self.example_num += 1
			skip -= 1
		seq_str = bytes(seq.seq).decode('utf-8')
		seq_arr = np.zeros((len(seq_str), len(self.base_mapping)))
		for i, base in enumerate(seq_str.upper()):
			seq_arr[i] = self.base_mapping[base]
		res = {
			"seq_str": seq_str,
			"seq_arr": seq_arr,
			"label_str": self.label,
			"label": self.label_mapping[self.label],
			"part": self.part,
			"fa_file": self.fa_file,
			"example_num": self.example_num
		}
		#TODO is there a way to pass all the metadata and only take the tensors later?
		return res["seq_arr"], res["label"]

	def __iter__(self):
		return self

	def __len__(self):
		return self.len

	def _is_malformed(self, seq: SeqIO.SeqRecord):
		return 'N' in seq.upper()

class FaDatasetSampler(IterableDataset):
	def __init__(self, part=None, random_skip_range:int=None, epoch_len:int=None):
		it_args = [
			(FaExampleIterator, {'label': label, 'part': part, 'random_skip_range': random_skip_range})
			for label in ['pos', 'neg']
		]
		self.it_args = it_args
		self.iterators = [it_class(**kwargs) for it_class, kwargs in it_args]
		class_counts = np.array([len(it) for it in self.iterators])
		self.class_p = class_counts/class_counts.sum()
		self.epoch_len = epoch_len or class_counts.sum()
		self.num_classes = len(it_args)

	def __iter__(self):
		while True:
			it_idx = np.random.choice(self.num_classes, p=self.class_p)
			try:
				val = next(self.iterators[it_idx])
			except StopIteration:
				it_class, kwargs = self.it_args[it_idx]
				self.iterators[it_idx] = it_class(**kwargs)
				val = next(self.iterators[it_idx])
			yield val

	def __len__(self):
		return self.epoch_len

class FaDataset(IterableDataset):
	"""Iterable dataset of sequences (1-hot tensor) and labels (int).
	Examples alternate between positive and negative with equal frequency.
	Useful for training.

	Args:
	    part (str): in ['train', 'val', 'test']
	"""
	def __init__(self, part=None, random_skip_range:int=None):
		it_args = [
			(FaExampleIterator, {'label': label, 'part': part, 'random_skip_range': random_skip_range})
			for label in ['pos', 'neg']
		]
		self.epoch_len = max(len(it(**kwargs)) for it, kwargs in it_args) * len(it_args)
		self.it = roundrobin_batcher(it_args)

	def __iter__(self):
		return self.it

	def __len__(self):
		return self.epoch_len

def roundrobin_batcher(it_args, endless=True):
	"""Yield elements from each iterator until they have all been exhausted.
	If iterators exhaust at different times, they will keep yielding elements
	in a loop until all are exhausted, such that all iterators yield the same
	number of elements in total.

	E.g. if itA yields 'ABCDE' and itB yields 'XYZ',
	then roundrobin_batcher(itA, itB) yields 'AXBYCZDXEY'.

	Args:
	    it_args: list of tuple (IterableDataset subclass, dict of kwargs)
	    endless (bool): if True, then always refresh iterators without ending
	                    if False, then end once all iterators have been exhausted
	"""
	num_unfinished = len(it_args)
	iterators = [
		{
			"it_class": it_class,
			"kwargs": kwargs,
			"iterator": it_class(**kwargs),
			"finished": False
		}
		for it_class, kwargs in it_args
	]
	while True:
		batch = []
		for rep in iterators:
			try:
				batch.append(next(rep["iterator"]))
			except StopIteration:
				if not rep["finished"]:
					if not endless:
						num_unfinished -= 1

				rep["finished"] = True
				rep["iterator"] = rep["it_class"](**rep["kwargs"])
				batch.append(next(rep["iterator"]))
		if num_unfinished:
			for thing in batch:
				yield thing
		else:
			return

class FiniteIterator:
	def __init__(self, elements=None):
		self.it = iter(elements)

	def __next__(self):
		return next(self.it)

def roundrobin_test():
	it_args = [(FiniteIterator, {'elements': 'ABCDE'}), (FiniteIterator, {'elements': 'XYZ'})]
	res = [c for c in roundrobin_batcher(it_args, endless=False)]
	assert(res == ['A', 'X', 'B', 'Y', 'C', 'Z', 'D', 'X', 'E', 'Y'])

	from itertools import islice
	res = [c for c in islice(roundrobin_batcher(it_args, endless=True), 12)]
	assert(res == ['A', 'X', 'B', 'Y', 'C', 'Z', 'D', 'X', 'E', 'Y', 'A', 'Z'])

import itertools
class SinglePassDataset(IterableDataset):
	"""Iterable dataset where each example appears exactly once in each epoch.
	Useful for validation.

	Args:
		part (str): in ['train', 'val', 'test']
	"""
	def __init__(self, part, endless=True):
		self.part = part
		self._refresh_dataset()
		self.len = len(self.dataset)
		self.endless = endless

	def __len__(self):
		return self.len

	def __iter__(self):
		while True:
			for x in self.dataset:
				yield x
			if self.endless:
				self._refresh_dataset()
			else:
				break

	def _refresh_dataset(self):
		self.dataset = ChainDataset([FaExampleIterator(label, self.part) for label in ['pos', 'neg']])
