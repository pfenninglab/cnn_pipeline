import os.path

from Bio import SeqIO
import numpy as np
from torch.utils.data import IterableDataset, ChainDataset



DATA_DIR = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/"

#TODO revert to 10, make a class instantiation argument
RANDOM_SKIP_RANGE = 1

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

	def __init__(self, label, part):
		self.fa_file = get_fa_file(label, part)
		self.seqio_iter = SeqIO.parse(self.fa_file, "fasta")
		self.label = label.lower()
		self.part = part.upper()
		self.example_num = -1
		self.len = sum(1 for _ in SeqIO.parse(self.fa_file, "fasta"))

	def __next__(self):
		seq = None
		skip = np.random.randint(RANDOM_SKIP_RANGE)
		while ((seq is None) or ('N' in seq.upper()) or (skip > 0)):
			# skip a random number of sequences
			# (if not, then sequences are presented in a deterministic order)
			# also skip sequences with 'N' base
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

class FaDataset(IterableDataset):
	"""Iterable dataset that yields sequences and 1-hot encodings.
	Examples alternate between positive and negative.

	Args:
	    part (str): in ['train', 'val', 'test']
	"""
	def __init__(self, part=None):
		it_args = [
			(FaExampleIterator, {'label': label, 'part': part})
			for label in ['pos', 'neg']
		]
		self.epoch_len = max(len(it(**kwargs)) for it, kwargs in it_args) * len(it_args)
		self.it = roundrobin_batcher(it_args)

	def __iter__(self):
		return self.it

	def __len__(self):
		return self.epoch_len

class LoopingIterable:
	def __init__(self, it_class, **kwargs):
		self.it_class = it_class
		self.kwargs = kwargs
		self.it = self.it_class(**self.kwargs)

	def __next__(self):
		try:
			return next(self.it)
		except StopIteration:
			self.it = self.it_class(**self.kwargs)
			return next(self.it)

	def __iter__(self):
		return self

from itertools import cycle, islice
# from https://docs.python.org/3/library/itertools.html#recipes
def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

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

def get_singlepass_dataset(part):
	"""Returns an IterableDataset where each example is given exactly once."""
	return ChainDataset([FaExampleIterator(label, part) for label in ['pos', 'neg']])