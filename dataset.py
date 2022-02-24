import os.path

from Bio import SeqIO
import numpy as np
from torch.utils.data import IterableDataset, DataLoader



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

class FaExampleIterator:
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
		self.example_num = 0

	def __next__(self):
		seq_str = bytes(next(self.seqio_iter).seq).decode('utf-8')
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
		self.example_num += 1
		return res

	def __iter__(self):
		return self

class FaDataset(IterableDataset):
	"""Iterable dataset that yields sequences and 1-hot encodings.
	Examples alternate between positive and negative.
	Returns instances like:
	   {
	   		"seq_str": "atGGaCTA[...]",
	   		"seq_arr": torch.float64 tensor of shape (len(seq_str), 4),
	   		"label_str": "pos",
	   		"label": 1,
	   		"part": "TRAIN",
	   		"fa_file": path to fa file,
	   		"example_num": index of this example in fa file
	   	}
	"""
	def __init__(self, part=None):
		fa_iterators = [
			LoopingIterable(FaExampleIterator, label=label, part=part)
			for label in ['pos', 'neg']
		]
		self.it = roundrobin(*fa_iterators)

	def __iter__(self):
		return self.it

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

loader = DataLoader(FaDataset(part='val'), batch_size=10)
import itertools
for thing in itertools.islice(loader, 408*2 + 2):
	print(thing)