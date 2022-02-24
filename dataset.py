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

class FaDataset(IterableDataset):
	"""Iterable dataset that yields sequences and 1-hot encodings.
	Returns instances like:
	   {"seq_str": "atGGaCTA", "seq_arr": torch.float64 tensor of shape (8, 4)}
	"""
	def __init__(self, label=None, part=None):
		fa_path = get_fa_file(label, part)
		self.fa_it = self.FaIterator(SeqIO.parse(fa_path, "fasta"), label)

	class FaIterator:
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

		def __init__(self, seqio_iter, label):
			self.seqio_iter = seqio_iter
			self.label = label

		def __next__(self):
			seq_str = bytes(next(self.seqio_iter).seq).decode('utf-8')
			seq_arr = np.zeros((len(seq_str), len(self.base_mapping)))
			for i, base in enumerate(seq_str.upper()):
				seq_arr[i] = self.base_mapping[base]
			return {
				"seq_str": seq_str,
				"seq_arr": seq_arr,
				"label_str": self.label,
				"label": self.label_mapping[self.label]
			}

	def __iter__(self):
		return self.fa_it

loader = DataLoader(FaDataset(label='pos', part='train'), batch_size=8)
import itertools
for thing in itertools.islice(loader, 1):
	print(thing)