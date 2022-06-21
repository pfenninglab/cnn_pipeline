from collections import Counter

import numpy as np
from Bio import SeqIO
import pybedtools
from tqdm import tqdm

import constants

# A, C, G, T
NUM_BASES = 4

# random seed for reproducibility
SEED = 0
rng = np.random.default_rng(SEED)


class BedSource:
    """Iterator of sequences from a .bed or .narrowPeaks file and corresponding reference genome .fa file.
    Can reload itself once exhausted.

    Args:
        genome_file (str): path to whole-genome reference FASTA file.
        bed_file (str): path to .bed or .narrowPeaks file with intervals.
        endlesss (bool): if True, then restart iterator once exhausted.
    """
    base_mapping = {'A':0, 'a':0,
                    'C':1, 'c':1,
                    'G':2, 'g':2,
                    'T':3, 't':3}

    def __init__(self, genome_file: str, bed_file: str, endless: bool=False, bedfile_columns=None, reverse_complement: bool=False):
        self.genome_file = genome_file
        self.bed_file = bed_file
        self.endless = endless
        self.bedfile_columns = bedfile_columns
        self.reverse_complement = reverse_complement
        self.intervals = self.get_intervals(self.bed_file, self.genome_file)
        self.len = self._get_len()
        self.seq_len = self._get_seq_len()
        self._load_gen()
        self.seq_shape = (self.seq_len, NUM_BASES)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.gen)
        except StopIteration as e:
            if self.endless:
                self._load_gen()
                return next(self.gen)
            else:
                raise e

    def __len__(self):
        return self.len

    def _get_len(self):
        length = len(self.intervals)
        if self.reverse_complement:
            length *= 2
        return length

    def _get_seq_len(self):
        seq_len = None
        for interval in self.intervals:
            seq_len = seq_len or len(interval)
            if len(interval) != seq_len:
                raise ValueError(f"BED file contains sequences of different lengths! Found {seq_len} and {len(interval)}")
            if seq_len < 1:
                raise ValueError(f"Empty sequence in BED file: {self.bed_file}")
        return seq_len

    def _load_gen(self):
        def seq_gen():
            for seq in SeqIO.parse(self.intervals.seqfn, "fasta"):
                yield self._onehot(seq)
                if self.reverse_complement:
                    yield self._onehot(seq.reverse_complement())
        seq_gen = seq_gen()

        if not self.bedfile_columns:
            # Only yield sequences
            self.gen = seq_gen
        else:
            # Yield sequences and column values

            def convert(x):
                """Attempt to convert column value to number"""
                if x == ".":
                    return None
                try:
                    val = int(x)
                except ValueError:
                    try:
                        val = float(x)
                    except ValueError:
                        val = x
                return val

            def column_gen():
                """Yield tuples of selected columns"""
                for interval in self.intervals:
                    data = tuple(convert(interval.fields[i]) for i in self.bedfile_columns)
                    yield data
                    if self.reverse_complement:
                        yield data

            column_gen = column_gen()

            self.gen = zip(seq_gen, column_gen)

    def _onehot(self, seq):
        res = np.zeros(self.seq_shape, dtype='int8')
        for idx, base in enumerate(seq):
            if base in self.base_mapping:
                res[idx, self.base_mapping[base]] = 1
        return res

    @staticmethod
    def get_intervals(bed_file, genome_file=None):
        """Get pybedtools.BedTool object from .bed or .narrowPeak file"""
        with open(bed_file, "r") as f:
            intervals = pybedtools.BedTool(f.read(), from_string=True)
        if genome_file is not None:
            intervals = intervals.sequence(fi=genome_file)
        return intervals

    @staticmethod
    def get_interval_seq(chrom, start, stop, genome_file):
        # .bed file uses 0-based, [start, stop) numbering
        # BedTool.seq() assumes 1-based, inclusive numbering
        # so need to add 1 to start, and keep stop the same
        loc = f"{chrom}:{start + 1}-{stop}"
        return pybedtools.BedTool.seq(loc, genome_file)

class FastaSource:
    """Iterator of sequences from a FASTA file.
    Can reload itself once exhausted.

    Args:
        fa_file (str): FASTA file to read lines from.
        endlesss (bool): if True, then restart iterator once exhausted.
    """
    base_mapping = {'A':0, 'a':0,
                    'C':1, 'c':1,
                    'G':2, 'g':2,
                    'T':3, 't':3}

    def __init__(self, fa_file: str, endless: bool=False, reverse_complement: bool=False):
        self.fa_file = fa_file
        self.endless = endless
        self.reverse_complement = reverse_complement
        self.len = self._get_len()
        self.seq_len = self._get_seq_len()
        self._load_gen()
        self.seq_shape = (self.seq_len, NUM_BASES)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._onehot(next(self.fa_gen))
        except StopIteration as e:
            if self.endless:
                self._load_gen()
                return self._onehot(next(self.fa_gen))
            else:
                raise e

    def __len__(self):
        return self.len

    def _get_len(self):
        fa_len = sum(1 for _ in SeqIO.parse(self.fa_file, "fasta"))
        if fa_len < 1:
            raise ValueError("No sequences in FASTA file: {self.fa_file}")
        if self.reverse_complement:
            fa_len *= 2
        return fa_len

    def _get_seq_len(self):
        seq_len = None
        for seq in SeqIO.parse(self.fa_file, "fasta"):
            seq_len = seq_len or len(seq)
            if len(seq) != seq_len:
                raise ValueError("FASTA file contains sequences of different lengths! Found {seq_len} and {len(seq)}")
            if seq_len < 1:
                raise ValueError("Empty sequence in FASTA file: {self.fa_file}")
        return seq_len

    def _load_gen(self):
        def gen():
            for seq in SeqIO.parse(self.fa_file, "fasta"):
                yield seq
                if self.reverse_complement:
                    yield seq.reverse_complement()
        self.fa_gen = gen()

    def _onehot(self, seq):
        res = np.zeros(self.seq_shape, dtype='int8')
        for idx, base in enumerate(seq.seq):
            if base in self.base_mapping:
                res[idx, self.base_mapping[base]] = 1
        return res

class SequenceCollection:
    """Iterable collection of sequences from FASTA, BED, or NarrowPeak files.
    Can reload itself once exhausted, to function as an infinite iterator.

    See SequenceTfDataset for a description of args.
    """

    def __init__(self, source_files, targets, targets_are_classes: bool, endless: bool=True,
        map_targets: bool=True, reverse_complement: bool=False):
        if len(source_files) != len(targets):
            raise ValueError("Number of source_files and number of targets must be equal")

        self.source_files = source_files
        self.targets = targets
        self.targets_are_classes = targets_are_classes
        self.map_targets = map_targets
        self.endless = endless
        self.reverse_complement = reverse_complement
        self.sources = self._get_sources()
        self.num_sources = len(self.sources)
        self.seq_shape = self._get_seq_shape()
        self.source_freqs = self._get_source_freqs()
        self._get_classes()
        self.len = self.source_freqs['total_len']

    def _get_sources(self):
        sources = []
        for source, target_spec in zip(self.source_files, self.targets):
            if isinstance(source, str):
                # path to FASTA file of sequences
                source_obj = FastaSource(source, endless=self.endless, reverse_complement=self.reverse_complement)
            elif isinstance(source, dict):
                # genome FA file and interval BED file
                for key in ['genome', 'intervals']:
                    if key not in source:
                        raise ValueError(f"Missing expected key {key} in source specification {source}")
                bedfile_columns = None
                if isinstance(target_spec, dict):
                    if 'column' not in target_spec:
                        raise ValueError(f"Missing `column` in target specification {target_spec}")
                    bedfile_columns = (target_spec['column'],)
                source_obj = BedSource(
                    source['genome'], source['intervals'], endless=self.endless,
                    bedfile_columns=bedfile_columns, reverse_complement=self.reverse_complement)
            else:
                raise ValueError(f"Invalid source specification: {source}")
            sources.append(source_obj)
        return sources

    def _get_classes(self):
        unique_values = self._get_unique_values()

        if self.targets_are_classes:
            # Check that classes are all the same type
            class_types = set()
            for label in unique_values.keys():
                # Only allow int and str class values
                if not any(isinstance(label, t) for t in [int, str]):
                    raise ValueError(f"Invalid type for classification target, value {label}, type {type(label)}")
                class_types.add(type(label))
            if len(class_types) > 1:
                raise NotImplementedError(f"Class values must all be the same type. Found types: {class_types}")

            self.idx_to_class_mapping = {idx: v for idx, v in enumerate(sorted(unique_values.keys()))}
            self.class_to_idx_mapping = {v: k for k, v in self.idx_to_class_mapping.items()}
            self.num_classes = len(unique_values)

            class_counts = dict()
            for k, v in unique_values.items():
                if self.map_targets:
                    class_counts[self.class_to_idx_mapping[k]] = v
                else:
                    class_counts[k] = v
            self.class_counts = class_counts
        else:
            # targets are regression values

            # Only allow int and float regression targets
            print("Checking target values...")
            for value in tqdm(unique_values.keys()):
                if not any(isinstance(value, t) for t in [int, float]):
                    raise ValueError(f"Invalid type for regression target, value {value}, type {type(value)}")

            self.idx_to_class_mapping = None
            self.class_to_idx_mapping = None
            self.num_classes = None
            self.class_counts = None

    def _get_unique_values(self):
        unique_values = Counter()
        for source, source_file, target_spec in zip(self.sources, self.source_files, self.targets):
            if isinstance(target_spec, dict):
                # Get a new source object, so we can scan it without exhausting the original
                source_obj = BedSource(source_file['genome'], source_file['intervals'],
                    endless=False, bedfile_columns=(target_spec['column'],))
                # Scan the target column from the corresponding source
                print(f"Getting target values from {source_file['intervals']}...")
                for _, target_val in tqdm(source_obj):
                    # extract value from singleton tuple
                    target_val = target_val[0]
                    unique_values[target_val] += 1
            else:
                target_val = target_spec
                unique_values[target_val] += len(source)
        return unique_values

    def _get_source_freqs(self):
        """Get counts and frequencies for each data source, e.g.

        {'source_lens': [378382, 61628],
         'total_len': 440010,
         'source_freqs': array([0.85993955, 0.14006045])}
        """

        freqs = {'source_lens': [len(source) for source in self.sources]}
        freqs['total_len'] = sum(freqs['source_lens'])
        freqs['source_freqs'] = np.array(freqs['source_lens']) / freqs['total_len']

        return freqs

    def _get_seq_shape(self):
        shape = None
        for source in self.sources:
            shape = shape or source.seq_shape
            if source.seq_shape != shape:
                raise ValueError("Sources have inconsistent shapes, found {shape} and {source.seq_shape}")
        return shape

    def _get_example(self, data, target_spec):
        if isinstance(target_spec, dict):
            seq, target_val = data
            target_val = target_val[0]
        else:
            seq = data
            target_val = target_spec

        if self.targets_are_classes and self.map_targets:
            target_val = self.class_to_idx_mapping[target_val]

        return seq, target_val

    def __iter__(self):
        if self.endless:
            while True:
                source_idx = rng.choice(self.num_sources, p=self.source_freqs['source_freqs'])
                source, target_spec = self.sources[source_idx], self.targets[source_idx]
                data = next(source)
                seq, target_val = self._get_example(data, target_spec)
                yield seq, target_val
        else:
            for source, target_spec in zip(self.sources, self.targets):
                for data in source:
                    seq, target_val = self._get_example(data, target_spec)
                    yield seq, target_val

    def __call__(self):
        return self

    def __len__(self):
        return self.len

class FastaCollection:
    """DEPRECATED: Use SequenceCollection instead.

    Collection of FASTA sources and labels that allows sampling.
    Multiple FASTA files can be combined to the same class.

    Assumes: Each FA file has examples with all the same label.

    Sampling Logic:
        Let C_1, ..., C_n be the n classes.
        For each class C_i, let C_i_1, ..., C_i_k be the k sources of class C_i.

        Then for each sample,
            P(C_i) = |C_i| / sum(|C_1| + ... + |C_n|)
            P(C_i_j | C_i) = |C_i_j| / sum(|C_i_1| + ... + |C_i_k|)

    Args:
        fa_files (list of str): paths to FASTA files.
        labels (list of int): labels to assign to each file.
            If there are duplicate labels, then corresponding FASTA files are sampled
            as if they are in the same class.
        endless (bool): 
            if False, then yield each example from each file exactly once (useful for validation)
            if True, then randomly yield examples according to Sampling Logic (useful for training)

    E.g.:
    paths = ["/data/train_pos_A.fa", "/data/train_pos_B.fa", "/data/train_neg.fa"]
    fc = FastaCollection(paths, [1, 1, 0])

    NOTE This looping-and-sampling strategy works, but we could get the same behavior
    using tf.data.Dataset functions repeat() and sample_from_datasets(). Consider
    switching to that.
    """
    def __init__(self, fa_files, labels, endless: bool=True):
        if len(fa_files) != len(labels):
            raise ValueError("Number of fa_files and number of labels must be equal")

        self.fa_files = fa_files
        self.labels = labels
        self.endless = endless
        self.fa_sources = [FastaSource(fa_file, endless=endless) for fa_file in self.fa_files]
        self.seq_shape = self._get_seq_shape()
        self._make_frequency_tree()
        self.num_classes = len(self.class_freqs['classes'])
        self.len = self.class_freqs['total_len']

    def _make_frequency_tree(self):
        """ Make a tree of counts and frequencies for each class and each of its data sources, e.g.

        {   'class_freqs': array([0.66666, 0.33334]),
            'class_lens': [4, 2],
            'total_len': 6,
            'classes': {   0: {   'source_freqs': array([0.25, 0.75]),
                                  'source_lens': [1, 3],
                                  'sources': [   <FastaSource object at 0x7f137428c4a8>,
                                                 <FastaSource object at 0x7f137428cd30>],
                                  'len': 4},
                           1: {   'source_freqs': array([1.]),
                                  'source_lens': [2],
                                  'sources': [   <FastaSource object at 0x7f137428ccf8>],
                                  'len': 2}},
            'labels': [0, 1]}
        """
        freqs = {'classes': dict(), 'labels': [], 'class_lens': []}

        # get sub-class counts
        for source, label in zip(self.fa_sources, self.labels):
            if label not in freqs['classes']:
                freqs['classes'][label] = {'len': 0, 'sources': [], 'source_lens': []}

            freqs['classes'][label]['len'] += len(source)
            freqs['classes'][label]['sources'].append(source)
            freqs['classes'][label]['source_lens'].append(len(source))

        # get class counts
        for cl, cl_data in freqs['classes'].items():
            freqs['labels'].append(cl)
            freqs['class_lens'].append(cl_data['len'])
        freqs['total_len'] = sum(freqs['class_lens'])

        # normalize sub-class counts
        for _, cl_data in freqs['classes'].items():
            cl_data['source_freqs'] = np.array(cl_data['source_lens']) / cl_data['len']

        # normalize class counts
        freqs['class_freqs'] = np.array(freqs['class_lens']) / freqs['total_len']

        self.class_freqs = freqs

    def _get_seq_shape(self):
        shape = None
        for source in self.fa_sources:
            shape = shape or source.seq_shape
            if source.seq_shape != shape:
                raise ValueError("FASTA sources have inconsistent shapes, found {shape} and {source.seq_shape}")
        return shape

    def __iter__(self):
        if self.endless:
            while True:
                # Draw a class, proportional to all the classes in the dataset.
                label = rng.choice(self.class_freqs['labels'],
                    p=self.class_freqs['class_freqs'])
                # Draw a source, proportional to all the sources in that class.
                source = rng.choice(self.class_freqs['classes'][label]['sources'],
                    p=self.class_freqs['classes'][label]['source_freqs'])
                yield next(source), label
        else:
            for source, label in zip(self.fa_sources, self.labels):
                for seq in source:
                    yield seq, label

    def __call__(self):
        return self

    def __len__(self):
        return self.len


class SequenceTfDataset:
    """Sequence collection with a corresponding tf.data.Dataset.

    Args:
        source files (list of str or dict): Source files to get sequence data.
            Each source can be either:
                str, path to .fa file with sequences already extracted, or
                dict, with keys
                    "genome": path to .fa file of reference genome
                    "intervals": path to a .bed or .narrowPeaks file with intervals to extract from reference
        targets (list of str or dict): Targets to associate with data from each source.
            Each target can be either:
                int or str, a fixed value to associate with every example from the corresponding source
                dict, with key
                    "column": column to extract from each row of .bed or .narrowPeak file
        targets_are_classes (bool): Whether to treat targets as classes in a classification problem (True),
            or continuous values in a single-variable regression problem (False)
            if True, then the target outputs are int sparse labels in {0, ..., num_classes - 1}
            if False, then the target outputs are float regression targets
        endless (bool): 
            if False, then dataset is a tuple of fixed numpy arrays. Each example appears
                exactly once. Useful for validation.
            if True, then dataset is an infinite iterator. Examples are randomly sampled from
                sources, such that the expected number of times an example appears in each
                epoch is 1. Useful for training.
        batch_size (int): Batch size to yield when dataset is an iterator. Only has effect when
            endless == True.
        map_targets (bool):
            if True, then map target values to their class index before yielding.
            if False, then yield target values directly.
            For example, if you create a dataset where the only label is 1, then
                map_targets == True => yielded value is 0 (because 1 is the 0-th class)
                map-targets == False => yielded value is 1
        reverse_complement (bool): if True, then add the reverse complement of each sequence.

    Sampling Logic: When endless == True, each example is randomly sampled from the set of
    data sources, proportionally to the size of each source. That is, if we have:
        - N sources, S_1, ..., S_N
        - source S_i has k examples |S_i| = k
        - the full dataset has j examples |S_1| + ... + |S_N| = j
        - the batch size is b
    Then the expected number of examples from source S_i in a given batch is k / j * b.
    The expected number of examples from source S_i across all batches in an epoch is k / j.

    Attributes:
        sc (SequenceCollection): Streaming collection of sequences & targets.
        ds (tf.data.Dataset): Same collection, as a tf Dataset.
        dataset (tf.data.Dataset or tuple(np.ndarray)): Data to pass to keras fit().
            If endless is True, this is a tf Dataset yielding batches:
                xs (batch_size, seq_len, 4)
                ys (batch_size,)
            If endless is False, this is a tuple of numpy arrays:
                xs (num_sequences, seq_len, 4)
                ys (num_sequences,)
        class_to_idx_mapping (dict): Maps class labels to the integer class output by the model.
            Applicable only when targets_are_classes == True.
            e.g. {"neg": 0, "pos": 1} or {"chr1": 0, "chr2": 1, "chrX": 2}
        idx_to_class_mapping (dict): Maps integer classes to class labels.
            Applicable only when targets_are_classes = True.
            e.g. {0: "neg", 1: "pos"} or {0: "chr1", 1: "chr2", 2: "chrX"}
        seq_shape (tuple of int): Dimensions of each example's input features, e.g. (500, 4)
        num_classes (int): If targets_are_classes == True, the number of classes.
            If targets_are_classes == False, None.

    E.g.:
    paths = [
        "/data/train_pos_A.fa",
        {"genome": "/data/Mus_musculus.fa", "intervals": "/data/peaks_pos.narrowPeak"},
        {"genome": "/data/Mus_musculus.fa", "intervals": "/data/peaks_neg.narrowPeak"}
    ]
    train_data = SequenceTfDataset(paths, [1, 1, 0], True)
    """
    def __init__(self, source_files, targets, targets_are_classes: bool,
                    endless: bool=True, batch_size: int=constants.DEFAULT_BATCH_SIZE,
                    map_targets: bool=True, reverse_complement: bool=False):
        import tensorflow as tf
        self.sc = SequenceCollection(source_files, targets, targets_are_classes, endless=endless,
            map_targets=map_targets, reverse_complement=reverse_complement)
        self.targets_are_classes = targets_are_classes
        self.class_to_idx_mapping = self.sc.class_to_idx_mapping
        self.idx_to_class_mapping = self.sc.idx_to_class_mapping
        self.seq_shape = self.sc.seq_shape
        self.num_classes = self.sc.num_classes
        target_type = tf.int8 if targets_are_classes else tf.float32
        self.ds = tf.data.Dataset.from_generator(self.sc,
            output_types=(tf.int8, target_type),
            output_shapes=(tf.TensorShape(self.seq_shape), tf.TensorShape(())))
        self.batch_size = batch_size
        self.endless = endless
        self.dataset = self._get_dataset(endless)
        self.class_counts = self.sc.class_counts

    def get_subset_as_arrays(self, size):
        """Return a random subset as 2 numpy arrays.

        NOTE this method is slow, depending on the size of the entire
        SequenceTfDataset: ~1 minute for a SequenceTfDataset with 200K
        items. This is because to randomly sample from a stream dataset,
        you need to read every element. Please only use this method when
        you need a fixed random subset of the dataset.

        Args:
            size (int): Number of examples in subset.

        Returns:
            xs (np.ndarray): [size, num_bp, 4], one-hot sequences
            ys (np.ndarray): [size, num_bp], labels
        """
        if size > len(self):
            raise ValueError(f"Requested subset size {size} is too large for dataset of size {len(self)}")

        dataset = self.ds
        if size < len(self):
            dataset = dataset.shuffle(len(self))
        dataset = dataset.take(size)

        xs, ys = [], []
        for (x, y) in dataset.as_numpy_iterator():
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def _get_dataset(self, endless):
        if endless:
            return self.ds.shuffle(self.batch_size * 16).batch(self.batch_size)
        else:
            return self.get_subset_as_arrays(len(self))

    def __len__(self):
        return len(self.sc)