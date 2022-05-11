import numpy as np
from Bio import SeqIO
import pybedtools

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

    def __init__(self, genome_file: str, bed_file: str, endless: bool=False, bedfile_columns=None):
        self.genome_file = genome_file
        self.bed_file = bed_file
        self.endless = endless
        self.bedfile_columns = bedfile_columns
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
        return len(self.intervals)

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
        seq_gen = (self._onehot(seq) for seq in SeqIO.parse(self.intervals.seqfn, "fasta"))
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
                    yield tuple(convert(interval.fields[i]) for i in self.bedfile_columns)
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

    def __init__(self, fa_file: str, endless: bool=False):
        self.fa_file = fa_file
        self.endless = endless
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
        self.fa_gen = SeqIO.parse(self.fa_file, "fasta")

    def _onehot(self, seq):
        res = np.zeros(self.seq_shape, dtype='int8')
        for idx, base in enumerate(seq.seq):
            if base in self.base_mapping:
                res[idx, self.base_mapping[base]] = 1
        return res

class SequenceCollection:

    def __init__(self, source_files, targets, targets_are_classes: bool, endless: bool=True):
        if len(source_files) != len(targets):
            raise ValueError("Number of source_files and number of targets must be equal")

        self.source_files = source_files
        self.targets = targets
        self.targets_are_classes = targets_are_classes
        self.endless = endless
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
                source_obj = FastaSource(source, endless=self.endless)
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
                    bedfile_columns=bedfile_columns)
            else:
                raise ValueError(f"Invalid source specification: {source}")
            sources.append(source_obj)
        return sources

    def _get_classes(self):
        if self.targets_are_classes:
            unique_classes = set()
            for source, target_spec in zip(self.source_files, self.targets):
                if isinstance(target_spec, dict):
                    # scan the target column from the corresponding source
                    source_obj = BedSource(source['genome'], source['intervals'],
                        endless=False, bedfile_columns=(target_spec['column'],))
                    for _, target_val in source_obj:
                        # extract value from singleton tuple
                        target_val = target_val[0]
                        unique_classes.add(target_val)
                else:
                    unique_classes.add(target_spec)
            self.idx_to_class_mapping = {idx: v for idx, v in enumerate(sorted(unique_classes))}
            self.class_to_idx_mapping = {v: k for k, v in self.idx_to_class_mapping.items()}
            self.num_classes = len(unique_classes)
        else:
            # targets are regression values
            self.idx_to_class_mapping = None
            self.class_to_idx_mapping = None
            self.num_classes = None

    def _get_source_freqs(self):
        """ Make a tree of counts and frequencies for each class and each of its data sources, e.g.


        {'source_freqs': array([0.25, 0.75]),
         'source_lens': [1000, 3000],
         'sources': [   <FastaSource object at 0x7f137428c4a8>,
                        <BedSource object at 0x7f137428cd30>],
         'total_len': 4000},
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

        if self.targets_are_classes:
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
    """Collection of FASTA sources and labels that allows sampling.
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
        fa_files (list of str): paths to FASTA files.
        labels (list of int): labels to assign to each file.
            If there are duplicate labels, then corresponding FASTA files are sampled
            as if they are in the same class.
        endless (bool): 
            if False, then yield each example from each file exactly once (useful for validation)
            if True, then randomly yield examples according to Sampling Logic (useful for training)

    Attributes:
        sc (SequenceCollection): streaming collection of sequences & targets.
        ds (tf.data.Dataset): same collection, as a tf Dataset.
        dataset (tf.data.Dataset or tuple(np.ndarray)): data to pass to keras fit().
            If endless is True, this is a tf Dataset yielding batches:
                xs (batch_size, seq_len, 4)
                ys (batch_size,) (sparse labels in {0, ..., num_classes - 1})
            If endless is False, this is a tuple of numpy arrays:
                xs (num_sequences, seq_len, 4)
                ys (num_sequences,) (sparse labels in {0, ..., num_classes - 1})

    E.g.:
    paths = ["/data/train_pos_A.fa", "/data/train_pos_B.fa", "/data/train_neg.fa"]
    ftd = FastaTfDataset(paths, [1, 1, 0])
    """
    def __init__(self, source_files, targets, targets_are_classes: bool,
                    endless: bool=True, batch_size: int=512):
        import tensorflow as tf
        self.sc = SequenceCollection(source_files, targets, targets_are_classes, endless=endless)
        self.targets_are_classes = targets_are_classes
        self.class_to_idx_mapping = self.sc.class_to_idx_mapping
        self.idx_to_class_mapping = self.sc.idx_to_class_mapping
        self.seq_shape = self.sc.seq_shape
        self.num_classes = self.sc.num_classes
        self.ds = tf.data.Dataset.from_generator(self.sc,
            output_types=(tf.int8, tf.int8),
            output_shapes=(tf.TensorShape(self.seq_shape), tf.TensorShape(())))
        self.batch_size = batch_size
        self.dataset = self._get_dataset(endless)

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
        dataset = self.ds.shuffle(len(self)).take(size)
        xs, ys = [], []
        for (x, y) in dataset.as_numpy_iterator():
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def _get_dataset(self, endless):
        if endless:
            return self.ds.batch(self.batch_size)
        else:
            return self.get_subset_as_arrays(len(self))

    def __len__(self):
        return len(self.sc)