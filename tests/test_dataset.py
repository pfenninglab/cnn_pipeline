# allow importing from one directory up
import sys
sys.path.append('..')

from itertools import islice

import numpy as np

from dataset import FastaSource, BedSource, SequenceCollection

def test_bedsource():
    # No bed columns

    fa_source = FastaSource("/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.fa",
        endless=True)
    bed_source = BedSource(
        "/projects/pfenninggroup/machineLearningForComputationalBiology/halLiftover_chains/data/raw_data/2bit/fasta/Mus_musculus.fa",
        "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.bed",
        endless=True)

    assert fa_source.len == bed_source.len
    assert fa_source.seq_len == bed_source.seq_len
    assert fa_source.seq_shape == bed_source.seq_shape

    for fa_seq, bed_seq in islice(zip(fa_source, bed_source), 10):
        assert np.all(fa_seq == bed_seq)
        assert bed_seq.shape == bed_source.seq_shape

    # exhaust and restart generator
    for _ in range(bed_source.len):
        next(fa_source)
        next(bed_source)
    for fa_seq, bed_seq in islice(zip(fa_source, bed_source), 10):
        assert np.all(fa_seq == bed_seq)
        assert bed_seq.shape == bed_source.seq_shape

    # Bed columns
    bed_source = BedSource(
        "/projects/pfenninggroup/machineLearningForComputationalBiology/halLiftover_chains/data/raw_data/2bit/fasta/Mus_musculus.fa",
        "../example_files/example.narrowPeak",
        endless=True,
        bedfile_columns=(0, 5, 6, 7))

    assert bed_source.len == 3
    assert bed_source.seq_len == 100
    assert bed_source.seq_shape == (100, 4)

    expected_bed_values = [
        ("chr1", None, 182, 5.0945),
        ("chr1", None, 91, 4.6052),
        ("chr2", None, 182, 9.2103)
    ]

    for _ in range(2):
        # First iteration: tests bed_source before refresh
        # Second iteration: tests bed_source after refresh
        for expected_values, (bed_seq, bed_values) in zip(expected_bed_values, bed_source):
            assert bed_seq.shape == bed_source.seq_shape
            assert bed_values == expected_values

    return bed_source

def test_sequence_collection():
    fa_path_pos = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.fa"
    fa_path_neg = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_neg_VAL.fa"
    genome_path = "/projects/pfenninggroup/machineLearningForComputationalBiology/halLiftover_chains/data/raw_data/2bit/fasta/Mus_musculus.fa"
    mouse_interval_path_pos = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.bed"
    mouse_interval_path_neg = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_neg_VAL.bed"
    narrowpeak_path = "../example_files/example.narrowPeak"

    #####################
    # Test that FASTA source and genome+interval source produce the same results
    seq_collection_A = SequenceCollection([fa_path_pos], [1], True, endless=True)
    seq_collection_B = SequenceCollection([{'genome': genome_path, 'intervals': mouse_interval_path_pos}], [1], True, endless=True)

    assert len(seq_collection_A) == len(seq_collection_B)
    assert seq_collection_A.num_classes == seq_collection_B.num_classes
    for (seq_A, label_A), (seq_B, label_B) in islice(zip(seq_collection_A, seq_collection_B), 10):
        assert np.all(seq_A == seq_B)
        assert label_A == label_B
    # exhaust the generators and loop
    for _ in islice(zip(seq_collection_A, seq_collection_B), len(seq_collection_A)):
        pass
    for (seq_A, label_A), (seq_B, label_B) in islice(zip(seq_collection_A, seq_collection_B), 10):
        assert np.all(seq_A == seq_B)
        assert label_A == label_B

    #####################
    # Test multiple files per source
    seq_collection_A = SequenceCollection([fa_path_neg, fa_path_pos], [0, 1], True, endless=True)
    seq_collection_B = SequenceCollection(
        [{'genome': genome_path, 'intervals': mouse_interval_path_neg},
        {'genome': genome_path, 'intervals': mouse_interval_path_pos}],
        [0, 1], True, endless=True)

    pos_A, pos_B, neg_A, neg_B = [], [], [], []
    # get examples from both collections. they might come in a different interleaved order, but
    # should be the same order once sorted into postive and negative.
    for (seq_A, label_A), (seq_B, label_B) in islice(zip(seq_collection_A, seq_collection_B), 100):
        if label_A == 1:
            pos_A.append(seq_A)
        else:
            neg_A.append(seq_A)
        if label_B == 1:
            pos_B.append(seq_B)
        else:
            neg_B.append(seq_B)
    for seq_A, seq_B in zip(pos_A, pos_B):
        assert np.all(seq_A == seq_B)
    for idx, (seq_A, seq_B) in enumerate(zip(neg_A, neg_B)):
        assert np.all(seq_A == seq_B)

    #####################
    # Test mixed file types per source
    seq_collection_A = SequenceCollection([fa_path_neg, fa_path_pos], [0, 1], True, endless=True)
    seq_collection_B = SequenceCollection(
        [fa_path_neg,
        {'genome': genome_path, 'intervals': mouse_interval_path_pos}],
        [0, 1], True, endless=True)

    pos_A, pos_B, neg_A, neg_B = [], [], [], []
    # get examples from both collections. they might come in a different interleaved order, but
    # should be the same order once sorted into postive and negative.
    for (seq_A, label_A), (seq_B, label_B) in islice(zip(seq_collection_A, seq_collection_B), 100):
        if label_A == 1:
            pos_A.append(seq_A)
        else:
            neg_A.append(seq_A)
        if label_B == 1:
            pos_B.append(seq_B)
        else:
            neg_B.append(seq_B)
    for seq_A, seq_B in zip(pos_A, pos_B):
        assert np.all(seq_A == seq_B)
    for idx, (seq_A, seq_B) in enumerate(zip(neg_A, neg_B)):
        assert np.all(seq_A == seq_B)

    #####################
    # Test regression target extracted from bed column
    for endless in [True, False]:
        seq_collection = SequenceCollection(
            [{'genome': genome_path, 'intervals': narrowpeak_path}],
            [{'column': 6}], targets_are_classes=False, endless=endless)
        assert seq_collection.idx_to_class_mapping == None
        assert seq_collection.class_to_idx_mapping == None
        assert seq_collection.num_classes == None
        expected_values = [182, 91, 182]
        target_values = [val for _, val in islice(seq_collection, 3)]
        assert expected_values == target_values, target_values

        seq_collection = SequenceCollection(
            [{'genome': genome_path, 'intervals': narrowpeak_path}],
            [{'column': 7}], targets_are_classes=False, endless=endless)
        assert seq_collection.idx_to_class_mapping == None
        assert seq_collection.class_to_idx_mapping == None
        assert seq_collection.num_classes == None
        expected_values = [5.0945, 4.6052, 9.2103]
        target_values = [val for _, val in islice(seq_collection, 3)]
        assert expected_values == target_values, target_values

    #####################
    # Test classification label extracted from bed column and mapped
    for endless in [True, False]:
        seq_collection = SequenceCollection(
            [{'genome': genome_path, 'intervals': narrowpeak_path}],
            [{'column': 6}], targets_are_classes=True, endless=endless)
        assert seq_collection.idx_to_class_mapping == {0: 91, 1: 182}, seq_collection.idx_to_class_mapping
        assert seq_collection.class_to_idx_mapping == {91: 0, 182: 1}, seq_collection.class_to_idx_mapping
        expected_values = [1, 0, 1]
        target_values = [val for _, val in islice(seq_collection, 3)]
        assert expected_values == target_values, target_values

        seq_collection = SequenceCollection(
            [{'genome': genome_path, 'intervals': narrowpeak_path}],
            [{'column': 0}], targets_are_classes=True, endless=endless)
        assert seq_collection.idx_to_class_mapping == {0: "chr1", 1: "chr2"}, seq_collection.idx_to_class_mapping
        assert seq_collection.class_to_idx_mapping == {"chr1": 0, "chr2": 1}, seq_collection.class_to_idx_mapping
        expected_values = [0, 0, 1]
        target_values = [val for _, val in islice(seq_collection, 3)]
        assert expected_values == target_values, target_values 

    # With fixed target and bed target
    seq_collection = SequenceCollection(
        [{'genome': genome_path, 'intervals': narrowpeak_path},
         {'genome': genome_path, 'intervals': narrowpeak_path}],
        [{'column': 6}, 91], targets_are_classes=True, endless=False)
    assert seq_collection.idx_to_class_mapping == {0: 91, 1: 182}, seq_collection.idx_to_class_mapping
    assert seq_collection.class_to_idx_mapping == {91: 0, 182: 1}, seq_collection.class_to_idx_mapping
    # First 3 values are taken from bed column, second 3 values are taken from fixed target
    expected_values = [1, 0, 1, 0, 0, 0]
    target_values = [val for _, val in islice(seq_collection, 6)]
    assert expected_values == target_values, target_values

    # With fixed target (str) and bed target
    seq_collection = SequenceCollection(
        [{'genome': genome_path, 'intervals': narrowpeak_path},
         {'genome': genome_path, 'intervals': narrowpeak_path}],
        [{'column': 0}, "chr1"], targets_are_classes=True, endless=False)
    assert seq_collection.idx_to_class_mapping == {0: "chr1", 1: "chr2"}, seq_collection.idx_to_class_mapping
    assert seq_collection.class_to_idx_mapping == {"chr1": 0, "chr2": 1}, seq_collection.class_to_idx_mapping
    # First 3 values are taken from bed column, second 3 values are taken from fixed target
    expected_values = [0, 0, 1, 0, 0, 0]
    target_values = [val for _, val in islice(seq_collection, 6)]
    assert expected_values == target_values, target_values

    #####################
    # Test source sampling

    seq_collection = SequenceCollection(
        [{'genome': genome_path, 'intervals': mouse_interval_path_neg},
        {'genome': genome_path, 'intervals': mouse_interval_path_pos}],
        [0, 1], targets_are_classes=True, endless=True)
    num_examples = 10000
    source_freqs = seq_collection.source_freqs['source_freqs']
    # Count number of examples from each source
    labels = np.array([label for _, label in islice(seq_collection, num_examples)])
    freqs = [np.sum(labels == i) / num_examples for i in [0, 1]]
    for i in [0, 1]:
        assert np.allclose(freqs[i], source_freqs[i], rtol=0.05)

if __name__ == '__main__':
    test_bedsource()
    test_sequence_collection()
    