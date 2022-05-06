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
        ("chr1", None, 182, 9.2103)
    ]

    for _ in range(2):
        # First iteration: tests bed_source before refresh
        # Second iteration: tests bed_source after refresh
        for expected_values, (bed_seq, bed_values) in zip(expected_bed_values, bed_source):
            assert bed_seq.shape == bed_source.seq_shape
            assert bed_values == expected_values

    return bed_source

def test_sequence_collection():
    fa_path = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.fa"
    genome_path = "/projects/pfenninggroup/machineLearningForComputationalBiology/halLiftover_chains/data/raw_data/2bit/fasta/Mus_musculus.fa"
    mouse_interval_path = "/projects/pfenninggroup/mouseCxStr/NeuronSubtypeATAC/Zoonomia_CNN/mouse_SST/FinalModelData/mouse_SST_pos_VAL.bed"
    narrowpeak_path = "../example_files/example.narrowPeak"

    seq_collection_A = SequenceCollection([fa_path], [1], True, endless=True)
    seq_collection_B = SequenceCollection([{'genome_file': genome_path, 'intervals': mouse_interval_path}], [1], True, endless=True)

    assert len(seq_collection_A) == len(seq_collection_B)
    assert seq_collection_A.num_classes == seq_collection_B.num_classes
    for (seq_A, label_A), (seq_B, label_B) in islice(zip(seq_collection_A, seq_collection_B), 10):
        assert np.all(seq_A == seq_B)
        assert label_A == label_B
    for _ in islice(zip(seq_collection_A, seq_collection_B), len(seq_collection_A)):
        pass
    for (seq_A, label_A), (seq_B, label_B) in islice(zip(seq_collection_A, seq_collection_B), 10):
        assert np.all(seq_A == seq_B)
        assert label_A == label_B

    # TODO test multiple files per source
    # TODO test mixed file types per source
    # TODO test label extracted from bed column
    # TODO test class sampling
    # TODO test regression target from bed column

    print('done')


if __name__ == '__main__':
    test_sequence_collection()
    test_bedsource()