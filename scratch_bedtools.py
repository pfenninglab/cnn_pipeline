import pybedtools
import wandb
from itertools import islice

wandb.init(config='config-base.yaml', mode='disabled')
regions = pybedtools.BedTool(wandb.config.regions[0]['regions'])
for region in islice(regions, 2):
	print(get_interval_seq(region.chrom, region.start, region.stop, wandb.config.regions[0]['reference']))
	print("\n")

def get_interval_seq(chrom, start, stop, genome_file):
	# .bed file uses 0-based, [start, stop) numbering
	# BedTool.seq() assumes 1-based, inclusive numbering
	# so need to add 1 to start, and keep stop the same
	loc = f"{chrom}:{start + 1}-{stop}"
	return pybedtools.BedTool.seq(loc, genome_file)

def get_intervals(bed_file_path):
	"""Get pybedtools.BedTool object from .bed or .narrowPeak file"""
	with open(bed_file_path, "r") as f:
		intervals = pybedtools.BedTool(f.read(), from_string=True)
	return intervals


In [1]: import wandb, pybedtools

In [2]: wandb.init(config='config-base.yaml', mode='disabled')
Failed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.
Out[2]: 

# actual start of sequence
# tail -200 /projects/pfenninggroup/machineLearningForComputationalBiology/halLiftover_chains/data/raw_data/2bit/fasta/Mus_musculus.fa | cut -c-50
# >chr4_JH584295_random
# GGCTGAGCGGTGACATCATGGGCGGCGGGGTCCCAGACAGGAAGTGGGCG

# from BedTool.sequence()

In [3]: a = pybedtools.BedTool('chr4_JH584295_random 1 5', from_string=True)

In [4]: a = a.sequence(fi=wandb.config.regions[0]['reference'])

In [5]: print(open(a.seqfn).read())
>chr4_JH584295_random:1-5
GCTG

# from BedTool.seq()

In [6]: pybedtools.BedTool.seq('chr4_JH584295_random:1-5', wandb.config.regions[0]['reference'])
Out[6]: 'GGCTG'

# from BedTool.sequence() with a file
# printf "chr4_JH584295_random\t1\t5" > tmp.bed

In [7]: a = pybedtools.BedTool('tmp.bed')

In [8]: a = a.sequence(fi=wandb.config.regions[0]['reference'])

In [9]: print(open(a.seqfn).read())
>chr4_JH584295_random:1-5
GCTG