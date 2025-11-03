#!/bin/bash
#
# Usage: bash train.sh <path to config>

# If training env isn't already activated, then activate it
if [ "$CONDA_DEFAULT_ENV" != "keras2-tf27" ]; then
        conda activate keras2-tf27
fi

# Set SLURM variables as in reference
PARTITION_GPU="pfen3"
GRES="gpu:1"
NUM_TASKS=1
MEM=64000
TIME_LIMIT="3-00:00:00"
JOB_NAME="cnn_pipeline_interactive"

# Check arguments
config_path=$1

if [ -z "$config_path" ]; then
    echo "Error: Missing arguments"
    echo "Usage: bash train.sh <path to config>"
    exit 1
fi

sbatch -p $PARTITION_GPU \
       --gres $GRES \
       -n $NUM_TASKS \
       --mem $MEM \
       --time $TIME_LIMIT \
       --job-name="$JOB_NAME" \
       scripts/train_main.sb $config_path
