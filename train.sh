#!/bin/bash
#
# Usage: bash train.sh <path to config>

# Activate environment
source activate keras2-tf27

# Check environment variables
if [[ -z $PARTITION_GPU ]]
then
    echo "Error: \$PARTITION_GPU is not set."
    exit 1
fi

# Check arguments
config_path=$1

if [ -z $config_path ];
then
    echo "Error: Missing arguments"
    echo "Usage: bash train.sh <path to config>"
    exit 1
fi

export PARTITION_GPU="GPU-shared"

sbatch -p $PARTITION_GPU scripts/train_main.sb $config_path #-w compute-4-20 
