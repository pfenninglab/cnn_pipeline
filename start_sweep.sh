#!/bin/bash
#
# Usage: bash start_sweep.sh <path to sweep config>

# Check environment variables
if [[ -z $PARTITION_NO_GPU ]]
then
    echo "Error: \$PARTITION_NO_GPU is not set."
    exit 1
fi

# Check arguments
sweep_config_path=$1

if [ -z $sweep_config_path ];
then
    echo "Error: Missing arguments"
    echo "Usage: bash start_sweep.sh <path to sweep config>"
    exit 1
fi

srun -n 1 -p $PARTITION_NO_GPU --pty scripts/start_sweep_main.sh $sweep_config_path