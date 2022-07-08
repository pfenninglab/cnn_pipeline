#!/bin/bash
#
# Usage: bash start_agents.sh <num_agents> <throttle> <sweep_id>

# Activate environment
source activate keras2-tf27

# Check environment variables
if [[ -z $PARTITION_GPU ]]
then
    echo "Error: \$PARTITION_GPU is not set."
    exit 1
fi

# Check arguments
num_agents=$1
throttle=$2
sweep_id=$3

if [ -z $num_agents ] || [ -z $throttle ] || [ -z $sweep_id ];
then
    echo "Error: Missing arguments"
    echo "Usage: bash start_agents.sh <num_agents> <throttle> <sweep_id>"
    exit 1
fi

# Launch sweep
sbatch -p $PARTITION_GPU --array=1-${num_agents}%${throttle} scripts/start_agent_main.sb ${sweep_id}