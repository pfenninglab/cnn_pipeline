#!/bin/bash
#
# Usage: bash setup.sh <cluster_name>, where <cluster_name> is "lane" or "bridges"

# Check arguments
cluster_name=$1
if [ "$cluster_name" = "lane" ]; then
	PARTITION_GPU="pfen3"
elif [ "$cluster_name" = "bridges" ]; then
	PARTITION_GPU="GPU-shared"
else
	echo "Invalid cluster name: $cluster_name"
	echo "Usage: bash setup.sh <cluster_name>, where <cluster_name> is \"lane\" or \"bridges\""
	exit 1
fi

# Launch sweep
sbatch -p $PARTITION_GPU scripts/setup_main.sb ${cluster_name}
