#!/bin/bash
#
# Usage: bash setup.sh <cluster_name>, where <cluster_name> is "lane" or "bridges"

# Check arguments
cluster_name=$1
if [ "$cluster_name" = "lane" ]; then
	PARTITION_GPU="pfen3 --gres gpu:1 -n 1 --mem 16000"
elif [ "$cluster_name" = "bridges" ]; then
	PARTITION_GPU="GPU-shared --gpus=v100-32:1"
else
	echo "Invalid cluster name: $cluster_name"
	echo "Usage: bash setup.sh <cluster_name>, where <cluster_name> is \"lane\" or \"bridges\""
	exit 1
fi

# Update environment variables in  ~/.bashrc
# Put this repo on your $PYTHONPATH so you can access cnn pipe modules
echo "# CNN Pipeline environment variables" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$PWD" >> ~/.bashrc

# Run setup on a GPU cluster, so that conda knows to install the gpu version of tensorflow
sbatch -p $PARTITION_GPU scripts/setup_main.sb ${cluster_name}
