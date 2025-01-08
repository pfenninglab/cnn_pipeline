#!/bin/bash
#
# Usage: bash setup.sh <cluster_name>, where <cluster_name> is "lane" or "bridges"

# Check arguments
cluster_name=$1
if [ "$cluster_name" = "lane" ]; then
	SBATCH_ARGS="-p pfen3 --gres gpu:1 -n 1 --mem 16000"
elif [ "$cluster_name" = "bridges" ]; then
	SBATCH_ARGS="-p GPU-shared -N 1 --gpus=v100-32:1 -t 08:00:00"
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
sbatch ${SBATCH_ARGS} scripts/setup_main.sb ${cluster_name}
