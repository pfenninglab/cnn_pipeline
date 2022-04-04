#!/bin/bash

#SBATCH -p pool1
#SBATCH -n 1
#SBATCH -o /home/%u/outputs/out.txt
#SBATCH -e /home/%u/outputs/err.txt

# update conda
conda update -n base -c defaults conda
# create keras2 environment
conda env create -f=keras2.yml -n keras2