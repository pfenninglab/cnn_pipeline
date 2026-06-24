#!/bin/bash
# Usage: bash find_model.sh <wandb_dir> <model ID>
# e.g. bash find_model.sh 80e77oco
# You can get the model ID from the URL in wandb:
# https://wandb.ai/<wandb username>/<project name>/runs/80e77oco

find $1 -wholename *$2*/files/model-best.h5
