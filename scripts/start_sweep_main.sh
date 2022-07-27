#!/bin/bash

# If training env isn't already activated, then activate it
if [ "$CONDA_DEFAULT_ENV" != "keras2-tf27" ]; then
	source activate keras2-tf27
fi

wandb login
wandb sweep $1
