#!/bin/bash
# start-agent.sh: Launch `wandb` sweep agent.
# Adapted from https://github.com/elyall/wandb_on_slurm
# NOTE: This script is run from sweep.py. It is not meant to be run independently.

wandb agent $1