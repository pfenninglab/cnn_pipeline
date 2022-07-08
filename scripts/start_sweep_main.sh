#!/bin/bash

source activate keras2-tf27

wandb login
wandb sweep $1
