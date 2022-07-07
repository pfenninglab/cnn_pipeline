#!/bin/bash

wandb login
wandb sweep $1
