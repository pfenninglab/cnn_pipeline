"""sweep.py: Launch `wandb` sweep controller and agents on slurm.
NOTE: This script should be run via sbatch. It is not guaranteed 
to work if you simply run it with python.

Usage:
sbatch sweep.sb

Adapted from https://github.com/elyall/wandb_on_slurm
"""

import wandb

import subprocess
import yaml
import os
import json


def run(args):
    with open(args.sweep_config) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    project = config_dict['project']
    wandb.init(project=project, mode="disabled")
    sweep_id = wandb.sweep(config_dict, project=project)

    # Gather nodes allocated to current slurm job
    result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
    node_list = result.stdout.decode('utf-8').split('\n')[:-1]
    
    sp = []
    for node in node_list:
        sp.append(subprocess.Popen(['srun',
                        '--nodes=1',
                        '--ntasks=1',
                        '-w',
                        node,
                        'start-agent.sh',
                        sweep_id]))
    exit_codes = [p.wait() for p in sp] # wait for processes to finish
    return exit_codes 

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sweep_config', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    run(get_args())