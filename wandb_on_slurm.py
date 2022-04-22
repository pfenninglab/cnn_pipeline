"""wandb_on_slurm.py: Launch `wandb` sweep controller and agents with slurm.

Adapted from https://github.com/elyall/wandb_on_slurm
"""

import wandb
import subprocess
import yaml
import os
import json


# Gather nodes allocated to current slurm job
result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]

def run(args):

    wandb.init(project=args.project, mode="disabled")
    
    with open(args.sweep_config) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = args.train_file

    sweep_id = wandb.sweep(config_dict, project=args.project)
    
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
    parser.add_argument('-train_file', type=str, required=True)
    parser.add_argument('-project', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    run(get_args())