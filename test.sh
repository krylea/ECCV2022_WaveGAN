#!/bin/bash
#SBATCH --job-name=waveganvggface
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --nodes=1  # number of nodes
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --gres=gpu:4  # number of gpus per node
#SBATCH --partition=t4v2
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=100GB

python main_metric.py
