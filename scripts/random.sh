#!/bin/bash
#SBATCH --job-name=rand
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=50GB

dataset=$1
num=$2

python random_baseline.py --gpu 0 --dataset $dataset --num $num \
--real_dir fids/random_${dataset}_${num}/real \
--fake_dir fids/random_${dataset}_${num}/fake

