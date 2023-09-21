#!/bin/bash
#SBATCH --job-name=wavegan-fid
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000,t4v2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=50GB
#SBATCH --exclude=gpu109

dataset=$1
num=$2

python main_metric.py --gpu 0 --dataset $dataset --num $num \
--name results/${dataset}_sept18 \
--real_dir fids/${dataset}_${num}/real --ckpt ${dataset}_checkpoint.pt \
--fake_dir fids/${dataset}_${num}/fake

