#!/bin/bash
#SBATCH --job-name=rand
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --partition=rtx6000,t4v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=50GB

dataset=$1
num=$2
eval_backbone=${3:-'inception'}
invert=${4:-0}

argstring="--gpu 0 --dataset $dataset --num $num \
--eval_backbone $eval_backbone \
--real_dir fids/random_${dataset}_${num}/real \
--fake_dir fids/random_${dataset}_${num}/fake"

if [ $invert -eq 1 ]
then
    argstring="${argstring} --invert_rgb"
fi

python random_baseline.py $argstring
