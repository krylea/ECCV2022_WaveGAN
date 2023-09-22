#!/bin/bash
#SBATCH --job-name=waveganfid
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
eval_backbone=${3:-'inception'}
invert=${4:-0}
random=${5:-0}

name="${dataset}_${num}"
if [ $random -eq 1 ]
then
    name="random_${name}"
fi

argstring=" --dataset $dataset --num $num \
--eval_backbone $eval_backbone \
--real_dir fids/${name}/real \
--fake_dir fids/${name}/fake"

if [ $invert -eq 1 ]
then
    argstring="${argstring} --invert_rgb"
fi

if [ $random -eq 1 ]
then
    argstring="${argstring} --prefix random"
fi

python calc_fid.py $argstring

