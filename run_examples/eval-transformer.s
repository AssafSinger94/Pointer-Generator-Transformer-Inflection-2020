#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=as12152
#SBATCH --mail-user=as12152@nyu.edu
#SBATCH --output=slurm_%j.out


. ~/.bashrc

conda activate ml-framework

cd /scratch/as12152/code/Thesis/transformer
python evaluate.py --model "checkpoints/model_50.pth" --test-file "english-dev" --vocab-file "english-train-medium-vocab" >> transformer-test.out