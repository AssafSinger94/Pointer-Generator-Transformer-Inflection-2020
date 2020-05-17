#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=as12152
#SBATCH --mail-user=as12152@nyu.edu
#SBATCH --output=slurm_%j.out


. ~/.bashrc

conda activate ssl-framework

lang=$1
patience=$2
model_copy=$3

cd /scratch/as12152/code/Thesis/transformer
bash run-pointer_generator-invsqr.sh $lang $patience $model_copy >> train_log-pointer_generator-$lang-base.out
bash run-pointer_generator-augmented.sh $lang $patience $model_copy >> train_log-pointer_generator-$lang-augmented.out