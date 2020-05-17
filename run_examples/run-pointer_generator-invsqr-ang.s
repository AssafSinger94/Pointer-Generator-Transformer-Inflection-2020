#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:20:00
#SBATCH --mem=16GB
#SBATCH --job-name=as12152
#SBATCH --mail-user=as12152@nyu.edu
#SBATCH --output=slurm_%j.out


. ~/.bashrc

conda activate ssl-framework

cd /scratch/as12152/code/Thesis/transformer
bash run-pointer_generator-invsqr.sh ang 10 1 >> train_log-pointer_generator-invsqr-ang.out