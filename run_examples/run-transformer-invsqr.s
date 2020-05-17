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

cd /scratch/as12152/code/Thesis/transformer
bash run-transformer-invsqr.sh ang 1 >> train_log-transformer-invsqr-ang.out