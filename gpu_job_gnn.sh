#!/bin/bash
#SBATCH --job-name=gnn_6      # 
#SBATCH --output=gpu_job_gnn_6.log  # 
#SBATCH --partition=gpu            # 
#SBATCH --gres=gpu:1                   # 
#SBATCH --cpus-per-task=16              #
#SBATCH --mem=64G                       # 
#SBATCH --time=08:00:00                 # 





source ~/.bashrc
conda activate gnn_env


python ~/finger_gnn/gnn.py
