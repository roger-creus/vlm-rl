#!/bin/bash
#SBATCH --partition=lab-real
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=140:00:00
#SBATCH -o /network/scratch/r/roger.creus-castanyer/slurm-%j.out
#SBATCH --exclude=cn-k004,cn-j001

# 1. Load your environment
module load anaconda/3
conda activate /home/mila/r/roger.creus-castanyer/miniconda3/envs/clean-llm-rl

accelerate launch --config_file=default_config.yaml src/train_shared_actor_critic.py --track