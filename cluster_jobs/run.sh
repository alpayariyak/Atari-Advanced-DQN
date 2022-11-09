#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -t 01:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --n_episodes=300000 > test1.out
