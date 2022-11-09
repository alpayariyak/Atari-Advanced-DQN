#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 0:03:00
#SBATCH --mem 8G

python -u main.py --test_dqn --checkpoint_name=150000 > test_dqn.out
