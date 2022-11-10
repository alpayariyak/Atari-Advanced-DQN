#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --n_episodes=2000000 --decay_end=150000 --optimize_interval=4 --target_update_interval=10000 --evaluate_interval=10000 --buffer_size=10000 --learning_rate=0.001 > test5.out
