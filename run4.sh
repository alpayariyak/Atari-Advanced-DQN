#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -t 1:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --n_episodes=2000000 --decay_end=200000 --optimize_interval=4 --target_update_interval=1000 --evaluate_interval=10000 --buffer_size=1000 --learning_rate=0.0005 > test6.out
