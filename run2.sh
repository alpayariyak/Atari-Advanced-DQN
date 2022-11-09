#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -t 30:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --n_episodes=1000000 --decay_end=50000 --optimize_interval=4 --target_update_interval=5000 --evaluate_interval=10000 --buffer_size=10000 --learning_rate=0.0005 > test4.out
