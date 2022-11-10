#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --n_episodes=30000 --decay_end=10000 --epsilon_end=0.025 --optimize_interval=4 --target_update_interval=5000 --evaluate_interval=10000 --buffer_size=50000 > test11.out