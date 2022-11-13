#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --n_episodes=5000000 --decay_end=80000 --epsilon_end=0.1 --optimize_interval=4 --target_update_interval=5000 --evaluate_interval=10000 --buffer_size=400000 --initialize_weights=False --clip_grad=True --learning_rate=0.001 > test6.out