#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
#SBATCH --mem 8G

python -u main.py --learning_rate=0.001 --epsilon_end=0.01 --train_dqn --test_n=9 --n_episodes=5000000 --buffer_size=100000 --load_checkpoint=5 > results/test9.out