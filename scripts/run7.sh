#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
#SBATCH --mem 8G

python -u main.py --train_dqn --test_n=7 --n_episodes=5000000 --buffer_size=100000 --load_checkpoint=5 --epsilon_end=0.01 > results/test7.out