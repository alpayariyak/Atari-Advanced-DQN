#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem 20G

python -u main.py --learning_rate=0.0005 --epsilon_end=0.01 --train_dqn --test_n=14 --n_episodes=5000000 --buffer_size=300000 --load_checkpoint=10 > results/test14.out