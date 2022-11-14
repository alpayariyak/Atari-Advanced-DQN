#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem 40G

python -u main.py --learning_rate=0.0005 --epsilon_end=0.01 --train_dqn --test_n=15 --n_episodes=2000000 --buffer_size=1000000 --load_checkpoint=10 > results/test15.out