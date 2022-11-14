#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 15:00:00
#SBATCH --mem 40G

python -u main.py --learning_rate=0.00006 --decay_end=130000 --epsilon_end=0.01 --train_dqn --test_n=99 --n_episodes=700000 --buffer_size=200000 > results/final.out